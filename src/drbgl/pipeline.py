"""dataset1 / dataset2 的自适应双分支排序流水线。

核心函数::

    read_dataset(data_root, dataset)                   读取并校验原始 CSV
    run_dataset1(data_root, output_dir)                dataset1：depop-v9 直接排序
    run_dataset2(data_root, output_dir, model_dir, cfg) dataset2：293 维特征 LightGBM 门控
    package_submission(output_dir)                     打包 submission.zip

分数约定：专家与门控输出均为 ``[Q, 100]`` 的 float32 矩阵，数值越大越靠前；
最终 CSV 的每一行是 100 个候选的序数名次除以 100（取值 0.01~1.00）。
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import ThreadpoolController, threadpool_limits

from .baselines.depop_v3 import _time_bins
from .baselines.depop_v9 import V9_CONFIG, score_depop_v9_components, score_v9
from .experts.extra_expert_features import extra_expert_features
from .experts.lightgcn import (
    CompactBipartiteGraph,
    build_compact_graph,
    lightgcn_candidate_scores,
    propagate_lightgcn,
    train_bpr_embeddings,
)
from .experts.listwise_gate import (
    ListwiseContext,
    listwise_features,
    transductive_exposure_scores,
)
from .experts.low_rank import average_row_ranks, normalized_svd_scores
from .experts.multiscale_exposure import (
    MULTISCALE_EXPOSURE_FEATURE_NAMES,
    multiscale_exposure_features,
)
from .experts.sequence_graph import SequenceGraphConfig, SequenceTransitionExpert
from .experts.structural_features import (
    STRUCTURAL_FEATURE_NAMES,
    StructuralFeatureIndex,
    transductive_history_features,
)
from .metrics import ordinal_rank_scores


CANDIDATE_COLUMNS = [f"c{index}" for index in range(1, 101)]
FEATURE_COUNT = 293
CLUSTERS = (8, 16, 32)
TIME_BINS = (4, 16)
LOG_MULTISCALE = tuple(
    index
    for index, name in enumerate(MULTISCALE_EXPOSURE_FEATURE_NAMES)
    if name.endswith("_log")
)
EXTRA_EXPERT_NAMES = (
    "lightgcn_l1",
    "lightgcn_s2027_l0",
    "lightgcn_s2027_l1",
    "lightgcn_s2027_l2",
    "lightgcn_d128_l0",
    "lightgcn_d128_l1",
    "lightgcn_d128_l2",
    "community_k8",
    "community_k16",
    "community_k32",
    "community_k8_t4",
    "community_k8_t16",
    "community_k16_t4",
    "community_k16_t16",
    "community_k32_t4",
    "community_k32_t16",
)
REFERENCE = {
    "dataset1.csv": "455716161d07b7ff40c0207a8d40e65bd576ebdd0fa33616875690a24994109e",
    "dataset2.csv": "6529919322140e2a4365804d281d1a085e233fbd735ee73328687e6d26d4bd6a",
}


@dataclass(frozen=True)
class RuntimeConfig:
    """流水线运行时配置。

    Attributes:
        seed: 全局随机种子（Python / NumPy / Jittor）。
        device: Jittor 设备，``auto`` / ``cpu`` / ``cuda``。
        score_backend: LightGCN 打分后端，``numpy``（默认，位级稳定）或 ``jittor``。
        embedding_dim: 默认 embedding 维度。
        epochs / batch_size / learning_rate: BPR-LightGCN 训练超参，仅在重训时生效。
        threads: 候选特征构建阶段的原生线程池上限。
        community_dim: 社区专家使用的 TruncatedSVD 维度。
        retrain_embeddings: 是否用 Jittor 重训 BPR embedding。
        rebuild_base_gate: 是否重算 base-gate 冷启动名次（不加载 npz 资产）。
        base_gate_openmp_threads / base_gate_blas_threads /
            base_gate_sklearn_threads: base-gate 阶段的线程池布局。
            这三个值只影响 base-gate 冷启动名次的数值细节，仅在
            ``rebuild_base_gate=True`` 时生效。
    """

    seed: int = 20240501
    device: str = "auto"
    score_backend: str = "numpy"
    embedding_dim: int = 64
    epochs: int = 5
    batch_size: int = 65536
    learning_rate: float = 0.02
    threads: int = 16
    community_dim: int = 32
    retrain_embeddings: bool = False
    rebuild_base_gate: bool = False
    base_gate_openmp_threads: int = 191
    base_gate_blas_threads: int = 64
    base_gate_sklearn_threads: int = 8


@contextmanager
def _base_gate_thread_layout(config: RuntimeConfig):
    """Match the native thread pools used to build the base-gate features."""
    changed = []
    for library in ThreadpoolController().lib_controllers:
        target = None
        if library.user_api == "openmp":
            target = (
                config.base_gate_sklearn_threads
                if "scikit_learn.libs" in library.filepath
                else config.base_gate_openmp_threads
            )
        elif library.user_api == "blas":
            target = config.base_gate_blas_threads
        if target is not None:
            changed.append((library, library.get_num_threads()))
            library.set_num_threads(target)
    try:
        yield
    finally:
        for library, original in reversed(changed):
            library.set_num_threads(original)


def _load_base_cold_scores(path: Path, candidates: np.ndarray) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing base-gate cold-slot asset: {path}")
    candidate_digest = hashlib.sha256(candidates.tobytes()).hexdigest()
    with np.load(path, allow_pickle=False) as state:
        rank_units = state["rank_units"]
        expected_digest = str(state["candidates_sha256"].item())
    if candidate_digest != expected_digest:
        raise ValueError(
            "base-gate cold-slot asset does not match dataset2 test candidates; "
            "use --rebuild-base-gate for a different candidate set"
        )
    if rank_units.shape != candidates.shape or rank_units.dtype != np.uint8:
        raise ValueError(f"invalid base-gate cold-slot asset shape: {rank_units.shape}")
    expected = np.arange(1, candidates.shape[1] + 1, dtype=np.uint8)
    if not np.all(np.sort(rank_units, axis=1) == expected[None, :]):
        raise ValueError("base-gate cold-slot rows must each be a rank permutation")
    return rank_units.astype(np.float32) / 100.0


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def read_dataset(data_root: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取并校验单个数据集的 train/test CSV。

    Args:
        data_root: 数据根目录，需含 ``<dataset>/train.csv`` 与 ``<dataset>/test.csv``。
        dataset: 数据集名，如 ``dataset1``。

    Returns:
        ``(train, test)``。train 含 ``src,dst,time``；test 含 ``src,time,c1..c100``。

    Raises:
        FileNotFoundError: 文件缺失，消息中给出期望路径。
        ValueError: 列缺失、表为空或测试时间未严格晚于训练时间。
    """
    train_path = data_root / dataset / "train.csv"
    test_path = data_root / dataset / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"missing {dataset} data; expected {train_path} and {test_path}"
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    missing_train = {"src", "dst", "time"} - set(train.columns)
    missing_test = {"src", "time", *CANDIDATE_COLUMNS} - set(test.columns)
    if missing_train or missing_test:
        raise ValueError(
            f"invalid {dataset} columns: train missing={sorted(missing_train)}, "
            f"test missing={sorted(missing_test)}"
        )
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"{dataset} train/test must be non-empty")
    if int(train["time"].max()) >= int(test["time"].min()):
        raise ValueError("test time must be strictly after all training events")
    return train, test


def _load_embeddings(
    graph: CompactBipartiteGraph,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing pretrained LightGCN asset: {path}")
    with np.load(path, allow_pickle=False) as state:
        graph_users = state["graph_users"]
        graph_items = state["graph_items"]
        users = state["user_embedding"].astype(np.float32, copy=False)
        items = state["item_embedding"].astype(np.float32, copy=False)
        losses = state["losses"].astype(np.float64).tolist()
    if not np.array_equal(graph.users, graph_users):
        raise ValueError(f"user mapping differs from checkpoint {path.name}")
    if not np.array_equal(graph.items, graph_items):
        raise ValueError(f"item mapping differs from checkpoint {path.name}")
    return users, items, losses


def _embedding_views(
    graph: CompactBipartiteGraph,
    config: RuntimeConfig,
    model_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, object]]]:
    """按 (维度, 种子) 组合取到三份 BPR-LightGCN embedding，并记录来源清单。

    Args:
        graph: 紧凑二部图，提供 user/item 映射。
        config: 运行时配置；``retrain_embeddings=True`` 时用 Jittor 重训并落盘。
        model_dir: 预训练检查点目录。
        output_dir: 重训结果的输出目录。

    Returns:
        ``(views, manifest)``；``views`` 形如 ``{"d64_s42": (user_emb, item_emb)}``，
        embedding 形状分别为 ``[user 数, 维度]`` 与 ``[item 数, 维度]``。
    """
    definitions = (
        ("d64_s42", 64, 42, "lightgcn_d64_s42.npz"),
        ("d64_s2027", 64, 2027, "lightgcn_d64_s2027.npz"),
        ("d128_s42", 128, 42, "lightgcn_d128_s42.npz"),
    )
    trained_dir = output_dir / "models"
    if config.retrain_embeddings:
        trained_dir.mkdir(parents=True, exist_ok=True)
    views: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    manifest: list[dict[str, object]] = []
    for name, dimension, seed, filename in definitions:
        if config.retrain_embeddings:
            print(f"training Jittor LightGCN {name}", flush=True)
            users, items, losses = train_bpr_embeddings(
                graph,
                embedding_dim=dimension,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                seed=seed,
                threads=config.threads,
                device=config.device,
            )
            checkpoint = trained_dir / filename
            np.savez_compressed(
                checkpoint,
                graph_users=graph.users,
                graph_items=graph.items,
                user_embedding=users,
                item_embedding=items,
                losses=np.asarray(losses, dtype=np.float64),
            )
            source = checkpoint
        else:
            source = model_dir / filename
            users, items, losses = _load_embeddings(graph, source)
        views[name] = (users, items)
        manifest.append(
            {
                "name": name,
                "dimension": dimension,
                "seed": seed,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "losses": losses,
                "checkpoint": str(source),
                "checkpoint_sha256": sha256_file(source),
                "trained_with_jittor": config.retrain_embeddings,
            }
        )
    return views, manifest


def _lightgcn_score(
    graph: CompactBipartiteGraph,
    embeddings: tuple[np.ndarray, np.ndarray],
    test: pd.DataFrame,
    layers: int,
    config: RuntimeConfig,
) -> np.ndarray:
    users, items = propagate_lightgcn(graph, *embeddings, layers=layers)
    return lightgcn_candidate_scores(
        graph,
        users,
        items,
        test,
        CANDIDATE_COLUMNS,
        backend=config.score_backend,
        device=config.device,
    )


def _base_experts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    graph: CompactBipartiteGraph,
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
    config: RuntimeConfig,
) -> list[np.ndarray]:
    """构造 15 个基础专家，每个返回 ``[Q, 100]`` 打分矩阵。

    Args:
        train: 训练交互表，列 ``src,dst,time``。
        test: 测试查询表，列 ``src,time,c1..c100``。
        graph: 由训练表构建的紧凑二部图。
        embeddings: ``_embedding_views`` 的返回值。
        config: 运行时配置。
    """
    print("building V5-V9 and low-rank experts", flush=True)
    v9_experts, components = score_depop_v9_components(train, test)
    temporal120 = components["temporal_120"]
    v8 = v9_experts["v7"] + 0.5 * average_row_ranks(temporal120)

    temporal90 = normalized_svd_scores(
        train,
        test,
        CANDIDATE_COLUMNS,
        rank=64,
        degree_power=0.5,
        n_iter=V9_CONFIG["n_iter"],
        seed=V9_CONFIG["seed"],
        half_life_days=90.0,
    )
    temporal120_rank128 = normalized_svd_scores(
        train,
        test,
        CANDIDATE_COLUMNS,
        rank=128,
        degree_power=0.5,
        n_iter=V9_CONFIG["n_iter"],
        seed=V9_CONFIG["seed"],
        half_life_days=120.0,
    )
    sequence = SequenceTransitionExpert.fit(
        train, int(test["time"].min()), SequenceGraphConfig()
    ).score(test, CANDIDATE_COLUMNS)
    lightgcn_l0 = _lightgcn_score(
        graph, embeddings["d64_s42"], test, 0, config
    )
    lightgcn_l2 = _lightgcn_score(
        graph, embeddings["d64_s42"], test, 2, config
    )
    experts = [
        components["v5"],
        v9_experts["v6"],
        v9_experts["v7"],
        v8,
        v9_experts["v9"],
        lightgcn_l0,
        lightgcn_l2,
        sequence,
        components["user_cf"],
        components["static"],
        temporal90,
        temporal120,
        components["temporal_180"],
        components["temporal_365"],
        temporal120_rank128,
    ]
    if len(experts) != 15:
        raise AssertionError("base expert count changed")
    return experts


def _community_experts(
    history: pd.DataFrame,
    test: pd.DataFrame,
    embedding_dim: int,
) -> list[np.ndarray]:
    """社区一致性专家：对 source 做 SVD + KMeans 聚类，统计簇内候选超额票数。

    Args:
        history: 历史交互表。
        test: 测试查询表。
        embedding_dim: TruncatedSVD 维度。

    Returns:
        9 个 ``[Q, 100]`` 矩阵（k=8/16/32 的全局版本 × 时间分桶版本）。
    """
    query_src = test["src"].to_numpy(np.int64)
    query_time = test["time"].to_numpy(np.int64)
    candidates = test[CANDIDATE_COLUMNS].to_numpy(np.int64)
    query_sources = np.unique(query_src)
    items = np.unique(
        np.concatenate([history["dst"].to_numpy(np.int64), candidates.ravel()])
    )
    history_src = history["src"].to_numpy(np.int64)
    source_pos = np.searchsorted(query_sources, history_src)
    source_safe = np.minimum(source_pos, len(query_sources) - 1)
    keep = (source_pos < len(query_sources)) & (
        query_sources[source_safe] == history_src
    )
    item_pos = np.searchsorted(items, history["dst"].to_numpy(np.int64))
    interactions = sparse.csr_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.float32),
            (source_pos[keep], item_pos[keep]),
        ),
        shape=(len(query_sources), len(items)),
    )
    interactions.sum_duplicates()
    interactions.data[:] = 1.0
    embedding = TruncatedSVD(
        embedding_dim, n_iter=7, random_state=44
    ).fit_transform(interactions)
    embedding /= np.maximum(
        np.linalg.norm(embedding, axis=1, keepdims=True), 1e-6
    )
    query_pos = np.searchsorted(query_sources, query_src)
    candidate_pos = np.searchsorted(items, candidates)
    global_scores: dict[int, np.ndarray] = {}
    temporal_scores: dict[tuple[int, int], np.ndarray] = {}
    for cluster_count in CLUSTERS:
        print(f"building community expert k={cluster_count}", flush=True)
        labels = MiniBatchKMeans(
            cluster_count,
            random_state=45,
            n_init=10,
            batch_size=512,
        ).fit_predict(embedding)
        query_labels = labels[query_pos]
        keys = query_labels[:, None] * np.int64(len(items)) + candidate_pos
        counts = np.bincount(
            keys.ravel(), minlength=cluster_count * len(items)
        ).reshape(cluster_count, len(items))
        query_counts = np.bincount(query_labels, minlength=cluster_count)
        expected = 99.0 * query_counts / len(items)
        global_scores[cluster_count] = np.maximum(
            counts[query_labels[:, None], candidate_pos]
            - expected[query_labels, None],
            0.0,
        ).astype(np.float32)
        for bin_count in TIME_BINS:
            local = np.zeros(candidates.shape, dtype=np.float64)
            for shift in (0.0, 1.0 / 3.0, 2.0 / 3.0):
                time_bin = _time_bins(query_time, bin_count, shift)
                actual_bins = int(time_bin.max()) + 1
                group = query_labels * actual_bins + time_bin
                group_count = np.bincount(
                    group, minlength=cluster_count * actual_bins
                )
                local_keys = group[:, None] * np.int64(len(items)) + candidate_pos
                local_counts = np.bincount(
                    local_keys.ravel(),
                    minlength=cluster_count * actual_bins * len(items),
                ).reshape(cluster_count * actual_bins, len(items))
                expected_local = 99.0 * group_count / len(items)
                local += np.maximum(
                    local_counts[group[:, None], candidate_pos]
                    - expected_local[group, None],
                    0.0,
                )
            temporal_scores[(cluster_count, bin_count)] = (
                local / 3.0
            ).astype(np.float32)
    output = [global_scores[value] for value in CLUSTERS]
    for cluster_count in CLUSTERS:
        output.extend(
            temporal_scores[(cluster_count, bin_count)]
            for bin_count in TIME_BINS
        )
    return output


def _source_query_counts(query_src: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(
        query_src, return_inverse=True, return_counts=True
    )
    return counts[inverse].astype(np.float32)


def _preserve_cold_slots(
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    cold: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """合并两个打分域：保留 base 的名次槽位，用 candidate 门控重排热门槽位。

    Args:
        base_scores: base-gate 分数，``[Q, 100]``。
        candidate_scores: candidate-gate 分数，``[Q, 100]``。
        cold: 冷候选掩码，``[Q, 100]``，True 表示该候选无历史。
        candidates: 候选 ID，``[Q, 100]``，用于稳定打破并列。

    Returns:
        融合后的序数名次（0.01~1.00），形状 ``[Q, 100]``。
    """
    base_rank = ordinal_rank_scores(base_scores, candidates)
    challenger_rank = ordinal_rank_scores(candidate_scores, candidates)
    output = base_rank.copy()
    for row in range(len(output)):
        hot = np.flatnonzero(~cold[row])
        if len(hot) < 2:
            continue
        slots = np.sort(base_rank[row, hot])
        order = hot[np.argsort(challenger_rank[row, hot], kind="stable")]
        output[row, order] = slots
    return output


def run_dataset1(data_root: Path, output_dir: Path) -> Path:
    """对 dataset1 打分并写出 CSV（depop-v9 + 序数名次）。

    Args:
        data_root: 数据根目录。
        output_dir: 产物目录，会写出 ``dataset1.csv``、``scores/dataset1_scores.npy``
            与 ``dataset1_manifest.json``。

    Returns:
        ``dataset1.csv`` 的路径，形状 ``[行数, 100]``，取值 0.01~1.00。
    """
    started = time.time()
    train, test = read_dataset(data_root, "dataset1")
    scores, strategy, bipartite, repeat_rate, model_config = score_v9(train, test)
    if strategy != "memory":
        raise ValueError("dataset1 did not select the expected memory strategy")
    score_dir = output_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / "dataset1_scores.npy"
    scores = np.asarray(scores, dtype=np.float32)
    np.save(score_path, scores)
    csv_path = output_dir / "dataset1.csv"
    ranks = ordinal_rank_scores(
        scores, test[CANDIDATE_COLUMNS].to_numpy(np.int64)
    )
    np.savetxt(csv_path, ranks, fmt="%.8f", delimiter=",")
    _write_json(
        output_dir / "dataset1_manifest.json",
        {
            "dataset": "dataset1",
            "strategy": strategy,
            "bipartite": bipartite,
            "holdout_repeat_rate": repeat_rate,
            "config": model_config,
            "rows": len(test),
            "scores_sha256": sha256_file(score_path),
            "csv_sha256": sha256_file(csv_path),
            "reference_csv_sha256": REFERENCE["dataset1.csv"],
            "elapsed_seconds": round(time.time() - started, 2),
        },
    )
    return csv_path


def run_dataset2(
    data_root: Path,
    output_dir: Path,
    model_dir: Path,
    config: RuntimeConfig,
) -> Path:
    """对 dataset2 打分并写出 CSV（293 维特征 LightGBM 门控 + 冷启动槽位保留）。

    Args:
        data_root: 数据根目录。
        output_dir: 产物目录，写出 ``dataset2.csv``、分数 npy 与 manifest。
        model_dir: 检查点目录，需含 ``base_gate.txt``、``candidate_gate.txt``、
            ``base_cold_ranks.npz`` 以及三个 LightGCN embedding（除非重训）。
        config: 运行时配置。

    Returns:
        ``dataset2.csv`` 的路径，形状 ``[行数, 100]``，取值 0.01~1.00。
    """
    started = time.time()
    train, test = read_dataset(data_root, "dataset2")
    graph = build_compact_graph(train, int(test["time"].min()))
    embeddings, embedding_manifest = _embedding_views(
        graph, config, model_dir, output_dir
    )
    candidates = test[CANDIDATE_COLUMNS].to_numpy(np.int64)
    query_src = test["src"].to_numpy(np.int64)
    query_time = test["time"].to_numpy(np.int64)
    rows = np.arange(len(test), dtype=np.int64)

    base_gate_path = model_dir / "base_gate.txt"
    base_rank_path = model_dir / "base_cold_ranks.npz"
    if config.rebuild_base_gate:
        print("rebuilding base-gate cold-slot ranks", flush=True)
        with _base_gate_thread_layout(config):
            base_experts = _base_experts(train, test, graph, embeddings, config)
            base_exposure = transductive_exposure_scores(
                query_src, query_time, candidates
            )
            base_context = ListwiseContext(
                base_experts, base_exposure, query_time
            )
            base_features = listwise_features(base_context, rows)
        if base_features.shape[-1] != 108:
            raise AssertionError("base-gate feature count changed")
        base_gate = lgb.Booster(model_file=str(base_gate_path))
        base_scores = base_gate.predict(base_features.reshape(-1, 108)).reshape(
            len(test), 100
        ).astype(np.float32)
        base_source = "recomputed_from_base_gate"
        del base_experts, base_exposure, base_context, base_features, base_gate
        gc.collect()
    else:
        print("loading base-gate cold-slot ranks", flush=True)
        base_scores = _load_base_cold_scores(base_rank_path, candidates)
        base_source = str(base_rank_path)

    # Keep numerical thread pools fixed for deterministic candidate features.
    print("building fixed-thread candidate features", flush=True)
    with threadpool_limits(limits=config.threads):
        experts = _base_experts(train, test, graph, embeddings, config)
        exposure = transductive_exposure_scores(
            query_src, query_time, candidates
        )
        context = ListwiseContext(experts, exposure, query_time)
        base = listwise_features(context, rows)
    if base.shape[-1] != 108:
        raise AssertionError("candidate base feature count changed")

    lightgcn_extra = [
        _lightgcn_score(graph, embeddings["d64_s42"], test, 1, config),
        _lightgcn_score(graph, embeddings["d64_s2027"], test, 0, config),
        _lightgcn_score(graph, embeddings["d64_s2027"], test, 1, config),
        _lightgcn_score(graph, embeddings["d64_s2027"], test, 2, config),
        _lightgcn_score(graph, embeddings["d128_s42"], test, 0, config),
        _lightgcn_score(graph, embeddings["d128_s42"], test, 1, config),
        _lightgcn_score(graph, embeddings["d128_s42"], test, 2, config),
    ]
    community_extra = _community_experts(train, test, config.community_dim)
    extra_scores = [*lightgcn_extra, *community_extra]
    if len(extra_scores) != len(EXTRA_EXPERT_NAMES):
        raise AssertionError("extra expert order changed")
    extra = extra_expert_features(base, extra_scores)
    print("building structural and multiscale features", flush=True)
    cutoff = int(test["time"].min())
    structural_index = StructuralFeatureIndex.fit(train, query_src, cutoff)
    structural = structural_index.transform(query_src, query_time, candidates)
    interactions = transductive_history_features(
        structural, exposure, _source_query_counts(query_src)
    )
    multiscale = multiscale_exposure_features(
        query_src, query_time, candidates
    )[..., LOG_MULTISCALE]
    features = np.concatenate(
        [base, structural, interactions, extra, multiscale], axis=-1
    ).astype(np.float32, copy=False)
    if features.shape[-1] != FEATURE_COUNT:
        raise AssertionError(
            f"candidate feature count {features.shape[-1]} != {FEATURE_COUNT}"
        )
    candidate_gate_path = model_dir / "candidate_gate.txt"
    candidate_gate = lgb.Booster(model_file=str(candidate_gate_path))
    candidate_scores = candidate_gate.predict(
        features.reshape(-1, FEATURE_COUNT)
    ).reshape(len(test), 100).astype(np.float32)
    cold_col = STRUCTURAL_FEATURE_NAMES.index("is_cold_item")
    final_scores = _preserve_cold_slots(
        base_scores,
        candidate_scores,
        structural[..., cold_col].astype(bool),
        candidates,
    )
    score_dir = output_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    base_score_path = score_dir / "dataset2_base_scores.npy"
    score_path = score_dir / "dataset2_scores.npy"
    np.save(base_score_path, np.asarray(base_scores, dtype=np.float32))
    np.save(score_path, final_scores)
    csv_path = output_dir / "dataset2.csv"
    ranks = ordinal_rank_scores(final_scores, candidates)
    np.savetxt(csv_path, ranks, fmt="%.8f", delimiter=",")
    _write_json(
        output_dir / "dataset2_manifest.json",
        {
            "dataset": "dataset2",
            "scorer": "adaptive_293_feature_gate_with_base_cold_slot_preservation",
            "rows": len(test),
            "feature_count": FEATURE_COUNT,
            "base_experts": 15,
            "extra_experts": list(EXTRA_EXPERT_NAMES),
            "embedding_models": embedding_manifest,
            "jittor_score_backend": config.score_backend,
            "jittor_device": config.device,
            "retrained_embeddings": config.retrain_embeddings,
            "base_gate_sha256": sha256_file(base_gate_path),
            "base_cold_rank_asset_sha256": (
                sha256_file(base_rank_path) if base_rank_path.is_file() else None
            ),
            "base_gate_source": base_source,
            "candidate_gate_sha256": sha256_file(candidate_gate_path),
            "base_openmp_threads": config.base_gate_openmp_threads,
            "base_blas_threads": config.base_gate_blas_threads,
            "base_sklearn_threads": config.base_gate_sklearn_threads,
            "candidate_native_threads": config.threads,
            "seed": config.seed,
            "base_scores_sha256": sha256_file(base_score_path),
            "scores_sha256": sha256_file(score_path),
            "csv_sha256": sha256_file(csv_path),
            "reference_csv_sha256": REFERENCE["dataset2.csv"],
            "elapsed_seconds": round(time.time() - started, 2),
        },
    )
    return csv_path


def package_submission(output_dir: Path) -> Path:
    """把两份 CSV 打包为可直接提交的 zip，并校验形状与取值范围。

    Args:
        output_dir: 产物目录，需已存在 ``dataset1.csv`` 与 ``dataset2.csv``。

    Returns:
        ``submission.zip`` 的路径。
    """
    paths = [output_dir / "dataset1.csv", output_dir / "dataset2.csv"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"cannot package missing result: {path}")
        matrix = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != 100:
            raise ValueError(f"invalid submission matrix {path}: {matrix.shape}")
        if not np.isfinite(matrix).all() or matrix.min() < 0 or matrix.max() > 1:
            raise ValueError(f"invalid values in {path}")
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.name)
    _write_json(
        output_dir / "submission_manifest.json",
        {
            "submission": zip_path.name,
            "sha256": sha256_file(zip_path),
            "members": {path.name: sha256_file(path) for path in paths},
            "reference_members": {
                name: digest for name, digest in REFERENCE.items() if name.endswith(".csv")
            },
            "leaderboard_score": 1.5186,
        },
    )
    return zip_path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
