#!/usr/bin/env python3
"""重训门控入口：从原始数据重建验证候选并训练 base / candidate 两个 LambdaRank 门控。

用法（在仓库根目录执行）::

    python scripts/train.py --config configs/train_gates.yaml
    python scripts/train.py --device cuda --uniform-final 40000

命令行参数优先级高于配置文件。产物写在 output_dir/models 下
（``base_gate.txt``、``candidate_gate.txt``）。
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from drbgl.candidates import build_from_source_time_templates
    from drbgl.config import (
        deep_update,
        load_config,
        require_files,
        save_command,
        save_config,
        set_seed,
        tee_stdout,
    )
    from drbgl.data import load_dataset, rolling_folds
    from drbgl.experts.extra_expert_features import extra_expert_features
    from drbgl.experts.lightgcn import build_compact_graph
    from drbgl.experts.listwise_gate import (
        ListwiseContext,
        listwise_features,
        transductive_exposure_scores,
    )
    from drbgl.experts.multiscale_exposure import multiscale_exposure_features
    from drbgl.experts.structural_features import (
        StructuralFeatureIndex,
        transductive_history_features,
    )
    from drbgl.pipeline import (
        CANDIDATE_COLUMNS,
        FEATURE_COUNT,
        LOG_MULTISCALE,
        RuntimeConfig,
        _base_experts,
        _community_experts,
        _embedding_views,
        _lightgcn_score,
        _source_query_counts,
    )
except ImportError as error:  # 依赖缺失时给出可修复的提示
    raise SystemExit(
        f"import failed: {error}\n"
        "how to fix: pip install -r requirements.txt "
        "(run from the repository root)"
    ) from error


FINAL_GATE_CONFIG = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "label_gain": [0, 1],
    "n_estimators": 900,
    "learning_rate": 0.02,
    "num_leaves": 127,
    "max_depth": 10,
    "min_child_samples": 50,
    "colsample_bytree": 1.0,
    "reg_lambda": 5.0,
    "lambdarank_truncation_level": 10,
}


def uniform_frame(data, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """构造均匀负采样验证集：正例 + 从候选池随机取的 99 个负例。

    Args:
        data: ``load_dataset`` 的返回值。
        seed: 负采样随机种子。
    Returns:
        ``(frame, positive_col)``；frame 列为 ``src,time,c1..c100``，
        正例固定在第 0 列，positive_col 因此全为 0。
    """
    _, fold = rolling_folds(data)
    source = data.src[fold.label_idx]
    event_time = data.time[fold.label_idx]
    truth = data.dst[fold.label_idx]
    pool = np.unique(data.test_candidates)
    rng = np.random.default_rng(seed)
    negatives = np.empty((len(truth), 99), dtype=np.int64)
    for start in range(0, len(truth), 50000):
        stop = min(start + 50000, len(truth))
        block = pool[
            rng.integers(0, len(pool), size=(stop - start, 99))
        ]
        collision = block == truth[start:stop, None]
        while collision.any():
            rows, columns = np.where(collision)
            block[rows, columns] = pool[
                rng.integers(0, len(pool), size=len(rows))
            ]
            collision = block == truth[start:stop, None]
        negatives[start:stop] = block
    candidates = np.column_stack([truth, negatives])
    frame = pd.DataFrame(
        np.column_stack([source, event_time, candidates]),
        columns=["src", "time", *CANDIDATE_COLUMNS],
    )
    return frame, np.zeros(len(frame), dtype=np.int64)


def strict_frame(data, fold, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """构造严格源-时间模板验证集（负例分布与线上测试最接近）。

    Args:
        data: ``load_dataset`` 的返回值。
        fold: 用于取标签的时间折。
        seed: 模板构造随机种子。
    Returns:
        ``(frame, positive_col)``；positive_col 为每行正例所在列号。
    """
    queries = build_from_source_time_templates(
        data.test_src,
        data.test_time,
        data.test_candidates,
        data.src[fold.label_idx],
        data.time[fold.label_idx],
        data.dst[fold.label_idx],
        seed,
    )
    labels = np.asarray(queries.label_row, dtype=np.int64)
    frame = pd.DataFrame(
        np.column_stack(
            [
                data.src[fold.label_idx][labels],
                data.time[fold.label_idx][labels],
                queries.candidates,
            ]
        ),
        columns=["src", "time", *CANDIDATE_COLUMNS],
    )
    return frame, np.asarray(queries.positive_col, dtype=np.int64)


def history_frame(data, indices: np.ndarray) -> pd.DataFrame:
    """把给定行索引切成一个 ``src,dst,time`` 历史表。"""
    return pd.DataFrame(
        {
            "src": data.src[indices],
            "dst": data.dst[indices],
            "time": data.time[indices],
        }
    )


def build_feature_pair(
    history: pd.DataFrame,
    frame: pd.DataFrame,
    runtime: RuntimeConfig,
    model_dir: Path,
    cache_dir: Path,
    embedding_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    """为一个验证集构造 base(108 维) 与 final(293 维) 两组特征。

    Args:
        history: 历史交互表。
        frame: 查询表，列 ``src,time,c1..c100``。
        runtime: 运行时配置。
        model_dir: 预训练检查点目录。
        cache_dir: embedding 相关中间产物的输出目录。
        embedding_cache: 复用上一轮的 embedding，避免重复训练。
    Returns:
        ``(base_features, final_features, embeddings)``，特征形状分别为
        ``[Q, 100, 108]`` 与 ``[Q, 100, 293]``。
    """
    cutoff = int(frame["time"].min())
    graph = build_compact_graph(history, cutoff)
    if embedding_cache is None:
        embeddings, _ = _embedding_views(
            graph, runtime, model_dir, cache_dir
        )
    else:
        embeddings = embedding_cache
    experts = _base_experts(history, frame, graph, embeddings, runtime)
    candidates = frame[CANDIDATE_COLUMNS].to_numpy(np.int64)
    query_src = frame["src"].to_numpy(np.int64)
    query_time = frame["time"].to_numpy(np.int64)
    exposure = transductive_exposure_scores(query_src, query_time, candidates)
    base = listwise_features(
        ListwiseContext(experts, exposure, query_time),
        np.arange(len(frame), dtype=np.int64),
    )
    extra_scores = [
        _lightgcn_score(graph, embeddings["d64_s42"], frame, 1, runtime),
        _lightgcn_score(graph, embeddings["d64_s2027"], frame, 0, runtime),
        _lightgcn_score(graph, embeddings["d64_s2027"], frame, 1, runtime),
        _lightgcn_score(graph, embeddings["d64_s2027"], frame, 2, runtime),
        _lightgcn_score(graph, embeddings["d128_s42"], frame, 0, runtime),
        _lightgcn_score(graph, embeddings["d128_s42"], frame, 1, runtime),
        _lightgcn_score(graph, embeddings["d128_s42"], frame, 2, runtime),
        *_community_experts(history, frame, runtime.community_dim),
    ]
    extra = extra_expert_features(base, extra_scores)
    index = StructuralFeatureIndex.fit(history, query_src, cutoff)
    structural = index.transform(query_src, query_time, candidates)
    interactions = transductive_history_features(
        structural, exposure, _source_query_counts(query_src)
    )
    multiscale = multiscale_exposure_features(
        query_src, query_time, candidates
    )[..., LOG_MULTISCALE]
    final_features = np.concatenate(
        [base, structural, interactions, extra, multiscale], axis=-1
    ).astype(np.float32, copy=False)
    if base.shape[-1] != 108 or final_features.shape[-1] != FEATURE_COUNT:
        raise AssertionError("gate feature layout changed")
    return base, final_features, embeddings


def sample_rows(size: int, count: int, seed: int) -> np.ndarray:
    """按固定种子不重复抽样若干行（返回升序行号，保证顺序稳定）。"""
    if count <= 0 or count >= size:
        return np.arange(size, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(size, count, False))


def fit_final_gate(blocks: list[np.ndarray], model_path: Path) -> None:
    """训练 293 维 candidate 门控并落盘。

    Args:
        blocks: 若干特征块，每块形状 ``[Q, 100, FEATURE_COUNT]``，
            约定第 0 列为正例（重排后保证成立）。
        model_path: 模型输出路径。
    """
    labels = []
    groups = []
    for block in blocks:
        target = np.zeros(block.shape[:2], dtype=np.int8)
        target[:, 0] = 1
        labels.append(target.ravel())
        groups.append(np.full(len(block), block.shape[1], dtype=np.int32))
    model = lgb.LGBMRanker(
        random_state=4400,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        **FINAL_GATE_CONFIG,
    )
    model.fit(
        np.concatenate([block.reshape(-1, FEATURE_COUNT) for block in blocks]),
        np.concatenate(labels),
        group=np.concatenate(groups),
    )
    model.booster_.save_model(model_path)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（未给出时回落到配置文件）。"""
    parser = argparse.ArgumentParser(
        description="Train validation gates from raw dataset2 data"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "train_gates.yaml",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--score-backend", choices=("jittor", "numpy"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--uniform-base", type=int)
    parser.add_argument("--strict-base", type=int)
    parser.add_argument("--uniform-final", type=int)
    parser.add_argument("--strict-final", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--community-dim", type=int)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    """相对路径按仓库根目录解析，绝对路径原样返回。"""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    """按配置（命令行覆盖）重训 base / candidate 门控并落盘可复现产物。"""
    args = parse_args()
    config = deep_update(
        load_config(args.config),
        {
            "data_root": args.data_root,
            "model_dir": args.model_dir,
            "output_dir": args.output_dir,
            "sampling": {
                "uniform_base": args.uniform_base,
                "strict_base": args.strict_base,
                "uniform_final": args.uniform_final,
                "strict_final": args.strict_final,
            },
            "runtime": {
                "seed": args.seed,
                "device": args.device,
                "score_backend": args.score_backend,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "threads": args.threads,
                "community_dim": args.community_dim,
            },
        },
    )
    runtime_section = config.get("runtime", {})
    sampling = config.get("sampling", {})
    data_root = resolve_path(config.get("data_root", "data"))
    output_dir = resolve_path(config.get("output_dir", "outputs"))
    model_dir = resolve_path(config.get("model_dir", "models"))
    dataset = str(config.get("dataset", "dataset2"))
    require_files(
        [data_root / dataset / name for name in ("train.csv", "test.csv")],
        hint=f"check --data-root (now: {data_root}) and see data/README.md",
    )

    runtime = RuntimeConfig(
        seed=int(runtime_section.get("seed", 20240501)),
        device=str(runtime_section.get("device", "auto")),
        score_backend=str(runtime_section.get("score_backend", "jittor")),
        epochs=int(runtime_section.get("epochs", 5)),
        batch_size=int(runtime_section.get("batch_size", 65536)),
        learning_rate=float(runtime_section.get("learning_rate", 0.02)),
        threads=int(runtime_section.get("threads", 16)),
        community_dim=int(runtime_section.get("community_dim", 32)),
        retrain_embeddings=True,
    )
    uniform_base = int(sampling.get("uniform_base", 20000))
    strict_base = int(sampling.get("strict_base", 40000))
    uniform_final = int(sampling.get("uniform_final", 40000))
    strict_final = int(sampling.get("strict_final", 60000))

    set_seed(runtime.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    save_command(output_dir / "command.txt")

    with tee_stdout(output_dir / "train.log"):
        print(f"repo root:   {REPO_ROOT}")
        print(f"data root:   {data_root}")
        print(f"model dir:   {model_dir}")
        print(f"output dir:  {output_dir}")
        print(f"runtime:     {runtime}")
        data = load_dataset(data_root, dataset)
        folds, evaluation = rolling_folds(data)
        uniform_history = history_frame(data, evaluation.history_idx)
        strict_history = history_frame(data, folds[0].history_idx)
        feature_sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        uniform_embeddings = None
        for index, seed in enumerate((42, 2027)):
            print(f"building uniform validation seed={seed}", flush=True)
            frame, positive = uniform_frame(data, seed)
            base, augmented, uniform_embeddings = build_feature_pair(
                uniform_history,
                frame,
                runtime,
                model_dir,
                output_dir / "cache" / f"uniform_{seed}",
                uniform_embeddings,
            )
            feature_sets.append((base, augmented, positive))
        print("building strict source-time validation fold m2", flush=True)
        frame, positive = strict_frame(data, folds[0], 1700)
        base, augmented, _ = build_feature_pair(
            strict_history,
            frame,
            runtime,
            model_dir,
            output_dir / "cache" / "strict_m2",
        )
        feature_sets.append((base, augmented, positive))

        gate_dir = output_dir / "models"
        gate_dir.mkdir(parents=True, exist_ok=True)
        base_sets = []
        for index, (base, _, positive_col) in enumerate(feature_sets):
            count = uniform_base if index < 2 else strict_base
            seed = 2110 + index if index < 2 else 2100
            rows = sample_rows(len(base), count, seed)
            base_sets.append((base[rows], positive_col[rows]))
        base_features = []
        base_labels = []
        base_groups = []
        for block, positive_col in base_sets:
            target = np.zeros(block.shape[:2], dtype=np.int8)
            target[np.arange(len(block)), positive_col] = 1
            base_features.append(block.reshape(-1, 108))
            base_labels.append(target.ravel())
            base_groups.append(np.full(len(block), 100, dtype=np.int32))
        base_gate = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=[0, 1],
            n_estimators=450,
            learning_rate=0.03,
            num_leaves=47,
            max_depth=8,
            min_child_samples=200,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            random_state=2100,
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        base_gate.fit(
            np.concatenate(base_features),
            np.concatenate(base_labels),
            group=np.concatenate(base_groups),
        )
        base_gate.booster_.save_model(gate_dir / "base_gate.txt")

        final_blocks = []
        for index, (_, augmented, positive_col) in enumerate(feature_sets):
            if index < 2:
                pool = sample_rows(len(augmented), 100000, 2500 + index)
                selected = sample_rows(len(pool), uniform_final, 4200 + index)
                block = augmented[pool[selected]]
            else:
                rows = sample_rows(len(augmented), strict_final, 4300)
                original = augmented[rows]
                columns = np.arange(100)[None, :]
                order = np.argsort(
                    columns == positive_col[rows, None], axis=1, kind="stable"
                )[:, ::-1]
                block = np.take_along_axis(original, order[..., None], axis=1)
            final_blocks.append(np.asarray(block, dtype=np.float32))
        fit_final_gate(final_blocks, gate_dir / "candidate_gate.txt")
        print(f"trained gates: {gate_dir}", flush=True)
        del feature_sets, base_sets, final_blocks
        gc.collect()


if __name__ == "__main__":
    main()
