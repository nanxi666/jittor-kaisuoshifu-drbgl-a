#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..artifacts import git_metadata, sha256_file, write_json
from .heuristic_core import (
    DAY,
    EventIndex,
    candidate_features,
    hash_tiebreak,
    repeat_flags,
)
from .heuristic_submission import CAND_COLS, detect_strategy, item_cf_scores


# 仓库根目录（本文件位于 <repo>/src/drbgl/baselines/ 下），仅用作脚本默认路径，
# 实际路径一律由命令行参数或配置文件传入。
ROOT = Path(__file__).resolve().parents[3]


MEMORY_CONFIG = {
    "pair_count_weight": 2.0,
    "pair_recency_weight": 25.0,
    "pair_recency_days": 60.0,
    "activity_days": 90,
    "activity_weight": 0.0001,
    "cn_weight": 1.0,
    "ra_weight": 0.1,
    "exposure_weight": 0.01,
    "pair_exposure_weight": 0.4,
    "local_pair_exposure_bins": 4,
    "local_pair_exposure_weight": 0.2,
}
DEPOP_CONFIG = {
    "popularity_days": 365,
    "total_weight": 0.02,
    "recency_weight": 30.0,
    "recency_days": 730.0,
    "repeat_multiplier": 0.01,
    "cold_multiplier": 0.01,
    "cf_recent_items": 10,
    "cf_weight": 10.0,
    "user_cf_recent_items": 10,
    "user_cf_neighbors": 200,
    "user_cf_weight": 225.0,
    "exposure_mode": "positive_excess",
    "exposure_weight": 0.6,
    "local_exposure_bins": 8,
    "local_exposure_weight": 8.0,
}


def _time_bins(
    query_time: np.ndarray,
    bin_count: int,
    shift: float,
) -> np.ndarray:
    query_time = np.asarray(query_time, dtype=np.float64)
    span = max(float(query_time.max() - query_time.min()), 1.0)
    width = span / bin_count
    bins = np.floor(
        (query_time - query_time.min() + shift * width) / width
    ).astype(np.int64)
    return bins - bins.min()


def candidate_exposure_features(
    query_time: np.ndarray,
    candidates: np.ndarray,
    local_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate positive candidate exposure above uniform-negative noise."""
    candidates = np.asarray(candidates, dtype=np.int64)
    _, inverse, counts = np.unique(
        candidates,
        return_inverse=True,
        return_counts=True,
    )
    candidate_index = inverse.reshape(candidates.shape)
    pool_size = len(counts)
    negatives_per_query = candidates.shape[1] - 1
    expected = negatives_per_query * len(candidates) / pool_size
    global_excess = np.maximum(counts[candidate_index] - expected, 0.0)

    local_excess = np.zeros(candidates.shape, dtype=np.float64)
    for shift in (0.0, 1.0 / 3.0, 2.0 / 3.0):
        bins = _time_bins(query_time, local_bins, shift)
        bin_count = int(bins.max()) + 1
        rows_per_bin = np.bincount(bins, minlength=bin_count)
        keys = bins[:, None] * np.int64(pool_size) + candidate_index
        local_counts = np.bincount(
            keys.ravel(),
            minlength=bin_count * pool_size,
        ).reshape(bin_count, pool_size)
        observed = local_counts[bins[:, None], candidate_index]
        local_expected = (
            negatives_per_query * rows_per_bin[bins] / pool_size
        )
        local_excess += np.maximum(
            observed - local_expected[:, None],
            0.0,
        ) / np.sqrt(np.maximum(local_expected[:, None], 1.0))
    return global_excess.astype(np.float64), local_excess / 3.0


def candidate_log_exposure(candidates: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(
        candidates,
        return_inverse=True,
        return_counts=True,
    )
    return np.log1p(counts[inverse]).reshape(candidates.shape)


def _other_row_counts(keys: np.ndarray) -> np.ndarray:
    """Count rows other than the current row containing each key."""
    sorted_keys = np.sort(keys, axis=1)
    first_in_row = np.ones(sorted_keys.shape, dtype=bool)
    first_in_row[:, 1:] = sorted_keys[:, 1:] != sorted_keys[:, :-1]
    unique_row_keys = sorted_keys[first_in_row]
    unique_keys, row_counts = np.unique(unique_row_keys, return_counts=True)
    positions = np.searchsorted(unique_keys, keys.ravel())
    return (row_counts[positions] - 1).reshape(keys.shape).astype(np.float64)


def _compact_pair_keys(
    query_src: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, int]:
    _, source_index = np.unique(query_src, return_inverse=True)
    candidate_ids, candidate_index = np.unique(candidates, return_inverse=True)
    pool_size = len(candidate_ids)
    keys = (
        source_index[:, None] * np.int64(pool_size)
        + candidate_index.reshape(candidates.shape)
    )
    return keys, len(np.unique(query_src)) * pool_size


def pair_exposure(
    query_src: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Count other test rows where the same src-candidate pair appears.

    OpenJittor samples validation negatives with replacement, so a negative can
    occur more than once inside one row.  That is one observation rather than
    evidence from another query.  Deduplicating within each row avoids a
    validation-only sampling artefact and matches the feature definition.
    """
    keys, _ = _compact_pair_keys(query_src, candidates)
    return _other_row_counts(keys)


def local_pair_exposure(
    query_src: np.ndarray,
    query_time: np.ndarray,
    candidates: np.ndarray,
    local_bins: int,
) -> np.ndarray:
    """Average pair exposure over shifted, test-span-relative time bins."""
    keys, key_space = _compact_pair_keys(query_src, candidates)
    exposure = np.zeros(candidates.shape, dtype=np.float64)
    for shift in (0.0, 1.0 / 3.0, 2.0 / 3.0):
        bins = _time_bins(query_time, local_bins, shift)
        local_keys = bins[:, None] * np.int64(key_space) + keys
        exposure += _other_row_counts(local_keys)
    return exposure / 3.0


def graph_memory_scores(
    src: np.ndarray,
    dst: np.ndarray,
    query_src: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.unique(
        np.concatenate([src, dst, query_src, candidates.ravel()])
    )
    src_index = np.searchsorted(nodes, src)
    dst_index = np.searchsorted(nodes, dst)
    query_source_index = np.searchsorted(nodes, query_src)
    candidate_index = np.searchsorted(nodes, candidates)
    size = len(nodes)
    adjacency = sparse.csr_matrix(
        (
            np.ones(2 * len(src), dtype=np.float32),
            (
                np.concatenate([src_index, dst_index]),
                np.concatenate([dst_index, src_index]),
            ),
        ),
        shape=(size, size),
    )
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    ra_weight = np.zeros(size, dtype=np.float32)
    ra_weight[degree > 0] = 1.0 / degree[degree > 0]

    cn = np.zeros(candidates.shape, dtype=np.float32)
    ra = np.zeros(candidates.shape, dtype=np.float32)
    unique_src, inverse = np.unique(query_source_index, return_inverse=True)
    for group, node in enumerate(unique_src):
        rows = np.flatnonzero(inverse == group)
        block = candidate_index[rows]
        flat = block.ravel()
        row = adjacency.getrow(int(node))
        cn_values = np.asarray((row @ adjacency)[:, flat].todense()).reshape(
            block.shape
        )
        ra_values = np.asarray(
            (row.multiply(ra_weight) @ adjacency)[:, flat].todense()
        ).reshape(block.shape)
        cn[rows] = cn_values
        ra[rows] = ra_values
    return cn, ra


def user_knn_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    recent_items: int,
    neighbors: int,
) -> np.ndarray:
    """Cosine user-kNN scores from a frozen bipartite interaction graph."""
    user = train["src"].to_numpy(np.int64)
    item = train["dst"].to_numpy(np.int64)
    event_time = train["time"].to_numpy(np.int64)
    users, user_index = np.unique(user, return_inverse=True)
    items, item_index = np.unique(item, return_inverse=True)
    interactions = sparse.csr_matrix(
        (
            np.ones(len(train), dtype=np.float32),
            (user_index, item_index),
        ),
        shape=(len(users), len(items)),
    )
    interactions.data[:] = 1.0
    interactions_csc = interactions.tocsc()
    degree = np.asarray(interactions.sum(axis=1)).ravel()
    history = EventIndex(user, event_time, payload=item)

    query_src = test["src"].to_numpy(np.int64)
    candidates = test[CAND_COLS].to_numpy(np.int64)
    candidate_index = np.searchsorted(
        items,
        np.minimum(candidates, items.max()),
    )
    candidate_index = np.where(
        items[np.minimum(candidate_index, len(items) - 1)] == candidates,
        candidate_index,
        -1,
    )
    scores = np.zeros(candidates.shape, dtype=np.float32)
    unique_source, source_group = np.unique(query_src, return_inverse=True)
    started = time.time()
    for group, source in enumerate(unique_source):
        rows = np.flatnonzero(source_group == group)
        lo, hi = history.locate(source)
        if hi <= lo:
            continue
        recent = np.unique(
            history.payload[max(lo, hi - recent_items) : hi]
        )
        columns = np.searchsorted(items, recent)
        valid = (columns < len(items)) & (
            items[np.minimum(columns, len(items) - 1)] == recent
        )
        columns = columns[valid]
        if len(columns) == 0:
            continue

        overlap = np.asarray(
            interactions_csc[:, columns].sum(axis=1)
        ).ravel()
        own_index = np.searchsorted(users, source)
        if own_index < len(users) and users[own_index] == source:
            overlap[own_index] = 0.0
        similarity = overlap / np.sqrt(
            np.maximum(degree * len(columns), 1.0)
        )
        nonzero = np.flatnonzero(similarity > 0)
        nearest = nonzero[np.argsort(similarity[nonzero])[::-1]][:neighbors]
        if len(nearest) == 0:
            continue
        item_scores = np.asarray(
            interactions[nearest].T @ similarity[nearest]
        ).ravel()
        valid_candidate = candidate_index[rows] >= 0
        block = np.zeros((len(rows), candidates.shape[1]), dtype=np.float32)
        block[valid_candidate] = item_scores[
            candidate_index[rows][valid_candidate]
        ]
        scores[rows] = block
        if group % 500 == 0:
            print(
                f"  user-cf source {group}/{len(unique_source)} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )
    return scores


def score_memory(
    train: pd.DataFrame,
    test: pd.DataFrame,
    bipartite: bool,
) -> np.ndarray:
    config = MEMORY_CONFIG
    src = train["src"].to_numpy(np.int64)
    dst = train["dst"].to_numpy(np.int64)
    event_time = train["time"].to_numpy(np.int64)
    query_src = test["src"].to_numpy(np.int64)
    query_time = test["time"].to_numpy(np.int64)
    candidates = test[CAND_COLS].to_numpy(np.int64)
    nodes = np.unique(
        np.concatenate([src, dst, query_src, candidates.ravel()])
    )
    source_index = np.searchsorted(nodes, src)
    destination_index = np.searchsorted(nodes, dst)
    query_source_index = np.searchsorted(nodes, query_src)
    candidate_index = np.searchsorted(nodes, candidates)
    multiplier = len(nodes)

    if bipartite:
        pair_key = source_index * multiplier + destination_index
        pair_index = EventIndex(pair_key, event_time)
        activity_index = EventIndex(destination_index, event_time)
    else:
        pair_key = np.concatenate(
            [
                source_index * multiplier + destination_index,
                destination_index * multiplier + source_index,
            ]
        )
        doubled_time = np.concatenate([event_time, event_time])
        pair_index = EventIndex(pair_key, doubled_time)
        activity_index = EventIndex(
            np.concatenate([source_index, destination_index]),
            doubled_time,
        )

    query_key = query_source_index[:, None] * multiplier + candidate_index
    pair_count, pair_last, _ = candidate_features(
        pair_index, query_key, query_time, []
    )
    _, _, activity = candidate_features(
        activity_index,
        candidate_index,
        query_time,
        [config["activity_days"]],
    )
    pair_hit = pair_count > 0
    pair_gap = query_time[:, None].astype(np.float64) - pair_last
    recency = np.where(
        pair_hit,
        np.exp(
            -np.maximum(pair_gap, 0)
            / (config["pair_recency_days"] * DAY)
        ),
        0.0,
    )
    scores = np.where(
        pair_hit,
        100.0
        + config["pair_count_weight"] * pair_count
        + config["pair_recency_weight"] * recency,
        0.0,
    )
    scores += config["activity_weight"] * activity[config["activity_days"]]

    if not bipartite:
        cn, ra = graph_memory_scores(src, dst, query_src, candidates)
        scores += (
            config["cn_weight"] * np.log1p(cn)
            + config["ra_weight"] * np.log1p(ra)
        ) * (~pair_hit)

    scores += config["exposure_weight"] * candidate_log_exposure(candidates)
    scores += config["pair_exposure_weight"] * pair_exposure(
        query_src, candidates
    )
    scores += config["local_pair_exposure_weight"] * local_pair_exposure(
        query_src,
        query_time,
        candidates,
        config["local_pair_exposure_bins"],
    )
    return scores + hash_tiebreak(candidates)


def score_depop_components(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Build depop variants once and retain the independent user-CF view."""
    config = DEPOP_CONFIG
    event_time = train["time"].to_numpy(np.int64)
    query_src = test["src"].to_numpy(np.int64)
    query_time = test["time"].to_numpy(np.int64)
    candidates = test[CAND_COLS].to_numpy(np.int64)
    dst_index = EventIndex(train["dst"].to_numpy(np.int64), event_time)
    src_index = EventIndex(
        train["src"].to_numpy(np.int64),
        event_time,
        payload=train["dst"].to_numpy(np.int64),
    )
    total, last, recent = candidate_features(
        dst_index,
        candidates,
        query_time,
        [config["popularity_days"]],
    )
    repeated = repeat_flags(src_index, query_src, candidates, query_time)
    gap = query_time[:, None].astype(np.float64) - last
    recency = np.where(
        last > -(10**17),
        np.exp(
            -np.maximum(gap, 0)
            / (config["recency_days"] * DAY)
        ),
        0.0,
    )
    multiplier = np.where(
        repeated, config["repeat_multiplier"], 1.0
    ) * np.where(total == 0, config["cold_multiplier"], 1.0)
    scores = (
        recent[config["popularity_days"]]
        + config["total_weight"] * total
        + config["recency_weight"] * recency
    ) * multiplier

    cf = item_cf_scores(
        train,
        test,
        K=config["cf_recent_items"],
        use_iuf=False,
        item_decay=0,
    )
    scores += config["cf_weight"] * np.log1p(cf) * multiplier
    user_cf = user_knn_scores(
        train,
        test,
        recent_items=config["user_cf_recent_items"],
        neighbors=config["user_cf_neighbors"],
    )
    global_exposure, local_exposure = candidate_exposure_features(
        query_time,
        candidates,
        config["local_exposure_bins"],
    )
    scores += config["exposure_weight"] * global_exposure
    scores += config["local_exposure_weight"] * local_exposure
    v5_scores = scores + hash_tiebreak(candidates)
    v6_scores = (
        v5_scores
        + config["user_cf_weight"] * np.log1p(user_cf) * multiplier
    )
    return {
        "v5": v5_scores,
        "v6": v6_scores,
        "user_cf": user_cf,
    }


def score_depop(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return score_depop_components(train, test)["v6"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B-board adaptive OpenJittor-tuned heuristic scorer"
    )
    parser.add_argument(
        "--dataset",
        choices=("dataset1", "dataset2"),
        required=True,
    )
    parser.add_argument("--run-name", default="adaptive_v6")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    dataset_dir = Path(args.data_root) / args.dataset
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"
    missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing {args.dataset} data: {', '.join(missing)}; "
            "check --data-root and see data/README.md"
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if int(train["time"].max()) >= int(test["time"].min()):
        raise ValueError("test time must be strictly after all training events")
    strategy, bipartite, repeat_rate = detect_strategy(train)
    started = time.time()
    if strategy == "memory":
        scores = score_memory(train, test, bipartite)
        config = MEMORY_CONFIG
    else:
        scores = score_depop(train, test)
        config = DEPOP_CONFIG

    run_dir = Path(args.output_dir) / "v2" / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    score_path = run_dir / f"{args.dataset}_scores.npy"
    np.save(score_path, scores.astype(np.float32))
    manifest = {
        "dataset": args.dataset,
        "strategy": strategy,
        "bipartite": bipartite,
        "holdout_repeat_rate": repeat_rate,
        "config": config,
        "rows": {"train": len(train), "test": len(test)},
        "candidate_pool_size": int(
            np.unique(test[CAND_COLS].to_numpy(np.int64)).size
        ),
        "scores": str(score_path),
        "scores_sha256": sha256_file(score_path),
        "elapsed_seconds": round(time.time() - started, 2),
        "git": git_metadata(ROOT),
        "b_board_policy": (
            "No A-board ids are persisted; strategy, graph, time statistics, "
            "candidate pool, and exposure are rebuilt from current inputs."
        ),
    }
    manifest_path = run_dir / f"{args.dataset}_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
