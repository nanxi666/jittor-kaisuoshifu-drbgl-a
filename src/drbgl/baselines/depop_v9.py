#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import git_metadata, sha256_file, write_json
from ..experts.low_rank import average_row_ranks, normalized_svd_scores
from ..metrics import ordinal_rank_scores
from .depop_v3 import (
    DEPOP_CONFIG,
    MEMORY_CONFIG,
    candidate_exposure_features,
    local_pair_exposure,
    pair_exposure,
    score_depop_components,
    score_memory,
)
from .depop_v7 import LOW_RANK_CONFIG
from .heuristic_submission import CAND_COLS, detect_strategy


# 仓库根目录（本文件位于 <repo>/src/drbgl/baselines/ 下），仅用作脚本默认路径，
# 实际路径一律由命令行参数或配置文件传入。
ROOT = Path(__file__).resolve().parents[3]


DAY = 86400.0
V9_CONFIG = {
    "static": {
        "rank": 64,
        "degree_power": 0.5,
        "weight": 0.2,
    },
    "temporal": [
        {"half_life_days": 120.0, "weight": 0.2},
        {"half_life_days": 180.0, "weight": 0.3},
        {"half_life_days": 365.0, "weight": 0.1},
    ],
    "trend": {"days": 365.0, "weight": 0.1},
    "candidate_exposure": {"bins": [4, 16], "weight": 0.02},
    "pair_exposure": {
        "global_weight": -0.05,
        "local_bins": 8,
        "local_weight": 0.1,
    },
    "asymmetric_static": {
        "rank": 64,
        "user_degree_power": 0.5,
        "item_degree_power": 0.25,
        "weight": 0.02,
    },
    "n_iter": 5,
    "seed": 42,
}


def initialize_depop_expert_scores(
    base: np.ndarray,
    candidates: np.ndarray,
    static: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the V7 view and V9 accumulator from one static SVD result."""
    base_rank = 100.0 * ordinal_rank_scores(base, candidates)
    static_rank = average_row_ranks(static)
    v7_scores = base_rank + LOW_RANK_CONFIG["rank_weight"] * static_rank
    v9_scores = base_rank + V9_CONFIG["static"]["weight"] * static_rank
    return v7_scores, v9_scores


def item_trend_ranks(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidates: np.ndarray,
    days: float,
) -> np.ndarray:
    """Recent item activity rank minus its all-history activity rank."""
    items, total = np.unique(
        train["dst"].to_numpy(np.int64), return_counts=True
    )
    reference_time = float(test["time"].min())
    recent_dst = train.loc[
        train["time"] >= reference_time - days * DAY, "dst"
    ].to_numpy(np.int64)
    recent_items, recent_counts = np.unique(recent_dst, return_counts=True)

    total_pos = np.searchsorted(items, candidates)
    total_clip = np.minimum(total_pos, len(items) - 1)
    total_valid = (total_pos < len(items)) & (
        items[total_clip] == candidates
    )
    total_values = np.where(total_valid, total[total_clip], 0)

    if len(recent_items) == 0:
        recent_values = np.zeros(candidates.shape, dtype=np.int64)
    else:
        recent_pos = np.searchsorted(recent_items, candidates)
        recent_clip = np.minimum(recent_pos, len(recent_items) - 1)
        recent_valid = (recent_pos < len(recent_items)) & (
            recent_items[recent_clip] == candidates
        )
        recent_values = np.where(
            recent_valid, recent_counts[recent_clip], 0
        )
    return average_row_ranks(recent_values) - average_row_ranks(total_values)


def score_depop_v9_components(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build V9 experts and expose factor scores for downstream audits."""
    candidates = test[CAND_COLS].to_numpy(np.int64)
    query_src = test["src"].to_numpy(np.int64)
    query_time = test["time"].to_numpy(np.int64)
    config = V9_CONFIG

    depop = score_depop_components(train, test)
    base = depop["v6"]
    static_config = config["static"]
    static = normalized_svd_scores(
        train,
        test,
        CAND_COLS,
        rank=static_config["rank"],
        degree_power=static_config["degree_power"],
        n_iter=config["n_iter"],
        seed=config["seed"],
    )
    v7_scores, scores = initialize_depop_expert_scores(
        base, candidates, static
    )
    components = {
        "v5": depop["v5"],
        "user_cf": depop["user_cf"],
        "static": static,
    }

    for temporal_config in config["temporal"]:
        temporal = normalized_svd_scores(
            train,
            test,
            CAND_COLS,
            rank=static_config["rank"],
            degree_power=static_config["degree_power"],
            n_iter=config["n_iter"],
            seed=config["seed"],
            half_life_days=temporal_config["half_life_days"],
        )
        components[
            f"temporal_{temporal_config['half_life_days']:g}"
        ] = temporal
        scores += temporal_config["weight"] * average_row_ranks(temporal)

    trend_config = config["trend"]
    scores += trend_config["weight"] * item_trend_ranks(
        train, test, candidates, trend_config["days"]
    )

    exposure_config = config["candidate_exposure"]
    for bins in exposure_config["bins"]:
        _, local_exposure = candidate_exposure_features(
            query_time, candidates, bins
        )
        scores += exposure_config["weight"] * average_row_ranks(
            local_exposure
        )

    pair_config = config["pair_exposure"]
    global_pair = pair_exposure(query_src, candidates)
    local_pair = local_pair_exposure(
        query_src,
        query_time,
        candidates,
        pair_config["local_bins"],
    )
    scores += pair_config["global_weight"] * average_row_ranks(global_pair)
    scores += pair_config["local_weight"] * average_row_ranks(local_pair)

    asymmetric_config = config["asymmetric_static"]
    asymmetric = normalized_svd_scores(
        train,
        test,
        CAND_COLS,
        rank=asymmetric_config["rank"],
        n_iter=config["n_iter"],
        seed=config["seed"],
        user_degree_power=asymmetric_config["user_degree_power"],
        item_degree_power=asymmetric_config["item_degree_power"],
    )
    components["asymmetric_static"] = asymmetric
    scores += asymmetric_config["weight"] * average_row_ranks(asymmetric)
    return {"v6": base, "v7": v7_scores, "v9": scores}, components


def score_depop_v9_experts(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Build the V6, V7, and V9 views while sharing V9's factorization work."""
    experts, _ = score_depop_v9_components(train, test)
    return experts


def score_depop_v9(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return score_depop_v9_experts(train, test)["v9"]


def score_v9_experts(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], str, bool, float, dict]:
    strategy, bipartite, repeat_rate = detect_strategy(train)
    if strategy == "memory":
        scores = score_memory(train, test, bipartite)
        experts = {name: scores for name in ("v6", "v7", "v9")}
        config = {"memory": MEMORY_CONFIG}
    else:
        experts = score_depop_v9_experts(train, test)
        config = {"depop": DEPOP_CONFIG, "v9": V9_CONFIG}
    return experts, strategy, bipartite, repeat_rate, config


def score_v9(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, str, bool, float, dict]:
    experts, strategy, bipartite, repeat_rate, config = score_v9_experts(
        train, test
    )
    return experts["v9"], strategy, bipartite, repeat_rate, config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B-board adaptive V9 multiscale transductive scorer"
    )
    parser.add_argument(
        "--dataset", choices=("dataset1", "dataset2"), required=True
    )
    parser.add_argument("--run-name", default="adaptive_v9")
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
    started = time.time()
    scores, strategy, bipartite, repeat_rate, config = score_v9(train, test)
    run_dir = Path(args.output_dir) / "v2" / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    score_path = run_dir / f"{args.dataset}_scores.npy"
    np.save(score_path, scores.astype(np.float32))
    manifest = {
        "dataset": args.dataset,
        "scorer": "adaptive_v9",
        "strategy": strategy,
        "bipartite": bipartite,
        "holdout_repeat_rate": repeat_rate,
        "config": config,
        "rows": {"train": len(train), "test": len(test)},
        "candidate_pool_size": int(np.unique(test[CAND_COLS]).size),
        "scores": str(score_path),
        "scores_sha256": sha256_file(score_path),
        "elapsed_seconds": round(time.time() - started, 2),
        "git": git_metadata(ROOT),
        "b_board_policy": (
            "No A-board IDs or factors are persisted. Low-rank factors, "
            "trend statistics, and multiscale transductive exposure are "
            "rebuilt from the current train/test files."
        ),
    }
    write_json(run_dir / f"{args.dataset}_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
