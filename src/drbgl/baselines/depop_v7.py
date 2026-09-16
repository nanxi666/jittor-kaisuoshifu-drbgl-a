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
    score_depop,
    score_memory,
)
from .heuristic_submission import CAND_COLS, detect_strategy


# 仓库根目录（本文件位于 <repo>/src/drbgl/baselines/ 下），仅用作脚本默认路径，
# 实际路径一律由命令行参数或配置文件传入。
ROOT = Path(__file__).resolve().parents[3]


LOW_RANK_CONFIG = {
    "rank": 64,
    "degree_power": 0.5,
    "n_iter": 5,
    "seed": 42,
    "rank_weight": 0.4213,
}


def score_depop_v7(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    base = score_depop(train, test)
    low_rank = normalized_svd_scores(
        train,
        test,
        CAND_COLS,
        rank=LOW_RANK_CONFIG["rank"],
        degree_power=LOW_RANK_CONFIG["degree_power"],
        n_iter=LOW_RANK_CONFIG["n_iter"],
        seed=LOW_RANK_CONFIG["seed"],
    )
    candidates = test[CAND_COLS].to_numpy(np.int64)
    base_rank = 100.0 * ordinal_rank_scores(base, candidates)
    low_rank_rank = average_row_ranks(low_rank)
    return base_rank + LOW_RANK_CONFIG["rank_weight"] * low_rank_rank


def score_v7(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, str, bool, float, dict]:
    strategy, bipartite, repeat_rate = detect_strategy(train)
    if strategy == "memory":
        scores = score_memory(train, test, bipartite)
        config = {"memory": MEMORY_CONFIG}
    else:
        scores = score_depop_v7(train, test)
        config = {"depop": DEPOP_CONFIG, "low_rank": LOW_RANK_CONFIG}
    return scores, strategy, bipartite, repeat_rate, config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B-board adaptive V7 scorer with normalized low-rank CF"
    )
    parser.add_argument(
        "--dataset", choices=("dataset1", "dataset2"), required=True
    )
    parser.add_argument("--run-name", default="adaptive_v7")
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
    scores, strategy, bipartite, repeat_rate, config = score_v7(train, test)

    run_dir = Path(args.output_dir) / "v2" / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    score_path = run_dir / f"{args.dataset}_scores.npy"
    np.save(score_path, scores.astype(np.float32))
    manifest = {
        "dataset": args.dataset,
        "scorer": "adaptive_v7",
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
            "No A-board ids or factors are persisted; strategy, mappings, "
            "statistics, and the low-rank factorization are rebuilt from "
            "the current train/test files."
        ),
    }
    manifest_path = run_dir / f"{args.dataset}_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
