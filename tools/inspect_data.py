#!/usr/bin/env python3
"""数据统计与体检工具：检查数据是否齐备，并打印基本统计信息。

用法（在仓库根目录执行）::

    python tools/inspect_data.py
    python tools/inspect_data.py --data-root /path/to/data --datasets dataset1 dataset2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_COLUMNS = [f"c{index}" for index in range(1, 101)]
TRAIN_COLUMNS = ("src", "dst", "time")
TEST_COLUMNS = ("src", "time", *CANDIDATE_COLUMNS)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Inspect DRBGL datasets")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--datasets", nargs="+", default=["dataset1", "dataset2"]
    )
    return parser.parse_args()


def describe(name: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """打印单个数据集的统计信息并做基本合法性检查。

    Args:
        name: 数据集名。
        train: 训练表，列 ``src,dst,time``。
        test: 测试表，列 ``src,time,c1..c100``。

    Returns:
        统计字典（行数、节点数、时间范围等）。
    """
    train_users = int(train["src"].nunique())
    train_items = int(train["dst"].nunique())
    candidates = test[CANDIDATE_COLUMNS].to_numpy(np.int64)
    stats = {
        "dataset": name,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_users": train_users,
        "train_items": train_items,
        "train_time_range": [int(train["time"].min()), int(train["time"].max())],
        "test_time_range": [int(test["time"].min()), int(test["time"].max())],
        "candidate_pool": int(np.unique(candidates).size),
        "test_candidates_per_row": int(candidates.shape[1]),
        "bipartite": bool(
            len(np.intersect1d(train["src"].unique(), train["dst"].unique())) == 0
        ),
    }
    if int(train["time"].max()) >= int(test["time"].min()):
        print(f"[{name}] WARNING: 训练时间未严格早于测试时间")
    print(f"[{name}] " + ", ".join(f"{key}={value}" for key, value in stats.items()))
    return stats


def main() -> None:
    """逐个数据集做体检，缺文件给出修复提示。"""
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise SystemExit(
            f"data root not found: {data_root}\n"
            "how to fix: put train.csv / test.csv under "
            f"{data_root / '<dataset>'} (see data/README.md)"
        )
    for name in args.datasets:
        train_path = data_root / name / "train.csv"
        test_path = data_root / name / "test.csv"
        missing = [
            str(path) for path in (train_path, test_path) if not path.is_file()
        ]
        if missing:
            print(f"[{name}] MISSING: " + ", ".join(missing))
            continue
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        missing_train = set(TRAIN_COLUMNS) - set(train.columns)
        missing_test = set(TEST_COLUMNS) - set(test.columns)
        if missing_train or missing_test:
            print(
                f"[{name}] INVALID columns: "
                f"train missing={sorted(missing_train)}, "
                f"test missing={sorted(missing_test)}"
            )
            continue
        describe(name, train, test)


if __name__ == "__main__":
    main()
