from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_COLUMNS = [f"c{i}" for i in range(1, 101)]


@dataclass(frozen=True)
class DatasetArrays:
    name: str
    src: np.ndarray
    dst: np.ndarray
    time: np.ndarray
    split: np.ndarray | None
    test_src: np.ndarray
    test_time: np.ndarray
    test_candidates: np.ndarray


@dataclass(frozen=True)
class FrozenFold:
    name: str
    history_idx: np.ndarray
    label_idx: np.ndarray


def load_dataset(root: Path, name: str) -> DatasetArrays:
    """把某个数据集的原始 CSV 读成紧凑数组。

    Args:
        root: 数据根目录，需含 ``<name>/train.csv`` 与 ``<name>/test.csv``。
        name: 数据集名，如 ``dataset2``。

    Returns:
        ``DatasetArrays``：训练三元组数组 + 测试查询数组（均为 int64）。
    """
    train = pd.read_csv(root / name / "train.csv")
    test = pd.read_csv(root / name / "test.csv")
    split = train["split"].to_numpy(np.int8) if "split" in train else None
    return DatasetArrays(
        name=name,
        src=train["src"].to_numpy(np.int64),
        dst=train["dst"].to_numpy(np.int64),
        time=train["time"].to_numpy(np.int64),
        split=split,
        test_src=test["src"].to_numpy(np.int64),
        test_time=test["time"].to_numpy(np.int64),
        test_candidates=test[CANDIDATE_COLUMNS].to_numpy(np.int64),
    )


def _interval_fold(
    time: np.ndarray,
    start_time: int,
    end_time: int | None,
    name: str,
) -> FrozenFold:
    if not np.all(time[1:] >= time[:-1]):
        raise ValueError("training rows must be sorted by time")
    history_idx = np.flatnonzero(time < start_time)
    if end_time is None:
        label_idx = np.flatnonzero(time >= start_time)
    else:
        label_idx = np.flatnonzero((time >= start_time) & (time < end_time))
    if len(history_idx) == 0 or len(label_idx) == 0:
        raise ValueError(f"empty temporal fold {name}")
    if time[history_idx].max() >= time[label_idx].min():
        raise AssertionError(f"history leaks into labels for fold {name}")
    return FrozenFold(name=name, history_idx=history_idx, label_idx=label_idx)


def rolling_folds(
    data: DatasetArrays,
) -> tuple[list[FrozenFold], FrozenFold]:
    """按测试时间跨度构造严格时间切分：两个训练折 + 一个验证折。

    Args:
        data: ``load_dataset`` 的返回值。

    Returns:
        ``(train_folds, evaluation)``，每个折含 history_idx（历史）与
        label_idx（标签）两个行索引数组，且保证历史严格早于标签。
    """
    span = int(data.test_time.max() - data.test_time.min())
    if span <= 0:
        raise ValueError("test queries must cover a positive time span")
    evaluation_start = int(data.time.max()) - span
    train_folds = []
    for offset in (2, 1):
        start = evaluation_start - offset * span
        end = start + span
        train_folds.append(
            _interval_fold(
                data.time,
                start,
                end,
                f"{data.name}_span_m{offset}",
            )
        )
    evaluation = _interval_fold(
        data.time,
        evaluation_start,
        None,
        f"{data.name}_openjittor_eval",
    )
    return train_folds, evaluation


def sample_indices(indices: np.ndarray, size: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if size <= 0 or size >= len(indices):
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=size, replace=False))
