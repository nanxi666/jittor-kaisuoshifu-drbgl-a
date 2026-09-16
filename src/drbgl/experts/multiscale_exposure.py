from __future__ import annotations

import numpy as np

from .low_rank import average_row_ranks
from ..baselines.depop_v3 import (
    _compact_pair_keys,
    _other_row_counts,
    _time_bins,
)


ITEM_BINS = (2, 8, 32, 64, 128)
PAIR_BINS = (2, 4, 16, 32)
VIEW_SUFFIXES = ("log", "rank", "minmax", "positive")
MULTISCALE_EXPOSURE_FEATURE_NAMES = (
    *(f"item_local{bins}_{suffix}" for bins in ITEM_BINS for suffix in VIEW_SUFFIXES),
    *(f"pair_local{bins}_{suffix}" for bins in PAIR_BINS for suffix in VIEW_SUFFIXES),
)


def _scale_free_views(values: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    rank = average_row_ranks(values) / values.shape[1]
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return [
        np.log1p(np.maximum(values, 0.0)),
        rank,
        (values - low) / np.maximum(high - low, 1e-6),
        (values > 0).astype(np.float32),
    ]


def multiscale_exposure_features(
    query_src: np.ndarray,
    query_time: np.ndarray,
    candidates: np.ndarray,
    selected_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Derive label-free item/pair density views at several time scales."""
    query_src = np.asarray(query_src, dtype=np.int64)
    query_time = np.asarray(query_time, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.ndim != 2 or candidates.shape[0] != len(query_src):
        raise ValueError("multiscale query arrays do not align")
    if len(query_time) != len(query_src):
        raise ValueError("multiscale query times do not align")
    if selected_rows is None:
        selected_rows = np.arange(len(candidates), dtype=np.int64)
    else:
        selected_rows = np.asarray(selected_rows, dtype=np.int64)

    _, inverse, counts = np.unique(
        candidates, return_inverse=True, return_counts=True
    )
    candidate_index = inverse.reshape(candidates.shape)
    pool_size = len(counts)
    negatives_per_query = candidates.shape[1] - 1
    pair_keys, pair_key_space = _compact_pair_keys(query_src, candidates)

    columns: list[np.ndarray] = []
    for bins in ITEM_BINS:
        local = np.zeros(candidates.shape, dtype=np.float64)
        for shift in (0.0, 1.0 / 3.0, 2.0 / 3.0):
            time_bin = _time_bins(query_time, bins, shift)
            bin_count = int(time_bin.max()) + 1
            rows_per_bin = np.bincount(time_bin, minlength=bin_count)
            keys = time_bin[:, None] * np.int64(pool_size) + candidate_index
            observed = np.bincount(
                keys.ravel(), minlength=bin_count * pool_size
            ).reshape(bin_count, pool_size)[time_bin[:, None], candidate_index]
            expected = negatives_per_query * rows_per_bin[time_bin] / pool_size
            local += np.maximum(observed - expected[:, None], 0.0) / np.sqrt(
                np.maximum(expected[:, None], 1.0)
            )
        local /= 3.0
        columns.extend(_scale_free_views(local[selected_rows]))
    for bins in PAIR_BINS:
        local = np.zeros(candidates.shape, dtype=np.float64)
        for shift in (0.0, 1.0 / 3.0, 2.0 / 3.0):
            time_bin = _time_bins(query_time, bins, shift)
            local_keys = time_bin[:, None] * np.int64(pair_key_space) + pair_keys
            local += _other_row_counts(local_keys)
        local /= 3.0
        columns.extend(_scale_free_views(local[selected_rows]))
    output = np.stack(columns, axis=-1).astype(np.float32)
    if output.shape[-1] != len(MULTISCALE_EXPOSURE_FEATURE_NAMES):
        raise AssertionError("multiscale exposure names and columns differ")
    return np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
