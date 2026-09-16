from __future__ import annotations

import numpy as np

from .low_rank import average_row_ranks


EXTRA_EXPERT_FEATURE_SUFFIXES = (
    "rank",
    "minmax",
    "time_rank",
    "minus_consensus_mean",
    "minus_consensus_max",
    "query_margin",
    "query_correlation",
    "query_top_consensus",
)
EXTRA_EXPERT_CONSENSUS_NAMES = (
    "rank_mean",
    "rank_median",
    "rank_max",
    "rank_min",
    "rank_std",
    "rank_top2_mean",
    "rank_top3_mean",
)


def extra_expert_features(
    base_features: np.ndarray,
    score_matrices: list[np.ndarray] | tuple[np.ndarray, ...],
) -> np.ndarray:
    """Build scale-free candidate and reliability views for extra experts."""
    base = np.asarray(base_features, dtype=np.float32)
    if base.ndim != 3 or base.shape[-1] < 108:
        raise ValueError("base features must be [queries, candidates, >=108]")
    if not score_matrices:
        raise ValueError("at least one extra expert is required")
    base_ranks = base[..., :15]
    consensus = base_ranks.mean(axis=-1)
    consensus_centered = consensus - consensus.mean(axis=1, keepdims=True)
    consensus_std = consensus_centered.std(axis=1)
    time_fraction = base[..., 47]
    columns: list[np.ndarray] = []
    expected_shape = base.shape[:2]
    for values in score_matrices:
        raw = np.asarray(values, dtype=np.float32)
        if raw.shape != expected_shape:
            raise ValueError(
                f"extra expert shape {raw.shape} differs from {expected_shape}"
            )
        rank = (average_row_ranks(raw) / raw.shape[1]).astype(np.float32)
        low = raw.min(axis=1, keepdims=True)
        high = raw.max(axis=1, keepdims=True)
        minmax = (raw - low) / np.maximum(high - low, 1e-6)
        top_two = np.partition(raw, -2, axis=1)[:, -2:]
        margin = (top_two[:, 1] - top_two[:, 0]) / np.maximum(
            raw.std(axis=1), 1e-6
        )
        centered = rank - rank.mean(axis=1, keepdims=True)
        correlation = (centered * consensus_centered).mean(axis=1)
        correlation /= np.maximum(centered.std(axis=1) * consensus_std, 1e-6)
        top_col = np.argmax(rank, axis=1)
        top_consensus = consensus[np.arange(len(rank)), top_col]
        broadcast_shape = expected_shape
        columns.extend(
            [
                rank,
                minmax.astype(np.float32),
                rank * time_fraction,
                rank - base[..., 40],
                rank - base[..., 42],
                np.broadcast_to(margin[:, None], broadcast_shape),
                np.broadcast_to(correlation[:, None], broadcast_shape),
                np.broadcast_to(top_consensus[:, None], broadcast_shape),
            ]
        )
    result = np.stack(columns, axis=-1).astype(np.float32)
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)


def extra_expert_consensus_features(extra_features: np.ndarray) -> np.ndarray:
    """Summarize agreement among independently trained extra experts."""
    features = np.asarray(extra_features, dtype=np.float32)
    if features.ndim != 3 or features.shape[-1] % len(
        EXTRA_EXPERT_FEATURE_SUFFIXES
    ):
        raise ValueError("extra expert feature layout is invalid")
    ranks = features[..., :: len(EXTRA_EXPERT_FEATURE_SUFFIXES)]
    if ranks.shape[-1] < 2:
        raise ValueError("extra consensus requires at least two experts")
    ordered = np.sort(ranks, axis=-1)
    top3_count = min(3, ranks.shape[-1])
    output = np.stack(
        [
            ranks.mean(axis=-1),
            np.median(ranks, axis=-1),
            ranks.max(axis=-1),
            ranks.min(axis=-1),
            ranks.std(axis=-1),
            ordered[..., -2:].mean(axis=-1),
            ordered[..., -top3_count:].mean(axis=-1),
        ],
        axis=-1,
    ).astype(np.float32)
    return np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
