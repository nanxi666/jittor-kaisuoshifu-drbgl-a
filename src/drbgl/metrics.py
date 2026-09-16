from __future__ import annotations

import numpy as np


def positive_ranks(scores: np.ndarray, positive_col: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with larger scores ranked first."""
    scores = np.asarray(scores)
    positive_col = np.asarray(positive_col, dtype=np.int64)
    if scores.ndim != 2 or len(positive_col) != len(scores):
        raise ValueError("scores must be [queries, candidates] and positive_col [queries]")
    pos = scores[np.arange(len(scores)), positive_col][:, None]
    higher = (scores > pos).sum(axis=1)
    equal = (scores == pos).sum(axis=1)
    return higher + (equal + 1.0) / 2.0


def mean_reciprocal_rank(scores: np.ndarray, positive_col: np.ndarray) -> float:
    return float(np.mean(1.0 / positive_ranks(scores, positive_col)))


def ordinal_rank_scores(scores: np.ndarray, candidates: np.ndarray | None = None) -> np.ndarray:
    """Map every row to 0.01..1.00, preserving larger-is-better ordering."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 100:
        raise ValueError(f"expected [N, 100] scores, got {values.shape}")
    values = np.nan_to_num(values, nan=-1e30, posinf=1e30, neginf=-1e30)
    if candidates is not None:
        cands = np.asarray(candidates, dtype=np.uint64)
        if cands.shape != values.shape:
            raise ValueError("candidate shape does not match score shape")
        tie = (cands * np.uint64(2654435761)) % np.uint64(100003)
        values = values + tie.astype(np.float64) * 1e-14
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty(order.shape, dtype=np.float32)
    rows = np.arange(len(values))[:, None]
    ranks[rows, order] = np.arange(1, 101, dtype=np.float32)[None, :]
    return ranks / 100.0


def blend_rank_scores(
    base: np.ndarray,
    challenger: np.ndarray,
    challenger_weight: float,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    if not 0.0 <= challenger_weight <= 1.0:
        raise ValueError("challenger_weight must be in [0, 1]")
    base_rank = ordinal_rank_scores(base, candidates)
    challenger_rank = ordinal_rank_scores(challenger, candidates)
    mixed = (1.0 - challenger_weight) * base_rank + challenger_weight * challenger_rank
    return ordinal_rank_scores(mixed, candidates)
