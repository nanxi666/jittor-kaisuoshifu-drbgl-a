from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .low_rank import average_row_ranks
from ..baselines.depop_v3 import (
    candidate_exposure_features,
    local_pair_exposure,
    pair_exposure,
)


@dataclass(frozen=True)
class ListwiseContext:
    expert_scores: Sequence[np.ndarray]
    exposure_scores: Sequence[np.ndarray]
    query_time: np.ndarray

    def __post_init__(self) -> None:
        if len(self.expert_scores) < 2:
            raise ValueError("at least two expert matrices are required")
        if len(self.exposure_scores) != 5:
            raise ValueError("exactly five exposure matrices are required")
        shape = np.shape(self.expert_scores[0])
        matrices = [*self.expert_scores, *self.exposure_scores]
        if any(np.shape(value) != shape for value in matrices):
            raise ValueError("all candidate matrices must have identical shapes")
        if len(self.query_time) != shape[0]:
            raise ValueError("query times and candidate matrices differ in length")


def transductive_exposure_scores(
    query_src: np.ndarray,
    query_time: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return label-free exposure views computed from the candidate batch."""
    global_exposure, local4 = candidate_exposure_features(
        query_time, candidates, 4
    )
    _, local16 = candidate_exposure_features(query_time, candidates, 16)
    global_pair = pair_exposure(query_src, candidates)
    local_pair = local_pair_exposure(query_src, query_time, candidates, 8)
    return tuple(
        np.asarray(value, dtype=np.float32)
        for value in (
            global_exposure,
            local4,
            local16,
            global_pair,
            local_pair,
        )
    )


def _row_minmax(values: np.ndarray) -> np.ndarray:
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return (values - low) / np.maximum(high - low, 1e-6)


def listwise_features(
    context: ListwiseContext,
    rows: np.ndarray,
) -> np.ndarray:
    """Build candidate and query-level scale-free multi-expert features."""
    rows = np.asarray(rows, dtype=np.int64)
    raw = np.stack(
        [
            np.asarray(scores[rows], dtype=np.float32)
            for scores in context.expert_scores
        ],
        axis=-1,
    )
    ranks = np.stack(
        [
            average_row_ranks(raw[..., expert]) / raw.shape[1]
            for expert in range(raw.shape[-1])
        ],
        axis=-1,
    ).astype(np.float32)
    minmax = _row_minmax(raw)

    # Query-level confidence lets the ranker change expert weights when a
    # model is decisive or when its preferred candidate agrees with peers.
    top_two = np.partition(raw, -2, axis=1)
    margin = (top_two[:, -1, :] - top_two[:, -2, :]) / np.maximum(
        raw.std(axis=1), 1e-6
    )
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    consensus_centered = centered.mean(axis=-1)
    correlation = (centered * consensus_centered[..., None]).mean(axis=1)
    correlation /= np.maximum(
        centered.std(axis=1) * consensus_centered.std(axis=1)[:, None],
        1e-6,
    )
    top_candidate = np.argmax(ranks, axis=1)
    consensus = ranks.mean(axis=-1)
    top_consensus = np.take_along_axis(
        consensus[..., None], top_candidate[:, None, :], axis=1
    )[:, 0, :]
    query_features = np.concatenate(
        [margin, correlation, top_consensus], axis=1
    )
    query_features = np.broadcast_to(
        query_features[:, None, :],
        (len(rows), raw.shape[1], query_features.shape[1]),
    ).astype(np.float32)

    exposure = np.stack(
        [scores[rows] for scores in context.exposure_scores], axis=-1
    ).astype(np.float32)
    exposure_ranks = np.stack(
        [
            average_row_ranks(exposure[..., feature]) / exposure.shape[1]
            for feature in range(exposure.shape[-1])
        ],
        axis=-1,
    ).astype(np.float32)
    exposure_scale = np.log1p(np.maximum(exposure, 0.0))
    exposure_scale /= np.maximum(
        exposure_scale.max(axis=1, keepdims=True), 1e-6
    )

    votes = np.concatenate(
        [
            (ranks >= threshold).sum(axis=-1, keepdims=True)
            for threshold in (0.995, 0.975, 0.895)
        ],
        axis=-1,
    ).astype(np.float32)
    query_time = context.query_time[rows].astype(np.float64)
    time_min = float(context.query_time.min())
    time_span = max(float(context.query_time.max()) - time_min, 1.0)
    time_fraction = ((query_time - time_min) / time_span)[:, None, None]
    time_fraction = np.broadcast_to(
        time_fraction, (len(rows), raw.shape[1], 1)
    ).astype(np.float32)

    return np.concatenate(
        [
            ranks,
            minmax,
            exposure_ranks,
            exposure_scale,
            ranks.mean(axis=-1, keepdims=True),
            ranks.std(axis=-1, keepdims=True),
            ranks.max(axis=-1, keepdims=True),
            ranks.min(axis=-1, keepdims=True),
            votes,
            time_fraction,
            ranks * time_fraction,
            query_features,
        ],
        axis=-1,
    ).astype(np.float32)


def fit_listwise_gate(
    training_sets: Sequence[
        tuple[ListwiseContext, np.ndarray]
        | tuple[ListwiseContext, np.ndarray, np.ndarray]
    ],
    *,
    trees: int = 450,
    seed: int = 42,
) -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("LightGBM is required for the listwise gate") from exc

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for training_set in training_sets:
        if len(training_set) == 2:
            context, rows = training_set
            positive_col = None
        elif len(training_set) == 3:
            context, rows, positive_col = training_set
        else:
            raise ValueError(
                "training sets require context, rows, and optional positive "
                "columns"
            )
        block = listwise_features(context, rows)
        target = np.zeros(block.shape[:2], dtype=np.int8)
        if positive_col is None:
            target[:, 0] = 1
        else:
            positive_col = np.asarray(positive_col, dtype=np.int64)
            if len(positive_col) != len(context.query_time):
                raise ValueError("positive columns and context differ in length")
            selected_col = positive_col[np.asarray(rows, dtype=np.int64)]
            if np.any((selected_col < 0) | (selected_col >= block.shape[1])):
                raise ValueError("positive column is outside the candidate matrix")
            target[np.arange(len(rows)), selected_col] = 1
        features.append(block.reshape(-1, block.shape[-1]))
        labels.append(target.ravel())
        groups.append(np.full(len(rows), block.shape[1], dtype=np.int32))
    if not features:
        raise ValueError("at least one training set is required")

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1],
        n_estimators=trees,
        learning_rate=0.03,
        num_leaves=47,
        max_depth=8,
        min_child_samples=200,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        lambdarank_truncation_level=20,
    )
    model.fit(
        np.concatenate(features),
        np.concatenate(labels),
        group=np.concatenate(groups),
    )
    return model


def predict_listwise_gate(
    model: Any,
    context: ListwiseContext,
    *,
    rows: np.ndarray | None = None,
    batch_size: int = 8192,
    output_path: Path | None = None,
) -> np.ndarray:
    if rows is None:
        rows = np.arange(len(context.query_time), dtype=np.int64)
    else:
        rows = np.asarray(rows, dtype=np.int64)
    candidate_count = np.shape(context.expert_scores[0])[1]
    shape = (len(rows), candidate_count)
    if output_path is None:
        output = np.empty(shape, dtype=np.float32)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=shape
        )

    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        features = listwise_features(context, rows[start:stop])
        output[start:stop] = model.predict(
            features.reshape(-1, features.shape[-1])
        ).reshape(stop - start, candidate_count)
    if isinstance(output, np.memmap):
        output.flush()
    return output
