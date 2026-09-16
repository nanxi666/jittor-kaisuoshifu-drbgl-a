from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .low_rank import average_row_ranks


DAY = 86400
WINDOW_DAYS = (7, 30, 90, 180, 365, 730)
STRUCTURAL_FEATURE_NAMES = (
    "item_log_count",
    "item_log_count_7d",
    "item_log_count_30d",
    "item_log_count_90d",
    "item_log_count_180d",
    "item_log_count_365d",
    "item_log_count_730d",
    "item_log_gap_days",
    "item_log_age_days",
    "item_log_active_span_days",
    "item_recent_30_over_180",
    "item_recent_90_over_365",
    "cohort_log_count",
    "cohort_log_count_90d",
    "cohort_log_count_365d",
    "cohort_share",
    "source_log_count",
    "source_log_count_90d",
    "source_log_count_365d",
    "source_log_gap_days",
    "source_item_popularity_delta",
    "source_item_popularity_z",
    "source_audience_activity_delta",
    "pair_log_count",
    "pair_log_count_90d",
    "pair_log_count_365d",
    "pair_log_gap_days",
    "pair_share_of_source",
    "pair_share_of_item",
    "pair_gap_minus_source_gap",
    "is_source_most_recent_item",
    "is_historical_pair",
    "is_cold_item",
    "is_duplicate_in_row",
)
TRANSDUCTIVE_HISTORY_FEATURE_NAMES = (
    "future_global_log_excess",
    "future_local4_log_excess",
    "future_local16_log_excess",
    "future_pair_log_count",
    "future_local_pair_log_count",
    "future_global_minus_history_total",
    "future_global_minus_history_365d",
    "future_global_minus_history_cohort",
    "future_pair_minus_history_pair",
    "future_pair_minus_source_query_count",
    "future_global_rank_minus_history_rank",
    "future_pair_rank_minus_history_pair_rank",
    "cold_future_global_exposure",
    "repeat_future_pair_exposure",
)


def _compact_positions(ids: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.int64)
    positions = np.searchsorted(ids, values)
    safe = np.minimum(positions, max(len(ids) - 1, 0))
    valid = (positions < len(ids)) & (ids[safe] == values) if len(ids) else np.zeros(values.shape, bool)
    return safe, valid


def _counts_by_window(
    compact_ids: np.ndarray,
    event_time: np.ndarray,
    size: int,
    cutoff: int,
) -> dict[int, np.ndarray]:
    result = {}
    for days in WINDOW_DAYS:
        mask = event_time >= cutoff - days * DAY
        result[days] = np.bincount(
            compact_ids[mask], minlength=size
        ).astype(np.float32)
    return result


def _pair_keys(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.uint64)
    destination = np.asarray(destination, dtype=np.uint64)
    if np.any(source > np.iinfo(np.uint32).max) or np.any(
        destination > np.iinfo(np.uint32).max
    ):
        raise ValueError("structural pair ids must fit in unsigned 32 bits")
    return (source << np.uint64(32)) | destination


@dataclass(frozen=True)
class StructuralFeatureIndex:
    cutoff: int
    items: np.ndarray
    sources: np.ndarray
    item_count: np.ndarray
    item_window_count: dict[int, np.ndarray]
    item_first_time: np.ndarray
    item_last_time: np.ndarray
    cohort_count: np.ndarray
    cohort_window_count: dict[int, np.ndarray]
    source_count: np.ndarray
    source_window_count: dict[int, np.ndarray]
    source_last_time: np.ndarray
    source_popularity_mean: np.ndarray
    source_popularity_std: np.ndarray
    item_audience_mean: np.ndarray
    pair_keys: np.ndarray
    pair_count: np.ndarray
    pair_window_count: dict[int, np.ndarray]
    pair_last_time: np.ndarray

    @classmethod
    def fit(
        cls,
        history: pd.DataFrame,
        query_sources: np.ndarray,
        cutoff: int,
    ) -> "StructuralFeatureIndex":
        frozen = history[history["time"] < cutoff]
        if frozen.empty:
            raise ValueError("structural history is empty")
        src = frozen["src"].to_numpy(np.int64)
        dst = frozen["dst"].to_numpy(np.int64)
        event_time = frozen["time"].to_numpy(np.int64)
        sources, source_index = np.unique(src, return_inverse=True)
        items, item_index = np.unique(dst, return_inverse=True)

        item_count = np.bincount(item_index, minlength=len(items)).astype(np.float32)
        source_count = np.bincount(source_index, minlength=len(sources)).astype(np.float32)
        item_window = _counts_by_window(
            item_index, event_time, len(items), cutoff
        )
        source_window = _counts_by_window(
            source_index, event_time, len(sources), cutoff
        )
        item_first = np.full(len(items), np.iinfo(np.int64).max, dtype=np.int64)
        item_last = np.full(len(items), np.iinfo(np.int64).min, dtype=np.int64)
        source_last = np.full(len(sources), np.iinfo(np.int64).min, dtype=np.int64)
        np.minimum.at(item_first, item_index, event_time)
        np.maximum.at(item_last, item_index, event_time)
        np.maximum.at(source_last, source_index, event_time)

        cohort_mask = np.isin(src, np.unique(np.asarray(query_sources, dtype=np.int64)))
        cohort_count = np.bincount(
            item_index[cohort_mask], minlength=len(items)
        ).astype(np.float32)
        cohort_window = _counts_by_window(
            item_index[cohort_mask],
            event_time[cohort_mask],
            len(items),
            cutoff,
        )

        item_log_popularity = np.log1p(item_count[item_index]).astype(np.float64)
        source_pop_sum = np.bincount(
            source_index, weights=item_log_popularity, minlength=len(sources)
        )
        source_pop_sq = np.bincount(
            source_index,
            weights=item_log_popularity * item_log_popularity,
            minlength=len(sources),
        )
        source_pop_mean = source_pop_sum / np.maximum(source_count, 1.0)
        source_pop_var = source_pop_sq / np.maximum(source_count, 1.0)
        source_pop_std = np.sqrt(
            np.maximum(source_pop_var - source_pop_mean * source_pop_mean, 1e-4)
        )

        source_activity = np.log1p(source_count[source_index]).astype(np.float64)
        item_audience_sum = np.bincount(
            item_index, weights=source_activity, minlength=len(items)
        )
        item_audience_mean = item_audience_sum / np.maximum(item_count, 1.0)

        pair_keys, pair_index = np.unique(
            _pair_keys(src, dst), return_inverse=True
        )
        pair_count = np.bincount(
            pair_index, minlength=len(pair_keys)
        ).astype(np.float32)
        pair_window = _counts_by_window(
            pair_index, event_time, len(pair_keys), cutoff
        )
        pair_last = np.full(
            len(pair_keys), np.iinfo(np.int64).min, dtype=np.int64
        )
        np.maximum.at(pair_last, pair_index, event_time)
        return cls(
            cutoff=int(cutoff),
            items=items,
            sources=sources,
            item_count=item_count,
            item_window_count=item_window,
            item_first_time=item_first,
            item_last_time=item_last,
            cohort_count=cohort_count,
            cohort_window_count=cohort_window,
            source_count=source_count,
            source_window_count=source_window,
            source_last_time=source_last,
            source_popularity_mean=source_pop_mean.astype(np.float32),
            source_popularity_std=source_pop_std.astype(np.float32),
            item_audience_mean=item_audience_mean.astype(np.float32),
            pair_keys=pair_keys,
            pair_count=pair_count,
            pair_window_count=pair_window,
            pair_last_time=pair_last,
        )

    def transform(
        self,
        query_src: np.ndarray,
        query_time: np.ndarray,
        candidates: np.ndarray,
    ) -> np.ndarray:
        query_src = np.asarray(query_src, dtype=np.int64)
        query_time = np.asarray(query_time, dtype=np.int64)
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.ndim != 2 or candidates.shape[0] != len(query_src):
            raise ValueError("structural query arrays do not align")
        if len(query_time) != len(query_src):
            raise ValueError("structural query times do not align")
        if len(query_time) and int(query_time.min()) < self.cutoff:
            raise ValueError("a structural query predates its history cutoff")

        item_pos, item_valid = _compact_positions(self.items, candidates)
        source_pos, source_valid = _compact_positions(self.sources, query_src)

        def item_values(values: np.ndarray) -> np.ndarray:
            return np.where(item_valid, values[item_pos], 0.0).astype(np.float32)

        def source_values(values: np.ndarray) -> np.ndarray:
            return np.where(source_valid, values[source_pos], 0.0).astype(np.float32)

        item_count = item_values(self.item_count)
        item_windows = {
            days: item_values(values)
            for days, values in self.item_window_count.items()
        }
        cohort_count = item_values(self.cohort_count)
        cohort_90 = item_values(self.cohort_window_count[90])
        cohort_365 = item_values(self.cohort_window_count[365])
        repeated_time = np.broadcast_to(query_time[:, None], candidates.shape)
        first_time = np.where(item_valid, self.item_first_time[item_pos], self.cutoff)
        last_time = np.where(item_valid, self.item_last_time[item_pos], -(10**18))
        gap_days = np.where(
            item_valid,
            np.maximum(repeated_time - last_time, 0) / DAY,
            1e6,
        )
        age_days = np.where(
            item_valid,
            np.maximum(repeated_time - first_time, 0) / DAY,
            0.0,
        )
        active_span = np.where(
            item_valid,
            np.maximum(last_time - first_time, 0) / DAY,
            0.0,
        )

        source_count = source_values(self.source_count)
        source_90 = source_values(self.source_window_count[90])
        source_365 = source_values(self.source_window_count[365])
        source_last = np.where(
            source_valid, self.source_last_time[source_pos], -(10**18)
        )
        source_gap = np.where(
            source_valid,
            np.maximum(query_time - source_last, 0) / DAY,
            1e6,
        )
        source_pop_mean = source_values(self.source_popularity_mean)[:, None]
        source_pop_std = source_values(self.source_popularity_std)[:, None]
        item_log_count = np.log1p(item_count)
        popularity_delta = item_log_count - source_pop_mean
        popularity_z = popularity_delta / np.maximum(source_pop_std, 0.01)
        audience_delta = item_values(self.item_audience_mean) - np.log1p(
            source_count
        )[:, None]

        pair_key = _pair_keys(query_src[:, None], candidates)
        pair_pos = np.searchsorted(self.pair_keys, pair_key)
        pair_safe = np.minimum(pair_pos, len(self.pair_keys) - 1)
        historical_pair = (pair_pos < len(self.pair_keys)) & (
            self.pair_keys[pair_safe] == pair_key
        )
        pair_count = np.where(
            historical_pair, self.pair_count[pair_safe], 0.0
        ).astype(np.float32)
        pair_90 = np.where(
            historical_pair,
            self.pair_window_count[90][pair_safe],
            0.0,
        ).astype(np.float32)
        pair_365 = np.where(
            historical_pair,
            self.pair_window_count[365][pair_safe],
            0.0,
        ).astype(np.float32)
        pair_last = np.where(
            historical_pair, self.pair_last_time[pair_safe], -(10**18)
        )
        pair_gap = np.where(
            historical_pair,
            np.maximum(repeated_time - pair_last, 0) / DAY,
            1e6,
        )
        pair_share_source = pair_count / np.maximum(source_count[:, None], 1.0)
        pair_share_item = pair_count / np.maximum(item_count, 1.0)
        pair_gap_minus_source = np.log1p(pair_gap) - np.log1p(source_gap)[:, None]
        source_most_recent = historical_pair & (
            pair_last == source_last[:, None]
        )
        candidate_order = np.argsort(candidates, axis=1, kind="stable")
        sorted_candidates = np.take_along_axis(candidates, candidate_order, axis=1)
        adjacent_equal = sorted_candidates[:, 1:] == sorted_candidates[:, :-1]
        sorted_duplicate = np.zeros(candidates.shape, dtype=np.float32)
        sorted_duplicate[:, 1:] = adjacent_equal
        sorted_duplicate[:, :-1] = np.maximum(
            sorted_duplicate[:, :-1], adjacent_equal
        )
        duplicate = np.zeros(candidates.shape, dtype=np.float32)
        np.put_along_axis(duplicate, candidate_order, sorted_duplicate, axis=1)

        columns = (
            item_log_count,
            *(np.log1p(item_windows[days]) for days in WINDOW_DAYS),
            np.log1p(gap_days),
            np.log1p(age_days),
            np.log1p(active_span),
            (item_windows[30] + 1.0) / (item_windows[180] + 6.0),
            (item_windows[90] + 1.0) / (item_windows[365] + 4.0),
            np.log1p(cohort_count),
            np.log1p(cohort_90),
            np.log1p(cohort_365),
            (cohort_count + 1.0) / (item_count + 2.0),
            np.broadcast_to(np.log1p(source_count)[:, None], candidates.shape),
            np.broadcast_to(np.log1p(source_90)[:, None], candidates.shape),
            np.broadcast_to(np.log1p(source_365)[:, None], candidates.shape),
            np.broadcast_to(np.log1p(source_gap)[:, None], candidates.shape),
            popularity_delta,
            popularity_z,
            audience_delta,
            np.log1p(pair_count),
            np.log1p(pair_90),
            np.log1p(pair_365),
            np.log1p(pair_gap),
            pair_share_source,
            pair_share_item,
            pair_gap_minus_source,
            source_most_recent.astype(np.float32),
            historical_pair.astype(np.float32),
            (~item_valid).astype(np.float32),
            duplicate,
        )
        output = np.stack(columns, axis=-1).astype(np.float32)
        if output.shape[-1] != len(STRUCTURAL_FEATURE_NAMES):
            raise AssertionError("structural feature names and columns differ")
        return np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)


def structural_rank_features(raw_features: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_features, dtype=np.float32)
    if raw.ndim != 3:
        raise ValueError("structural features must be [queries, candidates, features]")
    ranks = np.stack(
        [
            average_row_ranks(raw[..., feature]) / raw.shape[1]
            for feature in range(raw.shape[-1])
        ],
        axis=-1,
    ).astype(np.float32)
    low = raw.min(axis=1, keepdims=True)
    high = raw.max(axis=1, keepdims=True)
    minmax = (raw - low) / np.maximum(high - low, 1e-6)
    return np.concatenate([raw, ranks, minmax], axis=-1).astype(np.float32)


def transductive_history_features(
    structural_features: np.ndarray,
    exposure_scores: tuple[np.ndarray, ...] | list[np.ndarray],
    source_query_count: np.ndarray,
) -> np.ndarray:
    """Combine future candidate exposure with frozen historical statistics."""
    structural = np.asarray(structural_features, dtype=np.float32)
    if structural.ndim != 3:
        raise ValueError("structural features must be [queries, candidates, features]")
    if len(exposure_scores) != 5:
        raise ValueError("exactly five exposure matrices are required")
    exposure = np.stack(
        [np.asarray(values, dtype=np.float32) for values in exposure_scores],
        axis=-1,
    )
    if exposure.shape[:2] != structural.shape[:2]:
        raise ValueError("exposure and structural candidate matrices differ")
    source_query_count = np.asarray(source_query_count, dtype=np.float32)
    if source_query_count.shape != (len(structural),):
        raise ValueError("source query counts must have one value per query")

    exposure_log = np.log1p(np.maximum(exposure, 0.0))
    item_total = structural[
        ..., STRUCTURAL_FEATURE_NAMES.index("item_log_count")
    ]
    item_365 = structural[
        ..., STRUCTURAL_FEATURE_NAMES.index("item_log_count_365d")
    ]
    cohort_total = structural[
        ..., STRUCTURAL_FEATURE_NAMES.index("cohort_log_count")
    ]
    pair_history = structural[
        ..., STRUCTURAL_FEATURE_NAMES.index("pair_log_count")
    ]
    cold = structural[..., STRUCTURAL_FEATURE_NAMES.index("is_cold_item")]
    repeat = structural[
        ..., STRUCTURAL_FEATURE_NAMES.index("is_historical_pair")
    ]
    global_rank = average_row_ranks(exposure[..., 0]) / structural.shape[1]
    pair_rank = average_row_ranks(exposure[..., 3]) / structural.shape[1]
    item_rank = average_row_ranks(item_total) / structural.shape[1]
    pair_history_rank = average_row_ranks(pair_history) / structural.shape[1]
    source_query_log = np.log1p(source_query_count)[:, None]

    output = np.stack(
        [
            *(exposure_log[..., feature] for feature in range(5)),
            exposure_log[..., 0] - item_total,
            exposure_log[..., 0] - item_365,
            exposure_log[..., 0] - cohort_total,
            exposure_log[..., 3] - pair_history,
            exposure_log[..., 3] - source_query_log,
            global_rank - item_rank,
            pair_rank - pair_history_rank,
            cold * exposure_log[..., 0],
            repeat * exposure_log[..., 3],
        ],
        axis=-1,
    ).astype(np.float32)
    if output.shape[-1] != len(TRANSDUCTIVE_HISTORY_FEATURE_NAMES):
        raise AssertionError("transductive feature names and columns differ")
    return np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
