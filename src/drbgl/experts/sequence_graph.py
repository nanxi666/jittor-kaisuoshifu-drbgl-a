from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class SequenceGraphConfig:
    recent_items: int = 400
    decay: float = 1.0
    row_normalize: bool = True
    remove_self_transitions: bool = True


@dataclass
class SequenceTransitionExpert:
    """Fixed-history directed item propagation with compact node IDs."""

    users: np.ndarray
    user_starts: np.ndarray
    user_counts: np.ndarray
    history_items: np.ndarray
    items: np.ndarray
    transitions: sparse.csr_matrix
    config: SequenceGraphConfig

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        cutoff: int,
        config: SequenceGraphConfig | None = None,
    ) -> "SequenceTransitionExpert":
        config = config or SequenceGraphConfig()
        if config.recent_items <= 0:
            raise ValueError("recent_items must be positive")
        if not 0.0 < config.decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")

        mask = train["time"].to_numpy(np.int64) < cutoff
        src = train.loc[mask, "src"].to_numpy(np.int64)
        dst = train.loc[mask, "dst"].to_numpy(np.int64)
        event_time = train.loc[mask, "time"].to_numpy(np.int64)
        if len(src) == 0:
            raise ValueError("no training interactions precede the cutoff")

        order = np.lexsort((event_time, src))
        src = src[order]
        dst = dst[order]
        users, user_starts, user_counts = np.unique(
            src, return_index=True, return_counts=True
        )
        items, history_items = np.unique(dst, return_inverse=True)
        same_user = src[1:] == src[:-1]
        previous = history_items[:-1][same_user]
        following = history_items[1:][same_user]
        if config.remove_self_transitions:
            nonself = previous != following
            previous = previous[nonself]
            following = following[nonself]
        transitions = sparse.csr_matrix(
            (
                np.ones(len(previous), dtype=np.float32),
                (previous, following),
            ),
            shape=(len(items), len(items)),
        )
        transitions.sum_duplicates()
        transitions.data = np.log1p(transitions.data).astype(np.float32)
        if config.row_normalize:
            degree = np.asarray(transitions.sum(axis=1)).ravel()
            transitions = (
                sparse.diags(1.0 / np.maximum(degree, 1.0)) @ transitions
            ).tocsr()

        return cls(
            users=users,
            user_starts=user_starts,
            user_counts=user_counts,
            history_items=history_items.astype(np.int32, copy=False),
            items=items,
            transitions=transitions,
            config=config,
        )

    def score(
        self,
        test: pd.DataFrame,
        candidate_columns: list[str],
    ) -> np.ndarray:
        query_sources = test["src"].to_numpy(np.int64)
        candidates = test[candidate_columns].to_numpy(np.int64)
        positions = np.searchsorted(self.items, candidates)
        clipped = np.minimum(positions, len(self.items) - 1)
        valid = (positions < len(self.items)) & (
            self.items[clipped] == candidates
        )
        safe_positions = np.where(valid, clipped, 0)
        scores = np.zeros(candidates.shape, dtype=np.float32)

        query_order = np.argsort(query_sources, kind="stable")
        ordered_sources = query_sources[query_order]
        _, group_starts, group_counts = np.unique(
            ordered_sources, return_index=True, return_counts=True
        )
        for group_start, group_count in zip(group_starts, group_counts):
            group_stop = group_start + group_count
            group_rows = query_order[group_start:group_stop]
            source = ordered_sources[group_start]
            user_position = np.searchsorted(self.users, source)
            if (
                user_position == len(self.users)
                or self.users[user_position] != source
            ):
                continue

            history_stop = (
                self.user_starts[user_position]
                + self.user_counts[user_position]
            )
            history_start = max(
                self.user_starts[user_position],
                history_stop - self.config.recent_items,
            )
            history = self.history_items[history_start:history_stop]
            weights = np.power(
                self.config.decay,
                np.arange(len(history) - 1, -1, -1, dtype=np.float32),
            )

            block_positions = safe_positions[group_rows]
            block_valid = valid[group_rows]
            unique_candidates = np.unique(block_positions[block_valid])
            if len(unique_candidates) == 0:
                continue
            propagated = np.asarray(
                weights @ self.transitions[history][:, unique_candidates]
            ).ravel()
            candidate_offsets = np.searchsorted(
                unique_candidates, block_positions
            )
            block_scores = propagated[candidate_offsets]
            block_scores[~block_valid] = 0.0
            scores[group_rows] = block_scores
        return scores
