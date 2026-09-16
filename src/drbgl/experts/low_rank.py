from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def average_row_ranks(
    scores: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    """Return ascending average ranks without inventing order among ties."""
    values = np.asarray(scores)
    ranks = np.empty(values.shape, dtype=np.float32)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        block = values[start:stop]
        order = np.argsort(block, axis=1, kind="stable")
        ordered = np.take_along_axis(block, order, axis=1)
        new_group = np.ones(ordered.shape, dtype=bool)
        new_group[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
        group = np.cumsum(new_group, axis=1, dtype=np.int32) - 1

        rows = len(block)
        columns = block.shape[1]
        keys = (
            np.arange(rows, dtype=np.int64)[:, None] * columns + group
        )
        flat_keys = keys.ravel()
        counts = np.bincount(flat_keys, minlength=rows * columns)
        rank_sum = np.bincount(
            flat_keys,
            weights=np.broadcast_to(
                np.arange(1, columns + 1, dtype=np.float64),
                ordered.shape,
            ).ravel(),
            minlength=rows * columns,
        )
        average = rank_sum / np.maximum(counts, 1)
        ordered_ranks = average[flat_keys].reshape(ordered.shape)
        block_ranks = np.empty(ordered.shape, dtype=np.float32)
        np.put_along_axis(
            block_ranks,
            order,
            ordered_ranks.astype(np.float32),
            axis=1,
        )
        ranks[start:stop] = block_ranks
    return ranks


def normalized_svd_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate_columns: list[str],
    rank: int = 64,
    degree_power: float = 0.5,
    n_iter: int = 5,
    seed: int = 42,
    batch_size: int = 4096,
    half_life_days: float | None = None,
    user_degree_power: float | None = None,
    item_degree_power: float | None = None,
) -> np.ndarray:
    """Low-rank scores from a degree-normalized frozen interaction graph."""
    if user_degree_power is None:
        user_degree_power = degree_power
    if item_degree_power is None:
        item_degree_power = degree_power
    src = train["src"].to_numpy(np.int64)
    dst = train["dst"].to_numpy(np.int64)
    users, user_index = np.unique(src, return_inverse=True)
    items, item_index = np.unique(dst, return_inverse=True)
    if half_life_days is None:
        values = np.ones(len(train), dtype=np.float32)
    else:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        reference_time = float(test["time"].min())
        age_days = np.maximum(
            reference_time - train["time"].to_numpy(np.float64), 0.0
        ) / 86400.0
        values = np.exp2(-age_days / half_life_days).astype(np.float32)
    interactions = sparse.csr_matrix(
        (
            values,
            (user_index, item_index),
        ),
        shape=(len(users), len(items)),
    )
    interactions.sum_duplicates()
    if half_life_days is None:
        interactions.data[:] = 1.0
    user_degree = np.asarray(interactions.sum(axis=1)).ravel()
    item_degree = np.asarray(interactions.sum(axis=0)).ravel()
    degree_floor = 1.0 if half_life_days is None else 1e-12
    normalized = sparse.diags(
        np.power(np.maximum(user_degree, degree_floor), -user_degree_power)
    ) @ interactions @ sparse.diags(
        np.power(np.maximum(item_degree, degree_floor), -item_degree_power)
    )
    model = TruncatedSVD(
        n_components=rank,
        n_iter=n_iter,
        random_state=seed,
    )
    user_latent = model.fit_transform(normalized).astype(np.float32)
    item_latent = model.components_.T.astype(np.float32)

    query_users = test["src"].to_numpy(np.int64)
    candidates = test[candidate_columns].to_numpy(np.int64)
    user_positions = np.searchsorted(users, query_users)
    clipped_users = np.minimum(user_positions, len(users) - 1)
    valid_users = (user_positions < len(users)) & (
        users[clipped_users] == query_users
    )
    item_positions = np.searchsorted(items, candidates)
    clipped_items = np.minimum(item_positions, len(items) - 1)
    valid_items = (item_positions < len(items)) & (
        items[clipped_items] == candidates
    )
    scores = np.zeros(candidates.shape, dtype=np.float32)
    for start in range(0, len(test), batch_size):
        stop = min(start + batch_size, len(test))
        block = np.einsum(
            "br,bcr->bc",
            user_latent[clipped_users[start:stop]],
            item_latent[clipped_items[start:stop]],
            optimize=True,
        )
        block[~valid_items[start:stop]] = 0.0
        block[~valid_users[start:stop]] = 0.0
        scores[start:stop] = block
    return scores
