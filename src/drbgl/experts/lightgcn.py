"""Jittor 版 BPR-LightGCN 训练与 SciPy 版的图层传播。

方法来源（仅参考论文思路，未复制上游代码）:
    He et al., "LightGCN: Simplifying and Powering Graph Convolution
    for Recommendation", SIGIR 2020. 许可证与来源见仓库根目录 NOTICE。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class CompactBipartiteGraph:
    users: np.ndarray
    items: np.ndarray
    interactions: sparse.csr_matrix
    normalized: sparse.csr_matrix


def build_compact_graph(
    train: pd.DataFrame,
    cutoff: int,
) -> CompactBipartiteGraph:
    mask = train["time"].to_numpy(np.int64) < cutoff
    src = train.loc[mask, "src"].to_numpy(np.int64)
    dst = train.loc[mask, "dst"].to_numpy(np.int64)
    if len(src) == 0:
        raise ValueError("no training interactions precede the cutoff")
    users, user_index = np.unique(src, return_inverse=True)
    items, item_index = np.unique(dst, return_inverse=True)
    interactions = sparse.csr_matrix(
        (
            np.ones(len(src), dtype=np.float32),
            (user_index, item_index),
        ),
        shape=(len(users), len(items)),
    )
    interactions.sum_duplicates()
    interactions.data[:] = 1.0
    user_degree = np.asarray(interactions.sum(axis=1)).ravel()
    item_degree = np.asarray(interactions.sum(axis=0)).ravel()
    normalized = (
        sparse.diags(np.power(np.maximum(user_degree, 1.0), -0.5))
        @ interactions
        @ sparse.diags(np.power(np.maximum(item_degree, 1.0), -0.5))
    ).tocsr().astype(np.float32)
    return CompactBipartiteGraph(users, items, interactions, normalized)


def sample_unobserved_items(
    user_index: np.ndarray,
    item_index: np.ndarray,
    item_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform negatives with vectorized rejection of observed compact edges."""
    if item_count <= 0:
        raise ValueError("item_count must be positive")
    positive_keys = np.unique(
        user_index.astype(np.int64) * item_count + item_index
    )
    if len(positive_keys) == 0:
        return np.empty(0, dtype=np.int64)
    positive_users = positive_keys // item_count
    user_degree = np.bincount(positive_users)
    requested_users = np.unique(user_index)
    if np.any(user_degree[requested_users] >= item_count):
        raise ValueError("cannot sample a negative for a fully connected user")
    negative = rng.integers(
        0, item_count, size=len(user_index), dtype=np.int64
    )
    while True:
        keys = user_index.astype(np.int64) * item_count + negative
        positions = np.searchsorted(positive_keys, keys)
        clipped = np.minimum(positions, len(positive_keys) - 1)
        collision = (positions < len(positive_keys)) & (
            positive_keys[clipped] == keys
        )
        if not np.any(collision):
            return negative
        negative[collision] = rng.integers(
            0, item_count, size=int(collision.sum()), dtype=np.int64
        )


def train_bpr_embeddings(
    graph: CompactBipartiteGraph,
    *,
    embedding_dim: int = 64,
    epochs: int = 5,
    batch_size: int = 65536,
    learning_rate: float = 0.02,
    regularization: float = 1e-6,
    seed: int = 42,
    threads: int = 8,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train BPR embeddings with Jittor.

    LightGCN propagation remains a deterministic SciPy sparse operation after
    training. This keeps the graph pass memory bounded while all learnable
    parameters and optimization are owned by Jittor.
    """
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    import jittor as jt

    jt.set_global_seed(seed)
    if device == "cuda":
        if not bool(getattr(jt.compiler, "has_cuda", False)):
            raise RuntimeError("CUDA was requested but this Jittor build has no CUDA")
        jt.flags.use_cuda = 1
    elif device == "cpu":
        jt.flags.use_cuda = 0
    else:
        jt.flags.use_cuda = int(bool(getattr(jt.compiler, "has_cuda", False)))

    users, items = graph.interactions.nonzero()
    users = users.astype(np.int64, copy=False)
    items = items.astype(np.int64, copy=False)
    user_embedding = jt.init.gauss(
        (len(graph.users), embedding_dim), dtype="float32", std=0.1
    )
    item_embedding = jt.init.gauss(
        (len(graph.items), embedding_dim), dtype="float32", std=0.1
    )
    user_embedding.name("lightgcn_user_embedding")
    item_embedding.name("lightgcn_item_embedding")
    optimizer = jt.optim.Adam(
        [user_embedding, item_embedding], lr=learning_rate
    )
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(epochs):
        negative = sample_unobserved_items(
            users, items, len(graph.items), rng
        )
        permutation = rng.permutation(len(users))
        total_loss = 0.0
        total_rows = 0
        for start in range(0, len(users), batch_size):
            batch = permutation[start : start + batch_size]
            batch_users = jt.array(users[batch]).int64()
            batch_items = jt.array(items[batch]).int64()
            batch_negative = jt.array(negative[batch]).int64()
            user_values = user_embedding[batch_users]
            positive_values = item_embedding[batch_items]
            negative_values = item_embedding[batch_negative]
            positive_score = (user_values * positive_values).sum(dim=1)
            negative_score = (user_values * negative_values).sum(dim=1)
            ranking_loss = jt.nn.softplus(
                negative_score - positive_score
            ).mean()
            penalty = regularization * (
                (user_values * user_values).sum(dim=1)
                + (positive_values * positive_values).sum(dim=1)
                + (negative_values * negative_values).sum(dim=1)
            ).mean()
            loss = ranking_loss + penalty
            optimizer.step(loss)
            rows = len(batch)
            total_loss += float(loss.item()) * rows
            total_rows += rows
        losses.append(total_loss / total_rows)
        print(
            f"BPR epoch {epoch + 1}/{epochs} loss={losses[-1]:.6f}",
            flush=True,
        )
    jt.sync_all(True)
    return (
        user_embedding.numpy().astype(np.float32, copy=False),
        item_embedding.numpy().astype(np.float32, copy=False),
        losses,
    )


def propagate_lightgcn(
    graph: CompactBipartiteGraph,
    user_embedding: np.ndarray,
    item_embedding: np.ndarray,
    layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the official LightGCN mean-of-layers rule once and cache it."""
    if layers < 0:
        raise ValueError("layers must be nonnegative")
    if user_embedding.shape[0] != len(graph.users):
        raise ValueError("user embedding count does not match the graph")
    if item_embedding.shape[0] != len(graph.items):
        raise ValueError("item embedding count does not match the graph")
    user_current = np.asarray(user_embedding, dtype=np.float32)
    item_current = np.asarray(item_embedding, dtype=np.float32)
    user_sum = user_current.copy()
    item_sum = item_current.copy()
    for _ in range(layers):
        user_next = graph.normalized @ item_current
        item_next = graph.normalized.T @ user_current
        user_sum += user_next
        item_sum += item_next
        user_current = user_next
        item_current = item_next
    scale = 1.0 / (layers + 1)
    return user_sum * scale, item_sum * scale


def lightgcn_candidate_scores(
    graph: CompactBipartiteGraph,
    user_embedding: np.ndarray,
    item_embedding: np.ndarray,
    test: pd.DataFrame,
    candidate_columns: list[str],
    batch_size: int = 8192,
    backend: str = "jittor",
    device: str = "auto",
) -> np.ndarray:
    if backend not in {"jittor", "numpy"}:
        raise ValueError("backend must be jittor or numpy")
    query_users = test["src"].to_numpy(np.int64)
    candidates = test[candidate_columns].to_numpy(np.int64)
    user_positions = np.searchsorted(graph.users, query_users)
    clipped_users = np.minimum(user_positions, len(graph.users) - 1)
    valid_users = (user_positions < len(graph.users)) & (
        graph.users[clipped_users] == query_users
    )
    item_positions = np.searchsorted(graph.items, candidates)
    clipped_items = np.minimum(item_positions, len(graph.items) - 1)
    valid_items = (item_positions < len(graph.items)) & (
        graph.items[clipped_items] == candidates
    )
    scores = np.zeros(candidates.shape, dtype=np.float32)
    user_values_jt = None
    item_values_jt = None
    jt = None
    if backend == "jittor":
        import jittor as jt_module

        jt = jt_module
        if device == "cuda":
            if not bool(getattr(jt.compiler, "has_cuda", False)):
                raise RuntimeError("CUDA was requested but this Jittor build has no CUDA")
            jt.flags.use_cuda = 1
        elif device == "cpu":
            jt.flags.use_cuda = 0
        elif device == "auto":
            jt.flags.use_cuda = int(bool(getattr(jt.compiler, "has_cuda", False)))
        else:
            raise ValueError("device must be one of: auto, cpu, cuda")
        user_values_jt = jt.array(np.asarray(user_embedding, dtype=np.float32))
        item_values_jt = jt.array(np.asarray(item_embedding, dtype=np.float32))
    for start in range(0, len(test), batch_size):
        stop = min(start + batch_size, len(test))
        if backend == "jittor":
            user_index = jt.array(clipped_users[start:stop]).int64()
            item_index = jt.array(clipped_items[start:stop]).int64()
            block = (
                user_values_jt[user_index].unsqueeze(1)
                * item_values_jt[item_index]
            ).sum(dim=2).numpy()
        else:
            block = np.einsum(
                "br,bcr->bc",
                user_embedding[clipped_users[start:stop]],
                item_embedding[clipped_items[start:stop]],
                optimize=True,
            )
        block[~valid_items[start:stop]] = 0.0
        block[~valid_users[start:stop]] = 0.0
        scores[start:stop] = block
    if backend == "jittor":
        del user_values_jt, item_values_jt
        jt.clean()
    return scores
