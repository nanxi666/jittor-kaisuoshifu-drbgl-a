from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateQueries:
    candidates: np.ndarray
    positive_col: np.ndarray
    template_row: np.ndarray
    label_row: np.ndarray | None = None


def _inject_positives(
    templates: np.ndarray,
    positives: np.ndarray,
    positive_col: np.ndarray,
) -> np.ndarray:
    candidates = templates.copy()
    for row, (positive, col) in enumerate(zip(positives, positive_col)):
        existing = np.flatnonzero(candidates[row] == positive)
        if len(existing) and existing[0] != col:
            candidates[row, existing[0]] = candidates[row, col]
        candidates[row, col] = positive
    if np.any(candidates[np.arange(len(candidates)), positive_col] != positives):
        raise AssertionError("positive candidate injection failed")
    return candidates


def build_from_test_templates(
    test_candidates: np.ndarray,
    positives: np.ndarray,
    seed: int,
) -> CandidateQueries:
    """Reuse complete test candidate rows and inject one known positive per query."""
    templates = np.asarray(test_candidates, dtype=np.int64)
    positives = np.asarray(positives, dtype=np.int64)
    if templates.ndim != 2 or templates.shape[1] != 100:
        raise ValueError("test candidate templates must have 100 columns")
    rng = np.random.default_rng(seed)
    template_row = rng.integers(0, len(templates), size=len(positives), dtype=np.int64)
    selected = templates[template_row]
    positive_col = rng.integers(0, 100, size=len(positives), dtype=np.int64)
    candidates = _inject_positives(selected, positives, positive_col)
    if any(len(np.unique(row)) != 100 for row in candidates):
        raise ValueError("a test candidate template contains duplicate candidates")
    return CandidateQueries(candidates, positive_col, template_row)


def build_from_source_time_templates(
    test_src: np.ndarray,
    test_time: np.ndarray,
    test_candidates: np.ndarray,
    label_src: np.ndarray,
    label_time: np.ndarray,
    positives: np.ndarray,
    seed: int,
) -> CandidateQueries:
    """Match test rows and labels one-to-one by source and temporal quantile.

    Complete test rows retain the production source frequency and candidate
    co-exposure structure. Within a source, evenly spaced ordered templates
    are paired with evenly spaced ordered labels. A template and label can
    therefore appear at most once, preventing repeated truth injection from
    leaking the label through transductive exposure counts. Templates with
    duplicate candidates or without a same-source label are omitted.
    """
    template_src = np.asarray(test_src, dtype=np.int64)
    template_time = np.asarray(test_time, dtype=np.int64)
    templates = np.asarray(test_candidates, dtype=np.int64)
    source = np.asarray(label_src, dtype=np.int64)
    event_time = np.asarray(label_time, dtype=np.int64)
    positives = np.asarray(positives, dtype=np.int64)
    if templates.ndim != 2 or templates.shape[1] != 100:
        raise ValueError("test candidate templates must have 100 columns")
    if (
        len(template_src) != len(templates)
        or len(template_time) != len(templates)
    ):
        raise ValueError("test template arrays differ in length")
    if len(source) != len(event_time) or len(source) != len(positives):
        raise ValueError("label arrays differ in length")
    if len(templates) == 0 or len(source) == 0:
        raise ValueError("templates and labels must be non-empty")

    unique_template = np.empty(len(templates), dtype=bool)
    for start in range(0, len(templates), 8192):
        stop = min(start + 8192, len(templates))
        ordered = np.sort(templates[start:stop], axis=1)
        unique_template[start:stop] = np.all(
            ordered[:, 1:] != ordered[:, :-1], axis=1
        )
    valid_template = np.flatnonzero(unique_template)
    if len(valid_template) == 0:
        raise ValueError("no test candidate template has 100 unique candidates")
    template_order = valid_template[
        np.lexsort(
            (template_time[valid_template], template_src[valid_template])
        )
    ]
    ordered_template_source = template_src[template_order]
    template_sources, template_starts, template_counts = np.unique(
        ordered_template_source, return_index=True, return_counts=True
    )
    label_order = np.lexsort((event_time, source))
    ordered_source = source[label_order]
    label_sources, label_starts, label_counts = np.unique(
        ordered_source, return_index=True, return_counts=True
    )

    matched_template: list[np.ndarray] = []
    matched_label: list[np.ndarray] = []
    template_group = np.searchsorted(template_sources, label_sources)
    clipped_group = np.minimum(template_group, len(template_sources) - 1)
    matched_source = (template_group < len(template_sources)) & (
        template_sources[clipped_group] == label_sources
    )
    for label_group in np.flatnonzero(matched_source):
        current_template_group = template_group[label_group]
        template_start = template_starts[current_template_group]
        template_stop = template_start + template_counts[current_template_group]
        template_rows = template_order[template_start:template_stop]
        label_start = label_starts[label_group]
        label_stop = label_start + label_counts[label_group]
        label_rows = label_order[label_start:label_stop]
        match_count = min(len(template_rows), len(label_rows))
        template_position = (
            (np.arange(match_count) + 0.5)
            * len(template_rows)
            / match_count
        ).astype(np.int64)
        label_position = (
            (np.arange(match_count) + 0.5)
            * len(label_rows)
            / match_count
        ).astype(np.int64)
        matched_template.append(template_rows[template_position])
        matched_label.append(label_rows[label_position])

    if not matched_template:
        raise ValueError("no test source has a label in this interval")
    template_row = np.concatenate(matched_template)
    label_row = np.concatenate(matched_label)
    order = np.argsort(template_row, kind="stable")
    template_row = template_row[order]
    label_row = label_row[order]
    rng = np.random.default_rng(seed)
    positive_col = rng.integers(0, 100, size=len(template_row), dtype=np.int64)
    candidates = _inject_positives(
        templates[template_row], positives[label_row], positive_col
    )
    if any(len(np.unique(row)) != 100 for row in candidates):
        raise AssertionError("positive injection introduced duplicate candidates")
    if np.any(template_src[template_row] != source[label_row]):
        raise AssertionError("template and label sources differ")
    if len(np.unique(label_row)) != len(label_row):
        raise AssertionError("one label was injected more than once")
    return CandidateQueries(candidates, positive_col, template_row, label_row)


def build_from_candidate_pool(
    test_candidates: np.ndarray,
    positives: np.ndarray,
    seed: int,
) -> CandidateQueries:
    """Match OpenJittor: one truth plus 99 uniform draws from the test pool."""
    templates = np.asarray(test_candidates, dtype=np.int64)
    positives = np.asarray(positives, dtype=np.int64)
    if templates.ndim != 2 or templates.shape[1] != 100:
        raise ValueError("test candidate matrix must have 100 columns")
    pool = np.unique(templates)
    if len(pool) < 2:
        raise ValueError("candidate pool must contain at least two ids")

    rng = np.random.default_rng(seed)
    negatives = pool[rng.integers(0, len(pool), size=(len(positives), 99))]
    collision = negatives == positives[:, None]
    while collision.any():
        rows, cols = np.where(collision)
        negatives[rows, cols] = pool[
            rng.integers(0, len(pool), size=len(rows))
        ]
        collision = negatives == positives[:, None]

    candidates = np.column_stack([positives, negatives])
    positive_col = np.zeros(len(positives), dtype=np.int64)
    template_row = np.full(len(positives), -1, dtype=np.int64)
    return CandidateQueries(candidates, positive_col, template_row)
