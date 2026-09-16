"""排序指标与序数名次映射的单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drbgl.metrics import (  # noqa: E402
    blend_rank_scores,
    mean_reciprocal_rank,
    ordinal_rank_scores,
    positive_ranks,
)


class RankMetricTests(unittest.TestCase):
    def test_ordinal_ranks_are_a_permutation(self) -> None:
        raw = np.array([[1.0, 1.0, -2.0] + list(range(97))], dtype=np.float32)
        candidates = np.arange(100, dtype=np.int64)[None, :]
        ranks = ordinal_rank_scores(raw, candidates)
        self.assertEqual(ranks.shape, (1, 100))
        np.testing.assert_allclose(np.sort(ranks[0]), np.arange(1, 101) / 100)

    def test_ordinal_ranks_reject_bad_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected"):
            ordinal_rank_scores(np.zeros((4, 50), dtype=np.float32))

    def test_positive_ranks_average_ties(self) -> None:
        scores = np.array([[3.0, 3.0, 1.0]], dtype=np.float64)
        ranks = positive_ranks(scores, np.array([0], dtype=np.int64))
        np.testing.assert_allclose(ranks, [1.5])

    def test_mean_reciprocal_rank(self) -> None:
        scores = np.array([[3.0, 1.0], [1.0, 3.0]], dtype=np.float64)
        mrr = mean_reciprocal_rank(scores, np.array([0, 1], dtype=np.int64))
        self.assertAlmostEqual(mrr, 1.0)

    def test_blend_requires_valid_weight(self) -> None:
        scores = np.zeros((1, 100), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "challenger_weight"):
            blend_rank_scores(scores, scores, 1.5)


if __name__ == "__main__":
    unittest.main()
