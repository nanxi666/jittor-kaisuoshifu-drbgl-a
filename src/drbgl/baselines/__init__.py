"""启发式基线与 depop 家族。

- ``heuristic_core`` / ``heuristic_ds1`` / ``heuristic_ds2``：早期启发式主干。
- ``heuristic_submission``：含策略自动探测（memory / depop）的提交生成器。
- ``depop_v3`` / ``depop_v7`` / ``depop_v9``：逐步增强的 depop 排序器，
  ``depop_v9`` 被主线流水线用作 ``dataset1`` 的最终打分器。
"""

__all__ = [
    "heuristic_core",
    "heuristic_ds1",
    "heuristic_ds2",
    "heuristic_submission",
    "depop_v3",
    "depop_v7",
    "depop_v9",
]
