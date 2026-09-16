"""打分配专家：每个模块产出一个或多个 [Q, 100] 的候选打分矩阵。

约定：所有专家输出均为 ``numpy.ndarray``，形状 ``[query 数, 候选数=100]``
（float32/float64 均可，最终统一转 float32），数值越大表示越可能为正例。
"""

__all__ = [
    "extra_expert_features",
    "lightgcn",
    "listwise_gate",
    "low_rank",
    "multiscale_exposure",
    "sequence_graph",
    "structural_features",
]
