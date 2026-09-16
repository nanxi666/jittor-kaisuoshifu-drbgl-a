"""DRBGL A 榜的自适应双分支排序流水线。

模块一览::

    drbgl/
    |-- artifacts.py     落盘产物工具：sha256、JSON、git 元数据
    |-- candidates.py    候选集构造（源-时间模板采样）
    |-- config.py        配置读写、命令行覆盖、随机种子、日志与运行命令
    |-- data.py          原始数据加载与时间序列切分
    |-- metrics.py       排序指标与分数到序数的映射
    |-- pipeline.py      dataset1 / dataset2 的推理流水线
    |-- experts/         各类打分配专家（低保秩、LightGCN、结构/曝光特征等）
    `-- baselines/       启发式基线与 depop 家族（v3 / v7 / v9）

除 Jittor 相关的 embedding 训练外，数值路径默认使用 NumPy / SciPy，
以保证多次运行结果稳定一致。
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
