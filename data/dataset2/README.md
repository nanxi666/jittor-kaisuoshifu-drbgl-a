# dataset2

把赛方提供的 `train.csv` 与 `test.csv` 放在本目录（字段见 [../README.md](../README.md)）。

本数据集为 depop 型（pair 重复率近似为 0），流水线走 293 维特征 LightGBM
门控分支，需要 `models/` 下的检查点（或自行重训）。
