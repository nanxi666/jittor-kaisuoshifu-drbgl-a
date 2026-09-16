# 数据说明（不提交原始数据）

## 放置方式

从赛题页面下载原始数据后放到如下位置（目录已建好，只需放入 CSV）：

```text
data/
|-- dataset1/
|   |-- train.csv
|   `-- test.csv
`-- dataset2/
    |-- train.csv
    `-- test.csv
```

## 字段说明

| 文件 | 字段 | 说明 |
| --- | --- | --- |
| `train.csv` | `src,dst,time` | 历史交互边，`time` 为 Unix 秒 |
| `test.csv` | `src,time,c1,...,c100` | 每个查询的 100 个候选 item 编号 |

`--data-root` 可指向任意目录（相对路径按仓库根目录解析）。

## 校验规则

`drbgl.pipeline.read_dataset` 会检查：

- 两个 CSV 都存在（缺失时报出**期望的完整路径**）
- 列名完整（缺列时列出缺失字段）
- 行数非零
- `train.time.max() < test.time.min()`（否则报时间边界错误）

可用 `python tools/inspect_data.py` 在跑流水线之前做一次体检。

## 注意

- 原始 CSV 体积较大且含赛方数据，**已被 `.gitignore` 忽略**，请勿提交。
- `data/dataset*/README.md` 仅为占位说明。
