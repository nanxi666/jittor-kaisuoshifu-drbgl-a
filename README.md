# jittor-kaisuoshifu-drbgl-a

> 计图比赛（Jittor）赛道：**DRBGL 时序图链接预测 A 榜**复现代码，团队「开锁师傅 / locksmith」。

线上总分 **`1.5186`**。本项目针对两个数据集的结构差异采用**自适应双分支排序**，
嵌入部分用 **Jittor** 训练 BPR-LightGCN，排序部分用 LightGBM LambdaRank 门控。

> 仓库命名遵循规范 `jittor-开锁师傅-[项目名字]`。团队名在 GitHub 仓库 slug 中需使用
> ASCII，故渲染为 `kaisuoshifu` / `locksmith`；项目名取 `drbgl-a`。
> 本地文件夹名不影响远程仓库名，如需其它标识可在创建 GitHub 仓库时自行命名。

## 目录结构

```text
.
├── README.md                # 本文件
├── LICENSE                  # MIT 许可证
├── NOTICE                   # 第三方代码/数据声明
├── .gitignore
├── requirements.txt         # 依赖清单
├── configs/                 # 配置（命令行优先级更高）
│   ├── infer.yaml           # 默认推理 / 复现配置
│   ├── retrain.yaml         # 重训 embedding 后推理
│   └── train_gates.yaml     # 重训两个 LambdaRank 门控
├── src/drbgl/               # 核心代码
│   ├── config.py            # 配置读写、随机种子、日志、运行命令
│   ├── pipeline.py          # dataset1 / dataset2 推理流水线
│   ├── data.py              # 数据加载与严格时间切分
│   ├── candidates.py        # 候选构造（源-时间模板）
│   ├── metrics.py           # 排序指标与序数名次映射
│   ├── artifacts.py         # sha256 / JSON / git 元数据工具
│   ├── experts/             # 打分配专家（LightGCN、低保秩、结构、曝光…）
│   └── baselines/           # 启发式基线与 depop 家族（v3 / v7 / v9）
├── scripts/                 # 运行脚本
│   ├── infer.py             # 推理 / 复现（生成 CSV 与 submission.zip）
│   ├── train.py             # 重训门控
│   └── verify_submission.py # 校验提交包
├── tools/
│   └── inspect_data.py      # 数据统计与体检
├── tests/                   # 单元测试（不依赖原始数据）
├── data/                    # 仅放数据说明（原始数据不入库）
├── models/                  # 随仓库发布的检查点（用于严格复现）
└── outputs/                 # 日志/分数/CSV/zip（默认不提交）
```

## 1. 环境安装

- Python 3.9 ~ 3.11（本项目在 3.11 上验证）
- 安装依赖（Windows / Linux 通用）：

```bash
pip install -r requirements.txt
```

依赖包含 `jittor`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`lightgbm`、
`threadpoolctl`、`PyYAML`。Jittor 的安装与 CUDA 配置参考
[Jittor 官方文档](https://github.com/jittor/jittor#install)。

## 2. 数据准备

把赛方原始 CSV 放到：

```text
data/
|-- dataset1/
|   |-- train.csv
|   `-- test.csv
`-- dataset2/
    |-- train.csv
    `-- test.csv
```

- `train.csv`：`src,dst,time`（历史边）
- `test.csv`：`src,time,c1,...,c100`（每个查询的 100 个候选）

数据根目录通过 `--data-root` 或配置文件字段 `data_root` 指定
（相对路径按仓库根目录解析）。原始数据已被 `.gitignore` 忽略，请勿提交。

跑流水线前建议先体检：

```bash
python tools/inspect_data.py
```

## 3. 训练

**推理默认加载随包检查点**，通常不需要训练；下面两条命令用于从原始数据重新训练。

重训两个 LambdaRank 门控（在 dataset2 的时间验证折上）：

```bash
python scripts/train.py --config configs/train_gates.yaml
```

用 Jittor 重训 BPR-LightGCN 嵌入并推理：

```bash
python scripts/infer.py --config configs/retrain.yaml --device cuda
```

常用覆盖参数（命令行优先级高于配置文件）：

```bash
python scripts/train.py --device cpu --uniform-final 20000 --threads 8
```

> 完整训练的计算量与内存开销较大，建议 ≥128 GB 内存；
> 调小 `sampling.uniform_final / strict_final` 可显著降低开销（指标会有波动）。

## 4. 评测 / 推理

一条命令复现 A 榜两份 CSV 并打包：

```bash
python scripts/infer.py --config configs/infer.yaml
```

只跑单个数据集 / 指定检查点目录：

```bash
python scripts/infer.py --dataset dataset1 --output-dir outputs/ds1
python scripts/infer.py --dataset dataset2 --model-dir models --output-dir outputs/ds2
```

校验产物（成员、形状、取值范围、sha256）：

```bash
python scripts/verify_submission.py outputs/submission.zip
python scripts/verify_submission.py outputs/submission.zip --require-reference-hash
```

单元测试：

```bash
python -m unittest discover -s tests -v
```

每次运行会在 `output_dir` 下写出：

| 文件 | 内容 |
| --- | --- |
| `config.yaml` | 本次实际使用的配置（含命令行覆盖后的最终值） |
| `command.txt` | 本次运行的完整命令 |
| `train.log` / `infer.log` | 日志（含 stdout 与异常回溯） |
| `dataset1.csv` / `dataset2.csv` | 提交用结果，每行 100 列序数名次 |
| `*_manifest.json` | 输入/权重 sha256、行数、耗时等可复现信息 |
| `submission.zip` | 可直接提交的压缩包 |

## 5. 结果说明

- **线上指标**：A 榜总分 `1.5186`（赛方在隐藏测试标签上计算）。
- **本地可用指标**：
  - `MRR`（`drbgl.metrics.mean_reciprocal_rank`）：启发式基线调参与验证集分析使用，
    对每个查询计算正例的平均倒数排名；
  - `NDCG`（label_gain `[0, 1]`）：两个 LambdaRank 门控的训练目标。
- **提交形式**：`dataset1.csv` 61051 行、`dataset2.csv` 153420 行，
  每行 100 列，取值 `0.01~1.00`（候选序数名次除以 100，**非概率**）。
- **验收方式**：默认路径走 NumPy/SciPy 确定性数值链路，输出 CSV 与 A 榜提交
  **逐字节一致**，可用 `--require-reference-hash` 校验
  （参考 sha256 见 `scripts/verify_submission.py` 的 `REFERENCE_MEMBERS`）。
- **与线上成绩的差异说明**：
  - `--score-backend jittor` 会改用 Jittor 计算候选点积，浮点累加顺序不同，
    CSV 可能与参考哈希不一致，但排序质量基本一致；
  - `--retrain-embeddings` / `--rebuild-base-gate` 会重算嵌入与冷启动名次，
    受 Jittor 版本、CUDA 与否、线程数影响，会有合理波动；
  - base-gate 阶段固定了原生线程池布局（`runtime.base_gate_*_threads`），
    换机器复算冷启动名次时结果可能略有差异；
  - 测试集标签不在原始数据中，本地只能在与训练集同源的时间验证折上评测，
    绝对数值与线上总分不可直接比较。

## 6. 可复现说明

- 随机种子：`--seed`（或 `runtime.seed`，默认 `20240501`）统一设置
  Python / NumPy / Jittor 随机种子（见 `src/drbgl/config.py` 的 `set_seed`）；
  负采样、聚类、SVD 等环节也使用各自固定 `random_state`。
- 每次运行落盘：实际配置 `config.yaml`、运行命令 `command.txt`、日志 `*.log`。
- 缺失数据、缺失权重、数据列不合法、时间边界错误都会抛出**带修复建议**的异常
  （例如提示用哪个参数、去哪里准备数据）。
- 入口脚本不写死本机路径：所有路径来自 `--config` 或命令行参数，
  相对路径统一按仓库根目录解析。

## 7. 进阶：基线脚本

启发式基线与 depop 家族可作为对照实验（`PYTHONPATH` 需包含 `src`）：

```bash
# Linux / macOS
PYTHONPATH=src python -m drbgl.baselines.depop_v9 --dataset dataset1
# Windows (PowerShell)
$env:PYTHONPATH="src"; python -m drbgl.baselines.depop_v9 --dataset dataset1
```

它们同样支持 `--data-root` / `--output-dir`。

## 8. 第三方声明

本项目使用 Jittor、LightGBM、scikit-learn、SciPy、NumPy、pandas、PyYAML，
LightGCN 传播由本仓库自行实现，来源与许可证见 [`NOTICE`](NOTICE)。
随仓库发布的检查点来源见 [`models/README.md`](models/README.md)。

## 9. 提交规范

提交信息建议使用前缀：`feat:` / `fix:` / `docs:` / `refactor:` / `chore:`。
提交前请确保：代码可运行（`python -m unittest discover -s tests`）、
README 命令真实可用、未提交原始数据与大文件中间产物。
