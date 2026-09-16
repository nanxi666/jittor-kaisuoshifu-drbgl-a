# 随仓库发布的检查点

`models/` 下的文件由本仓库自身训练得到（`scripts/train.py`、
`scripts/infer.py --retrain-embeddings`），**不含测试集标签**。
保留它们的目的是让 A 榜结果可被逐字节复现；全部文件合计约 86 MB，
单文件最大约 29 MB（未超过 GitHub 单文件 50 MB 提示阈值）。

| 文件 | 大小 | 作用 |
| --- | --- | --- |
| `base_cold_ranks.npz` | ~12 MB | dataset2 base-gate 在冷启动槽位上的名次（含候选集指纹） |
| `base_gate.txt` | ~2.3 MB | 108 维特征 LambdaRank base 门控 |
| `candidate_gate.txt` | ~12 MB | 293 维特征 LambdaRank candidate 门控 |
| `lightgcn_d64_s42.npz` | ~14 MB | Jittor 训练的 BPR-LightGCN（dim=64, seed=42） |
| `lightgcn_d64_s2027.npz` | ~14 MB | Jittor 训练的 BPR-LightGCN（dim=64, seed=2027） |
| `lightgcn_d128_s42.npz` | ~29 MB | Jittor 训练的 BPR-LightGCN（dim=128, seed=42） |
| `manifest.json` | - | 上述文件的 sha256 与来源说明 |

## 校验

```bash
python -m unittest discover -s tests -v     # 含 sha256 校验
```

## 重新生成

全部资产都可以从原始数据重算（结果会与随包检查点存在合理差异）：

```bash
python scripts/train.py --config configs/train_gates.yaml            # 门控
python scripts/infer.py --config configs/retrain.yaml --device cuda  # embedding
```

再用 `--model-dir <output_dir>/models` 指向新生成的检查点即可。

## 说明

`.gitignore` 忽略了仓库内的 `*.npz`，但对本目录显式放行
（`!models/*.npz`），请勿误删这些例外规则。
