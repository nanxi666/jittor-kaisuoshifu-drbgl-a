#!/usr/bin/env python3
"""推理 / 复现入口：生成 dataset1.csv、dataset2.csv 与 submission.zip。

用法（在仓库根目录执行）::

    python scripts/infer.py --config configs/infer.yaml
    python scripts/infer.py --dataset dataset2 --output-dir outputs/ds2
    python scripts/infer.py --config configs/retrain.yaml --device cuda

命令行参数优先级高于配置文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from drbgl.config import (
        deep_update,
        load_config,
        require_files,
        save_command,
        save_config,
        set_seed,
        tee_stdout,
    )
    from drbgl.pipeline import (
        RuntimeConfig,
        package_submission,
        run_dataset1,
        run_dataset2,
    )
except ImportError as error:  # 依赖缺失时给出可修复的提示
    raise SystemExit(
        f"import failed: {error}\n"
        "how to fix: pip install -r requirements.txt "
        "(run from the repository root)"
    ) from error


EMBEDDING_CHECKPOINTS = (
    "lightgcn_d64_s42.npz",
    "lightgcn_d64_s2027.npz",
    "lightgcn_d128_s42.npz",
)
GATE_CHECKPOINTS = ("base_gate.txt", "candidate_gate.txt")


def parse_args() -> argparse.Namespace:
    """解析命令行参数（未显式给出的项回落到配置文件）。"""
    parser = argparse.ArgumentParser(
        description="Reproduce the A-board temporal link-ranking submission"
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "infer.yaml"
    )
    parser.add_argument("--dataset", choices=("dataset1", "dataset2", "both"))
    parser.add_argument("--data-root", type=Path, help="数据根目录")
    parser.add_argument("--output-dir", type=Path, help="产物目录")
    parser.add_argument("--model-dir", type=Path, help="检查点目录")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--score-backend", choices=("numpy", "jittor"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--community-dim", type=int)
    parser.add_argument(
        "--retrain-embeddings",
        dest="retrain_embeddings",
        action="store_true",
        default=None,
        help="用 Jittor 重训 BPR embedding 后再推理",
    )
    parser.add_argument(
        "--rebuild-base-gate",
        dest="rebuild_base_gate",
        action="store_true",
        default=None,
        help="重算 base-gate 冷启动名次，而不是加载 models/base_cold_ranks.npz",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    """相对路径按仓库根目录解析，绝对路径原样返回。"""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def build_runtime(section: dict) -> RuntimeConfig:
    """把配置字典转成 RuntimeConfig。"""
    return RuntimeConfig(
        seed=int(section.get("seed", 20240501)),
        device=str(section.get("device", "auto")),
        score_backend=str(section.get("score_backend", "numpy")),
        embedding_dim=int(section.get("embedding_dim", 64)),
        epochs=int(section.get("epochs", 5)),
        batch_size=int(section.get("batch_size", 65536)),
        learning_rate=float(section.get("learning_rate", 0.02)),
        threads=int(section.get("threads", 16)),
        community_dim=int(section.get("community_dim", 32)),
        retrain_embeddings=bool(section.get("retrain_embeddings", False)),
        rebuild_base_gate=bool(section.get("rebuild_base_gate", False)),
        base_gate_openmp_threads=int(section.get("base_gate_openmp_threads", 191)),
        base_gate_blas_threads=int(section.get("base_gate_blas_threads", 64)),
        base_gate_sklearn_threads=int(section.get("base_gate_sklearn_threads", 8)),
    )


def main() -> None:
    """按配置（命令行覆盖）跑通推理流水线并落盘可复现产物。"""
    args = parse_args()
    config = deep_update(
        load_config(args.config),
        {
            "dataset": args.dataset,
            "data_root": args.data_root,
            "output_dir": args.output_dir,
            "model_dir": args.model_dir,
            "runtime": {
                "seed": args.seed,
                "device": args.device,
                "score_backend": args.score_backend,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "threads": args.threads,
                "community_dim": args.community_dim,
                "retrain_embeddings": args.retrain_embeddings,
                "rebuild_base_gate": args.rebuild_base_gate,
            },
        },
    )

    runtime = build_runtime(config.get("runtime", {}))
    data_root = resolve_path(config.get("data_root", "data"))
    output_dir = resolve_path(config.get("output_dir", "outputs"))
    model_dir = resolve_path(config.get("model_dir", "models"))
    dataset = str(config.get("dataset", "both"))

    set_seed(runtime.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    save_command(output_dir / "command.txt")

    if dataset in {"dataset2", "both"} and not runtime.retrain_embeddings:
        require_files(
            [model_dir / name for name in EMBEDDING_CHECKPOINTS],
            hint=f"check --model-dir (now: {model_dir}) or add --retrain-embeddings",
        )
    if dataset in {"dataset2", "both"}:
        gates = [model_dir / name for name in GATE_CHECKPOINTS]
        if not runtime.rebuild_base_gate:
            gates.append(model_dir / "base_cold_ranks.npz")
        require_files(
            gates, hint=f"check --model-dir (now: {model_dir})"
        )

    with tee_stdout(output_dir / "infer.log"):
        print(f"repo root:   {REPO_ROOT}")
        print(f"data root:   {data_root}")
        print(f"model dir:   {model_dir}")
        print(f"output dir:  {output_dir}")
        print(f"runtime:     {runtime}")
        if dataset in {"dataset1", "both"}:
            path = run_dataset1(data_root, output_dir)
            print(f"dataset1 result: {path}", flush=True)
        if dataset in {"dataset2", "both"}:
            path = run_dataset2(data_root, output_dir, model_dir, runtime)
            print(f"dataset2 result: {path}", flush=True)
        if dataset == "both":
            print(f"submission: {package_submission(output_dir)}", flush=True)


if __name__ == "__main__":
    main()
