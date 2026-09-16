"""流水线与可复现基础设施的单元测试（不依赖原始数据）。"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drbgl.config import (  # noqa: E402
    deep_update,
    load_config,
    require_files,
    save_command,
    save_config,
    set_seed,
    tee_stdout,
)
from drbgl.pipeline import RuntimeConfig, read_dataset  # noqa: E402

MODEL_DIGESTS = {
    "base_cold_ranks.npz": (
        "c2544f96f113b940943543350b2aed5653fc1328a4249ac70221f57f6bae615f"
    ),
    "base_gate.txt": (
        "569a6a1673147c6ad7bcec5537df77172550ed163bcaa03dd3cef7b316529628"
    ),
    "candidate_gate.txt": (
        "7aacda64dfc6a1935e06bebf5251cd4949ca1ab8c3cc0ba6ff2a8c39654cdf32"
    ),
    "lightgcn_d64_s42.npz": (
        "d7fb2dadaaad051fcf85f70c4b88e6344c681055014afa1d19399321cacd2a57"
    ),
    "lightgcn_d64_s2027.npz": (
        "4e22a27fe71cead355e9258015a4492aa014b15dddfa43efb479cbb086442012"
    ),
    "lightgcn_d128_s42.npz": (
        "97f95e127ac2a08a8be0ecd061311b5b233b59c6214548f6e671b7c001aa7a02"
    ),
}


class CheckpointTests(unittest.TestCase):
    def test_gate_assets_match_reference_models(self) -> None:
        for name, digest in MODEL_DIGESTS.items():
            path = REPO_ROOT / "models" / name
            if not path.is_file():  # 允许只拉代码不拉权重的场景
                self.skipTest(f"checkpoint not present: {path}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, digest, msg=name)

    def test_embedding_training_uses_jittor(self) -> None:
        trainer = SRC_ROOT / "drbgl" / "experts" / "lightgcn.py"
        self.assertIn("import jittor as jt", trainer.read_text(encoding="utf-8"))


class ReproducibilityTests(unittest.TestCase):
    def test_missing_data_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "expected"):
                read_dataset(Path(directory), "dataset1")

    def test_missing_files_hint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "nope.txt"
            with self.assertRaisesRegex(FileNotFoundError, "how to fix"):
                require_files([missing], hint="do something")

    def test_config_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "config file"):
                load_config(Path(directory) / "missing.yaml")

    def test_deep_update_ignores_none(self) -> None:
        base = {"runtime": {"threads": 16, "seed": 1}, "dataset": "both"}
        merged = deep_update(base, {"dataset": None, "runtime": {"threads": 8}})
        self.assertEqual(merged["dataset"], "both")
        self.assertEqual(merged["runtime"]["threads"], 8)
        self.assertEqual(merged["runtime"]["seed"], 1)

    def test_config_command_and_log_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"dataset": "both", "runtime": {"seed": 7}}
            save_config(config, root / "config.yaml")
            save_command(root / "command.txt", ["python", "scripts/infer.py"])
            self.assertEqual(load_config(root / "config.yaml"), config)
            self.assertEqual(
                (root / "command.txt").read_text(encoding="utf-8").strip(),
                "python scripts/infer.py",
            )
            with tee_stdout(root / "train.log") as log_path:
                print("hello")
            self.assertIn("hello", log_path.read_text(encoding="utf-8"))

    def test_set_seed_is_deterministic(self) -> None:
        set_seed(20240501)
        first = np.random.default_rng(0).random(3)
        set_seed(20240501)
        second = np.random.default_rng(0).random(3)
        np.testing.assert_allclose(first, second)


class RuntimeConfigTests(unittest.TestCase):
    def test_defaults_match_reference_pipeline(self) -> None:
        config = RuntimeConfig()
        self.assertEqual(config.score_backend, "numpy")
        self.assertEqual(config.epochs, 5)
        self.assertEqual(config.batch_size, 65536)
        self.assertAlmostEqual(config.learning_rate, 0.02)
        self.assertEqual(config.community_dim, 32)


if __name__ == "__main__":
    unittest.main()
