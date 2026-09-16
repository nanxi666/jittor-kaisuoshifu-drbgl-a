#!/usr/bin/env python3
"""校验提交压缩包：成员、行数、取值范围，并可与 A 榜参考 sha256 逐字节比对。

用法（在仓库根目录执行）::

    python scripts/verify_submission.py outputs/submission.zip
    python scripts/verify_submission.py outputs/submission.zip --require-reference-hash
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

try:
    import numpy as np
except ImportError as error:  # 依赖缺失时给出可修复的提示
    raise SystemExit(
        f"import failed: {error}\n"
        "how to fix: pip install -r requirements.txt"
    ) from error


EXPECTED_ROWS = {"dataset1.csv": 61051, "dataset2.csv": 153420}
REFERENCE_MEMBERS = {
    "dataset1.csv": (
        "455716161d07b7ff40c0207a8d40e65bd576ebdd0fa33616875690a24994109e"
    ),
    "dataset2.csv": (
        "6529919322140e2a4365804d281d1a085e233fbd735ee73328687e6d26d4bd6a"
    ),
}


def check_archive(archive: Path, require_reference_hash: bool = False) -> bool:
    """校验一个 submission.zip，返回是否与 A 榜参考完全一致。

    Args:
        archive: zip 路径。
        require_reference_hash: 为 True 时，任一 CSV 与参考哈希不同即抛错。

    Returns:
        所有成员都与 A 榜参考 sha256 一致时为 True。

    Raises:
        ValueError: 成员名/形状/取值不合法。
    """
    identical = True
    with zipfile.ZipFile(archive) as zip_file:
        members = set(zip_file.namelist())
        if members != set(EXPECTED_ROWS):
            raise ValueError(
                f"archive members differ from {sorted(EXPECTED_ROWS)}: "
                f"{sorted(members)}"
            )
        for name, rows in EXPECTED_ROWS.items():
            content = zip_file.read(name)
            digest = hashlib.sha256(content).hexdigest()
            values = np.loadtxt(
                io.BytesIO(content), delimiter=",", dtype=np.float32
            )
            if values.shape != (rows, 100):
                raise ValueError(f"{name} shape {values.shape} != {(rows, 100)}")
            if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
                raise ValueError(f"{name} contains invalid values")
            matches = digest == REFERENCE_MEMBERS[name]
            identical = identical and matches
            print(
                f"{name}: shape={values.shape}, range=[{values.min():.2f}, "
                f"{values.max():.2f}], sha256={digest}, reference={matches}"
            )
    if require_reference_hash and not identical:
        raise SystemExit(
            "submission is valid but not byte-identical to the A-board result; "
            "see README '结果说明' for the expected reasons"
        )
    return identical


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="Validate a submission archive")
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--require-reference-hash",
        action="store_true",
        help="fail unless each CSV is byte-identical to the A-board member",
    )
    args = parser.parse_args()
    archive = Path(args.archive)
    if not archive.is_file():
        raise FileNotFoundError(
            f"submission archive not found: {archive}\n"
            "how to fix: python scripts/infer.py --dataset both"
        )
    check_archive(archive, args.require_reference_hash)


if __name__ == "__main__":
    main()
