from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, check=False, capture_output=True, text=True
        )
        return result.stdout.strip()

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_score(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record["recorded_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def inventory(root: Path) -> dict[str, Any]:
    generated_dirs = sorted(root.glob("kernel_meta*"))
    failed = []
    for path in (root / "output").rglob("*"):
        if path.is_file() and path.stat().st_size < 1024:
            failed.append({"path": str(path.relative_to(root)), "size": path.stat().st_size})
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated_kernel_dirs": [str(path.relative_to(root)) for path in generated_dirs],
        "small_output_files": failed,
        "policy": "inventory only; no files were deleted",
    }
