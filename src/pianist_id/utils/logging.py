"""Run bookkeeping: environment info, run metadata, CSV training history.

Every run directory ``results/<experiment>/<run_name>/`` gets:

* ``config.yaml``   -- the fully resolved config that was used
* ``run_info.json`` -- seed, git commit, timestamp, Python/package versions, GPU
* ``history.csv``   -- one row per epoch (written by :class:`CSVLogger`)
* ``metrics.json``  -- final metrics (written by the evaluation code)
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from pianist_id import paths
from pianist_id.utils.config import save_yaml

# Packages whose versions are recorded with every run and printed by the setup cell.
TRACKED_PACKAGES = [
    "numpy", "pandas", "scikit-learn", "torch", "pretty_midi", "symusic",
    "xgboost", "lightgbm", "pyarrow", "matplotlib", "pyyaml", "shap",
]


def package_versions(names: list[str] | None = None) -> dict[str, str | None]:
    """Installed version of each package, or None if it is not installed."""
    versions: dict[str, str | None] = {}
    for name in names or TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_info() -> dict[str, Any]:
    """GPU name and memory via torch, if torch and a GPU are available."""
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "note": "torch not installed"}
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "name": props.name,
        "memory_gb": round(props.total_memory / 1024**3, 1),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "cuda_version": torch.version.cuda,
    }


def git_commit(repo: Path | None = None) -> str | None:
    """Short git hash of HEAD (with '-dirty' if there are uncommitted changes)."""
    repo = repo or paths.repo_root()
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu_info(),
    }


def write_json(values: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2, ensure_ascii=False, default=str)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_run_info(
    run_dir: str | Path,
    config: dict[str, Any],
    seed: int | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``config.yaml`` and ``run_info.json`` into ``run_dir`` (created if needed)."""
    run_dir = paths.ensure_dir(Path(run_dir))
    save_yaml(config, run_dir / "config.yaml")
    info = {
        "seed": seed,
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **environment_info(),
        **(extra or {}),
    }
    write_json(info, run_dir / "run_info.json")
    return run_dir


class CSVLogger:
    """Append-only CSV log, one dict per row. Safe to reopen after a resume.

    The column order is fixed by the first row ever written; later rows must use
    the same keys.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fieldnames: list[str] | None = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="", encoding="utf-8") as f:
                self.fieldnames = next(csv.reader(f))

    def log(self, row: dict[str, Any]) -> None:
        new_file = self.fieldnames is None
        if new_file:
            self.fieldnames = list(row)
        elif set(row) != set(self.fieldnames):
            raise ValueError(f"Row keys {sorted(row)} != CSV columns {self.fieldnames}")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
