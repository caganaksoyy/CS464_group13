"""Session helpers called from the setup cell of every notebook.

The part of the setup that must happen *before* this package is importable (mounting
Drive, cloning the repo, pip install) lives in the notebook cell itself. Everything
after that lives here, so all notebooks behave identically.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from pianist_id import paths
from pianist_id.utils.config import is_tiny
from pianist_id.utils.logging import environment_info, git_commit


def in_colab() -> bool:
    return "google.colab" in sys.modules


def copy_dir_if_missing(src: Path, dst: Path) -> bool:
    """Copy directory ``src`` to ``dst`` unless ``dst`` already exists.

    Used to copy the parsed cache from Drive to the local SSD once per session.
    A ``.complete`` marker file is written last, so a copy interrupted half-way is
    detected and redone next time.

    Returns:
        True if a copy was made, False if it was skipped.
    """
    if src.resolve() == dst.resolve():
        return False  # local mode: persistent and scratch roots are the same folder
    marker = dst / ".complete"
    if marker.exists():
        return False
    if not src.exists():
        return False
    if dst.exists():
        shutil.rmtree(dst)  # incomplete earlier copy
    shutil.copytree(src, dst)
    marker.touch()
    return True


def print_environment() -> None:
    info = environment_info()
    gpu = info["gpu"]
    print(f"Python {info['python']} on {info['platform']}")
    if gpu.get("cuda_available"):
        print(f"GPU: {gpu['name']} ({gpu['memory_gb']} GB), bf16={gpu['bf16_supported']}, "
              f"CUDA {gpu['cuda_version']}")
    else:
        print("GPU: none (CPU only)")
    installed = {k: v for k, v in info["packages"].items() if v is not None}
    print("Packages:", ", ".join(f"{k}=={v}" for k, v in installed.items()))
    print("Git commit:", git_commit())


def prepare(copy_cache: bool = True) -> None:
    """Create storage folders, copy the cache to local disk, print the environment."""
    for d in (paths.root(), paths.raw_dir(), paths.results_dir(), paths.local_root()):
        paths.ensure_dir(d)
    print(f"PIANIST_ID_ROOT  (persistent): {paths.root()}")
    print(f"PIANIST_ID_LOCAL (scratch)   : {paths.local_root()}")
    print(f"Tiny smoke-test mode: {is_tiny()}")
    if copy_cache and paths.cache_dir().resolve() == paths.local_cache_dir().resolve():
        print("Cache is read in place (persistent and scratch roots are the same).")
    elif copy_cache:
        if copy_dir_if_missing(paths.cache_dir(), paths.local_cache_dir()):
            print(f"Copied cache {paths.cache_dir()} -> {paths.local_cache_dir()}")
        elif (paths.local_cache_dir() / ".complete").exists():
            print(f"Local cache already present: {paths.local_cache_dir()}")
        else:
            print("No cache on Drive yet (built in notebook 02).")
    print_environment()
