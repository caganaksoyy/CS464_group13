"""Central place for every file-system location used by the project.

Two roots are used:

* ``PIANIST_ID_ROOT`` -- *persistent* storage. On Colab this is the project folder on
  Google Drive (``/content/drive/MyDrive/pianist-id``). It holds the raw ATEPP zip,
  the parsed cache, split CSVs, features and all results/checkpoints.
* ``PIANIST_ID_LOCAL`` -- *fast scratch* storage. On Colab this is the VM's local SSD
  (``/content/local``). It holds the extracted ATEPP archive and a copy of the cache,
  because reading thousands of small files from Drive is very slow. It is wiped when
  the Colab session ends.

When the environment variables are not set (local development), both default to
``<repo>/workspace`` so nothing touches the repository's tracked files.

Every function returns a ``Path`` and never creates directories, except
:func:`ensure_dir`.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_ROOT = "PIANIST_ID_ROOT"
ENV_LOCAL = "PIANIST_ID_LOCAL"


def repo_root() -> Path:
    """The git repository root (``src/pianist_id/paths.py`` -> three levels up)."""
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return repo_root() / "configs"


def root() -> Path:
    """Persistent storage root (Drive on Colab)."""
    value = os.environ.get(ENV_ROOT)
    return Path(value) if value else repo_root() / "workspace"


def local_root() -> Path:
    """Fast scratch root (local SSD on Colab). Falls back to :func:`root`."""
    value = os.environ.get(ENV_LOCAL)
    return Path(value) if value else root()


# ---- persistent locations (Drive) -------------------------------------------------

def raw_dir() -> Path:
    """Where the downloaded ATEPP zip is placed by hand."""
    return root() / "raw"


def cache_dir() -> Path:
    """Parsed per-performance cache (built in M2), persistent copy."""
    return root() / "cache"


def splits_dir() -> Path:
    return root() / "splits"


def features_dir() -> Path:
    return root() / "features"


def results_dir() -> Path:
    return root() / "results"


# ---- fast scratch locations (local SSD) -------------------------------------------

def dataset_dir() -> Path:
    """Where the ATEPP zip is extracted (re-extracted in each Colab session)."""
    return local_root() / "atepp"


def local_cache_dir() -> Path:
    """Local-disk copy of :func:`cache_dir` used during training."""
    return local_root() / "cache"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
