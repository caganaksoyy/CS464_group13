"""Locate and extract the ATEPP archive that was uploaded to Drive by hand.

We never download ATEPP ourselves (it requires accepting a disclaimer). The user puts
the zip into ``<PIANIST_ID_ROOT>/raw/``; :func:`ensure_dataset` extracts it to fast
local disk once per session.
"""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

from pianist_id import paths

EXTRACT_MARKER = ".extracted"


def find_archive(raw_dir: Path | None = None) -> Path:
    """Return the ATEPP zip in ``raw_dir``. Fails if there is none."""
    raw_dir = raw_dir or paths.raw_dir()
    zips = sorted(raw_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not zips:
        raise FileNotFoundError(
            f"No .zip file found in {raw_dir}. Download ATEPP v1.2 (after accepting the "
            "disclaimer at github.com/tangjjbetsy/ATEPP) and upload the zip there."
        )
    if len(zips) > 1:
        print(f"Several zips in {raw_dir}; using the newest: {zips[-1].name}")
    return zips[-1]


def _is_safe_member(name: str) -> bool:
    """Reject absolute paths and '..' components (zip-slip protection)."""
    p = PurePosixPath(name.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts


def extract_archive(zip_path: Path, dest: Path) -> Path:
    """Extract ``zip_path`` into ``dest`` unless that was already done completely."""
    marker = dest / EXTRACT_MARKER
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == zip_path.name:
        return dest
    paths.ensure_dir(dest)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        unsafe = [m for m in members if not _is_safe_member(m)]
        if unsafe:
            raise ValueError(f"Refusing to extract unsafe paths, e.g. {unsafe[:3]}")
        zf.extractall(dest)
    marker.write_text(zip_path.name, encoding="utf-8")
    return dest


def ensure_dataset() -> Path:
    """Find the zip on Drive, extract it to local disk if needed, return the folder."""
    zip_path = find_archive()
    dest = paths.dataset_dir()
    already = (dest / EXTRACT_MARKER).exists()
    extract_archive(zip_path, dest)
    print(f"ATEPP {'already extracted' if already else 'extracted'} at {dest} "
          f"(from {zip_path.name}, {zip_path.stat().st_size / 1e6:.0f} MB)")
    return dest


def find_files(dataset_dir: Path, name: str) -> list[Path]:
    """All files called ``name`` anywhere below ``dataset_dir``."""
    return sorted(dataset_dir.rglob(name))


def index_midi_files(dataset_dir: Path) -> dict[str, Path]:
    """Map file stem (e.g. ``'00001'``) -> path for every .mid/.midi file.

    Matching on the 5-digit file name is robust to differences in how the zip encodes
    non-ASCII folder names (e.g. 'Mélodie'), which could break the relative
    ``midi_path`` from the metadata.
    """
    index: dict[str, Path] = {}
    for p in dataset_dir.rglob("*"):
        if p.suffix.lower() in (".mid", ".midi") and p.is_file():
            index.setdefault(p.stem, p)
    return index


def find_midi_root(dataset_dir: Path, relative_paths: list[str]) -> tuple[Path | None, float]:
    """Find the folder that the metadata's relative ``midi_path`` values start from.

    Tries ``dataset_dir`` and every sub-folder up to two levels deep, and returns the
    one where the largest fraction of ``relative_paths`` exists, with that fraction.
    """
    candidates = [dataset_dir]
    for depth1 in dataset_dir.iterdir():
        if depth1.is_dir():
            candidates.append(depth1)
            candidates.extend(d for d in depth1.iterdir() if d.is_dir())
    sample = relative_paths[:200]
    best, best_frac = None, 0.0
    for cand in candidates:
        frac = sum((cand / r).exists() for r in sample) / max(len(sample), 1)
        if frac > best_frac:
            best, best_frac = cand, frac
    return best, best_frac
