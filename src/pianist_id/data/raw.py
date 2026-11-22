"""Locate, download if needed, and extract the ATEPP archive.

The zip lives in ``<PIANIST_ID_ROOT>/raw/`` (Drive on Colab). If none is there,
:func:`ensure_dataset` downloads ATEPP v1.2 from the public Google Drive link of the
ATEPP repository (github.com/tangjjbetsy/ATEPP, CC BY 4.0) with ``gdown`` and keeps it in
``raw/``, so later sessions reuse it. The zip is then extracted to fast local disk once
per session.
"""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

from pianist_id import paths

EXTRACT_MARKER = ".extracted"
ATEPP_GDRIVE_ID = "1Df2KUdqvXtgvhvzx2D10YaQRbpP23rOS"   # ATEPP-1.2.zip, linked from the ATEPP README
ATEPP_ZIP_NAME = "ATEPP-1.2.zip"


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


def download_archive(raw_dir: Path | None = None, file_id: str = ATEPP_GDRIVE_ID) -> Path:
    """Download the ATEPP zip into ``raw_dir`` with gdown and return its path.

    The download goes to a ``.part`` file that is renamed only after it is checked to be
    a zip, so an interrupted download is never picked up by :func:`find_archive`.
    """
    import gdown   # preinstalled on Colab; only needed when the zip is missing

    raw_dir = paths.ensure_dir(raw_dir or paths.raw_dir())
    target = raw_dir / ATEPP_ZIP_NAME
    part = target.with_name(target.name + ".part")
    print(f"No ATEPP zip in {raw_dir}; downloading it from Google Drive (id {file_id}) ...")
    error: Exception | None = None
    try:   # older gdown returns None on failure, newer versions raise
        result = gdown.download(id=file_id, output=str(part), quiet=False)
    except Exception as e:  # noqa: BLE001 - re-raised below with a clear message
        result, error = None, e
    if result is None or not part.exists() or not zipfile.is_zipfile(part):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Downloading ATEPP (Google Drive id {file_id}) failed. Download "
                           f"it by hand from github.com/tangjjbetsy/ATEPP into {raw_dir}.") from error
    part.replace(target)
    return target


def ensure_dataset(download: bool = True) -> Path:
    """Find the zip (downloading it if missing and ``download``), extract it to local
    disk if needed, return the folder."""
    try:
        zip_path = find_archive()
    except FileNotFoundError:
        if not download:
            raise
        zip_path = download_archive()
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
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue    # macOS resource forks in the official zip, e.g. __MACOSX/.../._05855.mid
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
