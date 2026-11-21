"""Shared pytest fixtures: a synthetic ATEPP archive and an isolated storage root."""

from __future__ import annotations

from pathlib import Path

import pytest

from pianist_id import paths
from pianist_id.data import raw
from tests.fixtures.synthetic import make_fake_atepp


@pytest.fixture(scope="session")
def fake_zip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic ATEPP zip, built once per test session."""
    return make_fake_atepp(tmp_path_factory.mktemp("atepp_src"))


@pytest.fixture
def storage(tmp_path: Path, fake_zip: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fresh PIANIST_ID_ROOT / PIANIST_ID_LOCAL with the fake zip in raw/, tiny mode on."""
    root, local = tmp_path / "drive", tmp_path / "local"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / fake_zip.name).write_bytes(fake_zip.read_bytes())
    monkeypatch.setenv(paths.ENV_ROOT, str(root))
    monkeypatch.setenv(paths.ENV_LOCAL, str(local))
    monkeypatch.setenv("PIANIST_ID_TINY", "1")
    return root


@pytest.fixture
def dataset_dir(storage: Path) -> Path:
    """The extracted synthetic archive."""
    return raw.ensure_dataset()


@pytest.fixture
def metadata_path(dataset_dir: Path) -> Path:
    return raw.find_files(dataset_dir, "ATEPP-metadata-1.2.csv")[0]
