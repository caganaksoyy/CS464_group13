"""Config loading, path resolution, seeding and run logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from pianist_id import paths
from pianist_id.colab import copy_dir_if_missing
from pianist_id.utils import config as C
from pianist_id.utils.logging import CSVLogger, read_json, save_run_info
from pianist_id.utils.seed import seed_everything


def test_data_config_default_and_tiny(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(C.ENV_TINY, raising=False)
    cfg = C.load_data_config()
    assert cfg.columns.perf_id == "perf_id"
    assert cfg.columns.pianist == "artist"
    assert cfg.selection.n_classes == [10, 20]
    tiny = C.load_data_config(tiny=True)
    assert tiny.selection.n_classes == [3]
    assert tiny.columns.pianist == "artist"          # untouched keys survive the merge
    monkeypatch.setenv(C.ENV_TINY, "1")
    assert C.load_data_config().segments.length_s == 5.0


def test_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(KeyError, match="typo"):
        C.from_dict(C.DataConfig, {"selection": {"typo": 1}})


@dataclass
class Inner:
    a: int = 1
    b: int = 2


@dataclass
class Outer:
    inner: Inner = field(default_factory=Inner)
    name: str = "x"


def test_from_dict_nested_and_deep_merge() -> None:
    merged = C.deep_merge({"inner": {"a": 5, "b": 6}}, {"inner": {"b": 7}})
    obj = C.from_dict(Outer, merged)
    assert obj.inner == Inner(5, 7) and obj.name == "x"


def test_paths_follow_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path / "drive"))
    monkeypatch.delenv(paths.ENV_LOCAL, raising=False)
    assert paths.raw_dir() == tmp_path / "drive" / "raw"
    assert paths.local_root() == tmp_path / "drive"       # falls back to root
    monkeypatch.setenv(paths.ENV_LOCAL, str(tmp_path / "ssd"))
    assert paths.dataset_dir() == tmp_path / "ssd" / "atepp"


def test_copy_dir_if_missing(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "a.txt").write_text("hi")
    assert copy_dir_if_missing(src, dst) is True
    assert (dst / "a.txt").read_text() == "hi" and (dst / ".complete").exists()
    assert copy_dir_if_missing(src, dst) is False          # already complete
    assert copy_dir_if_missing(src, src) is False          # same folder: never deletes
    assert (src / "a.txt").exists()


def test_seed_everything_reproducible() -> None:
    seed_everything(123)
    a = np.random.rand(3)
    seed_everything(123)
    assert np.allclose(a, np.random.rand(3))


def test_run_info_and_csv_logger(tmp_path: Path) -> None:
    run = save_run_info(tmp_path / "run", {"lr": 0.1}, seed=0)
    assert (run / "config.yaml").exists()
    info = read_json(run / "run_info.json")
    assert info["seed"] == 0 and "packages" in info
    log = CSVLogger(run / "history.csv")
    log.log({"epoch": 0, "loss": 1.0})
    log2 = CSVLogger(run / "history.csv")                  # reopened, as after a resume
    log2.log({"epoch": 1, "loss": 0.5})
    assert (run / "history.csv").read_text().splitlines() == ["epoch,loss", "0,1.0", "1,0.5"]
    with pytest.raises(ValueError):
        log2.log({"epoch": 2})
