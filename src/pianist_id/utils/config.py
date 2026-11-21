"""YAML configs loaded into dataclasses.

Each file in ``configs/`` corresponds to one top-level dataclass (``data.yaml`` ->
:class:`DataConfig`; model configs are added in later milestones). ``configs/tiny.yaml``
holds CPU smoke-test overrides keyed by config name; they are merged on top when tiny
mode is active (``PIANIST_ID_TINY=1`` or ``tiny=True``).

Unknown keys raise an error, so a typo in a YAML file fails loudly instead of being
silently ignored.
"""

from __future__ import annotations

import dataclasses
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from pianist_id import paths

ENV_TINY = "PIANIST_ID_TINY"

T = TypeVar("T")


# ---- data config ------------------------------------------------------------------

@dataclass
class ColumnsConfig:
    """Role name -> column name in the ATEPP metadata CSV."""

    perf_id: str = "perf_id"
    pianist: str = "artist"
    pianist_id: str = "artist_id"
    composer: str = "composer"
    composition_id: str = "composition_id"
    title: str = "track"
    duration_s: str = "track_duration"
    midi_path: str = "midi_path"
    score_path: str = "score_path"
    quality: str = "quality"
    repetition: str = "repetition"
    album: str = "album"
    album_date: str = "album_date"
    youtube: str = "youtube_links"


@dataclass
class QualityConfig:
    exclude_tags: list[str] = field(
        default_factory=lambda: ["low quality", "background noise", "corrupted"]
    )


@dataclass
class SelectionConfig:
    group_level: str = "work"          # "work" or "movement"
    min_compositions: int = 8
    n_classes: list[int] = field(default_factory=lambda: [10, 20])


@dataclass
class SplitsConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    fractions: list[float] = field(default_factory=lambda: [0.70, 0.15, 0.15])
    n_folds: int = 5


@dataclass
class SegmentsConfig:
    length_s: float = 20.0
    hop_train_s: float = 10.0
    hop_eval_s: float = 20.0
    min_notes: int = 10


@dataclass
class AuditConfig:
    midi_sample_size: int = 300
    near_duplicate_tol: float = 0.01
    k_values: list[int] = field(default_factory=lambda: [3, 5, 8, 10, 12, 15, 20, 30])


@dataclass
class DataConfig:
    metadata_file: str = "ATEPP-metadata-1.2.csv"
    repeats_file: str = "repeats.csv"
    columns: ColumnsConfig = field(default_factory=ColumnsConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    segments: SegmentsConfig = field(default_factory=SegmentsConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)


# ---- generic helpers --------------------------------------------------------------

def is_tiny() -> bool:
    """True when smoke-test mode is switched on through the environment."""
    return os.environ.get(ENV_TINY, "0").lower() in ("1", "true", "yes")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` with ``override`` merged in (nested dicts merge)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def from_dict(cls: type[T], values: dict[str, Any]) -> T:
    """Build dataclass ``cls`` from a (possibly nested) dict.

    Nested dataclass fields are built recursively. Missing keys keep their defaults.
    Unknown keys raise ``KeyError``.
    """
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"Unknown key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            kwargs[name] = from_dict(hint, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def to_dict(cfg: Any) -> dict[str, Any]:
    """Dataclass -> plain dict (for saving next to results)."""
    return dataclasses.asdict(cfg)


def save_yaml(values: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(values, f, sort_keys=False, allow_unicode=True)


def load_raw_config(name: str, tiny: bool | None = None) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` as a dict, with tiny overrides if active."""
    values = load_yaml(paths.configs_dir() / f"{name}.yaml")
    if tiny is None:
        tiny = is_tiny()
    if tiny:
        overrides = load_yaml(paths.configs_dir() / "tiny.yaml").get(name, {})
        values = deep_merge(values, overrides)
    return values


def load_data_config(tiny: bool | None = None) -> DataConfig:
    """Load ``configs/data.yaml`` (+ tiny overrides) into a :class:`DataConfig`."""
    return from_dict(DataConfig, load_raw_config("data", tiny))
