"""Archive extraction, metadata loading/cleaning and MIDI parsing on synthetic data."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
import pytest

from pianist_id.data import metadata as md
from pianist_id.data import raw
from pianist_id.data.midi_io import load_performance, summarize_midi
from pianist_id.utils.config import ColumnsConfig, load_data_config

COLS = ColumnsConfig()


# ---- archive ----------------------------------------------------------------------

def test_extract_is_idempotent_and_indexes_unicode_paths(storage: Path) -> None:
    d1 = raw.ensure_dataset()
    marker_time = (d1 / raw.EXTRACT_MARKER).stat().st_mtime
    d2 = raw.ensure_dataset()
    assert d1 == d2 and (d2 / raw.EXTRACT_MARKER).stat().st_mtime == marker_time
    index = raw.index_midi_files(d1)
    assert any("Mélodie" in str(p) for p in index.values())
    assert all(len(k) == 5 for k in index)


def test_find_midi_root_resolves_metadata_paths(dataset_dir: Path, metadata_path: Path) -> None:
    meta = md.load_metadata(metadata_path)
    root, frac = raw.find_midi_root(dataset_dir, meta["midi_path"].tolist())
    assert frac == 1.0 and root is not None and root.name == "ATEPP-1.2"


def test_zip_slip_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        raw.extract_archive(bad, tmp_path / "out")


def test_missing_archive_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="disclaimer"):
        raw.find_archive(tmp_path)


# ---- metadata ---------------------------------------------------------------------

@pytest.fixture
def meta(metadata_path: Path) -> pd.DataFrame:
    return md.add_derived_columns(md.load_metadata(metadata_path, COLS), COLS)


def test_ids_keep_leading_zeros(meta: pd.DataFrame) -> None:
    assert meta["perf_id"].iloc[0] == "00001"
    assert all(md.check_columns(meta, COLS).values())


def test_derived_columns(meta: pd.DataFrame) -> None:
    assert meta["work_key"].notna().all()
    assert meta["work_key"].str.count("/").eq(1).all()
    assert set(meta["year"].dropna()) >= {1960.0, 1999.0}
    rep = md.hierarchy_report(meta, COLS)
    assert rep["composition_ids_in_several_work_folders"] == 0
    assert rep["n_work_keys"] == 6


@pytest.mark.parametrize("value, expected", [
    (np.nan, ()), ("", ()), ("low quality", ("low quality",)),
    ("Low Quality; Applause", ("low quality", "applause")),
])
def test_parse_quality_tags(value: object, expected: tuple[str, ...]) -> None:
    assert md.parse_quality_tags(value) == expected


def test_quality_filter(meta: pd.DataFrame) -> None:
    cfg = load_data_config()
    kept = md.filter_quality(meta, cfg)
    assert len(meta) - len(kept) == 3          # low quality, background noise, corrupted
    remaining = set(kept["quality_tags"].explode().dropna())
    assert {"applause", "high quality"} <= remaining
    assert not remaining & {"low quality", "background noise", "corrupted"}
    counts = md.quality_tag_counts(meta)
    assert counts["<none>"] == len(meta) - 5


def test_per_pianist_stats_and_selection(meta: pd.DataFrame) -> None:
    stats = md.per_pianist_stats(meta, COLS)
    assert len(stats) == 3
    assert (stats["n_works"] == 5).all()        # each pianist skips one of 6 works
    assert stats["hours"].gt(0).all()
    chosen = md.select_pianists(stats, n=2, min_compositions=5)
    assert len(chosen) == 2
    with pytest.raises(ValueError):
        md.select_pianists(stats, n=3, min_compositions=6)
    elig = md.eligibility_table(stats, [1, 5, 6])
    assert elig["pianists_with_K_works"].tolist() == [3, 3, 0]


def test_near_duplicates_found(meta: pd.DataFrame) -> None:
    near = md.near_duplicates(meta, COLS, tol=0.01)
    assert len(near) == 1
    assert {near.iloc[0]["a_album"], near.iloc[0]["b_album"]} >= {"Greatest Hits (compilation)"}


def test_overlap_and_album_report(meta: pd.DataFrame) -> None:
    pianists = sorted(meta["artist"].unique())
    ov = md.composition_overlap(meta, COLS, pianists)
    assert (ov.values == ov.values.T).all()
    assert (np.diag(ov.values) == 5).all()
    rep = md.album_report(meta, COLS)
    assert rep["album_column_present"] and rep["n_albums"] == 7   # 3 pianists x 2 composers + 1 compilation


def test_schema_report(meta: pd.DataFrame) -> None:
    rep = md.schema_report(meta)
    assert set(rep["column"]) >= set(COLS.__dict__.values())
    assert rep.loc[rep["column"] == "score_path", "frac_null"].item() == 1.0


# ---- MIDI parsing -----------------------------------------------------------------

def _perf_path(dataset_dir: Path, meta: pd.DataFrame, pianist: str) -> Path:
    pid = meta.loc[meta["artist"] == pianist, "perf_id"].iloc[0]
    return raw.index_midi_files(dataset_dir)[pid]


def test_load_performance(dataset_dir: Path, meta: pd.DataFrame) -> None:
    perf = load_performance(_perf_path(dataset_dir, meta, "Pianist Bold"))
    assert perf.notes.dtype == np.float32 and perf.notes.shape[1] == 4
    assert np.all(np.diff(perf.notes[:, 0]) >= 0)                  # sorted by onset
    assert np.all(perf.notes[:, 1] > perf.notes[:, 0])             # offset after onset
    assert perf.notes[:, 3].min() >= 1 and perf.notes[:, 3].max() <= 127
    assert 64 in perf.pedals and set(np.unique(perf.pedals[64][:, 1])) == {0, 127}
    assert perf.duration_s > 0


def test_half_pedal_and_summary(dataset_dir: Path, meta: pd.DataFrame) -> None:
    s = summarize_midi(_perf_path(dataset_dir, meta, "Pianist Dry"))
    assert s["has_cc64"] and s["cc64_frac_intermediate"] > 0
    assert 0 <= s["frac_ioi_below_30ms"] <= 1
    assert all(np.isfinite(v) for v in s.values() if isinstance(v, float))


def test_styles_differ_in_velocity(dataset_dir: Path, meta: pd.DataFrame) -> None:
    """Sanity check that the synthetic pianists really differ (labels carry signal)."""
    vel = {p: load_performance(_perf_path(dataset_dir, meta, p)).notes[:, 3].mean()
           for p in ("Pianist Soft", "Pianist Dry", "Pianist Bold")}
    assert vel["Pianist Soft"] < vel["Pianist Dry"] < vel["Pianist Bold"]


def test_empty_and_corrupt_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.mid"
    pretty_midi.PrettyMIDI().write(str(empty))
    perf = load_performance(empty)
    assert perf.n_notes == 0 and perf.notes.shape == (0, 4) and perf.pedals == {}
    assert summarize_midi(empty)["n_notes"] == 0
    bad = tmp_path / "bad.mid"
    bad.write_bytes(b"not a midi file")
    with pytest.raises(Exception):
        load_performance(bad)


def test_pedal_from_any_track_and_drums_ignored() -> None:
    pm = pretty_midi.PrettyMIDI()
    piano, drums = pretty_midi.Instrument(0), pretty_midi.Instrument(0, is_drum=True)
    piano.notes.append(pretty_midi.Note(80, 60, 1.0, 1.5))
    piano.notes.append(pretty_midi.Note(70, 64, 0.5, 1.2))
    drums.notes.append(pretty_midi.Note(100, 36, 0.0, 0.1))
    piano.control_changes += [pretty_midi.ControlChange(67, 127, 0.2),
                              pretty_midi.ControlChange(64, 127, 0.3)]
    pm.instruments += [piano, drums]
    buf = io.BytesIO()
    pm.write(buf)
    buf.seek(0)
    from pianist_id.data.midi_io import performance_from_pretty_midi
    perf = performance_from_pretty_midi(pretty_midi.PrettyMIDI(buf))
    assert perf.n_notes == 2 and perf.notes[0, 2] == 64             # sorted by onset
    assert set(perf.pedals) == {64, 67}
