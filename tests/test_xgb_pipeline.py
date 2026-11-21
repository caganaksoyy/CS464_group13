"""Splits, segmentation, handcrafted features and the XGBoost runner (notebook 01)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pianist_id.data import metadata as md
from pianist_id.data import raw
from pianist_id.data import splits as sp
from pianist_id.data.midi_io import Performance, load_performance
from pianist_id.data.segments import Segment, on_eval_grid, segment_performance, window_starts
from pianist_id.features.handcrafted import (FEATURE_GROUPS, FEATURE_NAMES, build_feature_table,
                                             chord_ids, segment_features)
from pianist_id.models import xgb
from pianist_id.utils.config import load_data_config, load_raw_config

SPLIT_KW = dict(perf_col="perf_id", label_col="pianist", group_col="work", fractions=[0.7, 0.15, 0.15])


def _toy_table(n_pianists: int = 3, works_per_pianist: int = 8, perfs_per_work: int = 3) -> pd.DataFrame:
    rows = []
    for p in range(n_pianists):
        for w in range(works_per_pianist):
            for k in range(perfs_per_work):
                rows.append({"perf_id": f"{p}-{w}-{k}", "pianist": f"P{p}", "work": f"W{w + p}"})
    return pd.DataFrame(rows)


# ---- splits -------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_group_split_is_group_disjoint_and_complete(seed: int) -> None:
    df = _toy_table()
    out = sp.group_split(df, seed=seed, **SPLIT_KW)
    assert len(out) == len(df) and set(out["split"]) == set(sp.SPLIT_NAMES)
    assert (out.groupby("group")["split"].nunique() == 1).all()
    assert not any(sp.missing_pianists(out).values())
    assert out.equals(sp.group_split(df, seed=seed, **SPLIT_KW))


def test_group_split_warns_retries_and_fails_when_impossible() -> None:
    df = _toy_table()
    lonely = pd.DataFrame({"perf_id": ["x1", "x2"], "pianist": ["Lonely"] * 2,
                           "work": ["Wx", "Wy"]})     # 2 groups cannot cover 3 splits
    with pytest.warns(UserWarning, match="retrying"), pytest.raises(RuntimeError):
        sp.group_split(pd.concat([df, lonely]), seed=0, max_attempts=3, **SPLIT_KW)


def test_random_split_is_stratified_and_deterministic() -> None:
    df = _toy_table(works_per_pianist=10, perfs_per_work=2)
    out = sp.random_split(df, seed=0, **SPLIT_KW)
    counts = out.groupby(["pianist", "split"]).size().unstack()
    assert (counts["val"] == 3).all() and (counts["test"] == 3).all() and (counts["train"] == 14).all()
    assert out.equals(sp.random_split(df, seed=0, **SPLIT_KW))
    assert not out.equals(sp.random_split(df, seed=1, **SPLIT_KW))


def test_split_roundtrip(tmp_path: Path) -> None:
    out = sp.group_split(_toy_table(), seed=0, **SPLIT_KW)
    loaded = sp.load_split(sp.save_split(out, "toy", tmp_path))
    pd.testing.assert_frame_equal(loaded, out, check_dtype=False)


# ---- segments -----------------------------------------------------------------------

def _perf(onsets: np.ndarray, duration: float, pedals: dict | None = None) -> Performance:
    notes = np.column_stack([onsets, onsets + 0.1, np.full(len(onsets), 60), np.full(len(onsets), 64)])
    return Performance(notes=notes.astype(np.float32), pedals=pedals or {}, duration_s=duration)


def test_window_starts() -> None:
    assert window_starts(3.0, 5.0, 2.5).tolist() == [0.0]
    assert window_starts(12.5, 5.0, 2.5).tolist() == [0.0, 2.5, 5.0, 7.5]


def test_segment_performance_lengths_and_min_notes() -> None:
    onsets = np.r_[np.arange(0, 5, 0.25), [7.0]]          # dense first 5 s, one note later
    pedal = {64: np.array([[1.0, 80], [6.0, 0]], dtype=np.float32)}
    segs = segment_performance(_perf(onsets, 10.0, pedal), length_s=5.0, hop_s=2.5, min_notes=3)
    assert [s.seg_idx for s in segs] == [0, 1]             # windows starting at 5 s have 1 note
    assert all(s.notes[:, 0].min() >= 0 and s.notes[:, 0].max() < 5.0 for s in segs)
    assert segs[1].pedals[64][0].tolist() == [0.0, 80.0]   # held pedal carried into window 2.5 s
    assert on_eval_grid(np.array([0.0, 2.5, 5.0]), 5.0, 2.5).tolist() == [True, False, True]
    with pytest.raises(ValueError):
        on_eval_grid(np.array([0.0]), 3.0, 2.0)


# ---- features -----------------------------------------------------------------------

def _seg(rows: list[tuple[float, float, int, int]], pedals: dict | None = None,
         length: float = 5.0) -> Segment:
    return Segment(seg_idx=0, start_s=0.0, length_s=length,
                   notes=np.asarray(rows, dtype=np.float32), pedals=pedals or {})


def test_chord_ids_are_anchored() -> None:
    assert chord_ids(np.array([0.0, 0.03, 0.06, 0.2])).tolist() == [0, 0, 1, 2]


def test_melody_emphasis_asynchrony_and_tempo() -> None:
    rows = []
    for i in range(10):                                     # even 0.5 s chords, top note loud
        t = 0.5 * i
        rows += [(t, t + 0.4, 48, 50), (t + 0.01, t + 0.4, 52, 50), (t + 0.02, t + 0.4, 60, 80)]
    f = segment_features(_seg(rows))
    assert f["mel_emph_mean"] == pytest.approx(20.0)
    assert f["mel_top_loudest_frac"] == 1.0
    assert f["async_mean_ms"] == pytest.approx(20.0, abs=1e-3)
    assert f["async_top_lead_ms"] == pytest.approx(-10.0, abs=1e-3)   # top note is late
    assert f["ioi_logratio_std"] == pytest.approx(0.0, abs=1e-5)
    assert f["tempo_rate_std"] == pytest.approx(0.0)
    assert f["chord_size_mean"] == 3.0


def test_pedal_features() -> None:
    rows = [(0.5 * i, 0.5 * i + 0.3, 60, 60) for i in range(10)]
    pedal = {64: np.array([[0.0, 100], [1.0, 0], [3.0, 60]], dtype=np.float32)}  # held to the end
    f = segment_features(_seg(rows, pedal))
    assert f["sustain_on_frac"] == pytest.approx(3.0 / 5.0)
    assert f["sustain_press_rate"] == pytest.approx(2 / 5.0)
    assert f["sustain_value_mean"] == pytest.approx(80.0)
    assert f["soft_on_frac"] == 0.0 and np.isnan(f["soft_value_mean"])


def test_feature_groups_cover_every_feature() -> None:
    assert list(FEATURE_GROUPS) == FEATURE_NAMES
    assert {"content", "pedal", "melody_emphasis", "tempo_variability"} <= set(FEATURE_GROUPS.values())


@pytest.fixture
def synthetic_features(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_data_config()
    cols = cfg.columns
    meta_path = raw.find_files(dataset_dir, cfg.metadata_file)[0]
    meta = md.filter_quality(md.add_derived_columns(md.load_metadata(meta_path, cols), cols), cfg)
    index = raw.index_midi_files(dataset_dir)
    table = build_feature_table({p: index[p] for p in meta[cols.perf_id]}, cfg.segments.length_s,
                                cfg.segments.hop_train_s, cfg.segments.min_notes, progress_every=0)
    return meta, table


def test_features_on_synthetic_styles(synthetic_features: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    meta, table = synthetic_features
    assert not table.attrs["failed"]
    assert np.isfinite(table[["vel_mean", "async_mean_ms", "art_staccato_frac"]].to_numpy()).all()
    by = table.merge(meta[["perf_id", "artist"]], on="perf_id").groupby("artist")[FEATURE_NAMES].mean()
    assert by.loc["Pianist Bold", "vel_mean"] > by.loc["Pianist Dry", "vel_mean"] > by.loc["Pianist Soft", "vel_mean"]
    assert by.loc["Pianist Bold", "async_mean_ms"] > by.loc["Pianist Soft", "async_mean_ms"]
    assert by.loc["Pianist Bold", "sustain_on_frac"] > by.loc["Pianist Dry", "sustain_on_frac"]
    assert by["art_staccato_frac"].idxmax() == "Pianist Dry"


def test_run_experiment_beats_chance(synthetic_features: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    meta, table = synthetic_features
    cfg = load_data_config()
    cols, seg = cfg.columns, cfg.segments
    with warnings.catch_warnings():   # the tiny archive has 6 works, so redraws are expected
        warnings.simplefilter("ignore", UserWarning)
        split_df = sp.group_split(meta, perf_col=cols.perf_id, label_col=cols.pianist,
                                  group_col="work_key", fractions=cfg.splits.fractions, seed=0)
    res = xgb.run_experiment(table, split_df, FEATURE_NAMES, load_raw_config("xgb")["params"], 0,
                             seg.hop_train_s, seg.hop_eval_s)
    assert res.metrics["perf_acc"] > res.metrics["chance"]
    assert res.confusion.to_numpy().sum() == res.metrics["n_perf_test"]
    shares = xgb.group_importance(res.feature_importance, FEATURE_GROUPS)
    assert shares.sum() == pytest.approx(1.0)


def test_summarize_runs_formats_mean_and_std() -> None:
    runs = pd.DataFrame({"split": ["a", "a", "b", "b"], "perf_acc": [0.5, 0.7, 0.2, 0.2],
                         "perf_macro_f1": [0.1, 0.1, 0.3, 0.5], "seg_acc": [0.0, 0.0, 0.0, 0.0]})
    out = xgb.summarize_runs(runs, by=["split"])
    assert out.loc["a", "perf_acc"] == "0.600 ± 0.141"
    assert out.loc["b", "perf_macro_f1"] == "0.400 ± 0.141"
