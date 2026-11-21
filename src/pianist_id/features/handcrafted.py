"""Handcrafted expressive-performance features, one row per segment.

Every feature belongs to one group in :data:`FEATURE_GROUPS`, so importances can be read
per musical aspect. The ``content`` group describes *what* is played (register,
polyphony) rather than *how*; it is kept apart so the share of the model that relies on
the piece instead of the pianist is visible.

Chords are onset clusters: a new chord starts when a note begins more than
``CHORD_WINDOW_S`` after the first note of the current chord (anchored, so fast runs are
not chained into one chord). Pedal features use on-time and press value only, because
ATEPP stores every press as a (value > 0, then 0) pair (D-017).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pianist_id.data.midi_io import load_performance
from pianist_id.data.segments import Segment, segment_performance

CHORD_WINDOW_S = 0.05
TEMPO_SUBWINDOW_S = 2.0
PEDAL_CCS = {64: "sustain", 67: "soft"}

FEATURE_GROUPS: dict[str, str] = {
    **{f"vel_{k}": "dynamics" for k in ("mean", "std", "p10", "p50", "p90", "iqr")},
    "vel_pitch_corr": "dynamics",
    "vel_chord_diff_std": "dynamics",
    "mel_emph_mean": "melody_emphasis",
    "mel_emph_std": "melody_emphasis",
    "mel_top_loudest_frac": "melody_emphasis",
    "note_rate": "timing",
    "chord_rate": "timing",
    "ioi_log_mean": "timing",
    "ioi_log_std": "timing",
    "tempo_rate_std": "tempo_variability",
    "tempo_rate_cv": "tempo_variability",
    "ioi_logratio_std": "tempo_variability",
    "ioi_logratio_absmean": "tempo_variability",
    "async_mean_ms": "asynchrony",
    "async_median_ms": "asynchrony",
    "async_std_ms": "asynchrony",
    "async_top_lead_ms": "asynchrony",
    "dur_log_mean": "articulation",
    "dur_log_std": "articulation",
    "art_ratio_log_median": "articulation",
    "art_staccato_frac": "articulation",
    "art_overlap_frac": "articulation",
    "short_note_frac": "articulation",
    **{f"{name}_{k}": "pedal" for name in PEDAL_CCS.values()
       for k in ("on_frac", "press_rate", "press_dur_log_mean", "value_mean")},
    "pitch_min": "content",
    "pitch_max": "content",
    "pitch_range": "content",
    "pitch_mean": "content",
    "pitch_std": "content",
    "chord_size_mean": "content",
}
FEATURE_NAMES = list(FEATURE_GROUPS)
INDEX_COLUMNS = ["perf_id", "seg_idx", "start_s"]


def _nan_stat(fn, x: np.ndarray) -> float:
    return float(fn(x)) if len(x) else np.nan


def chord_ids(onsets: np.ndarray, window_s: float = CHORD_WINDOW_S) -> np.ndarray:
    """Chord index per note (onsets sorted ascending), anchored onset clustering."""
    ids = np.empty(len(onsets), dtype=np.int64)
    current, anchor = -1, -np.inf
    for i, t in enumerate(onsets):
        if t - anchor > window_s:
            current += 1
            anchor = t
        ids[i] = current
    return ids


def _pedal_features(events: np.ndarray | None, length_s: float, prefix: str) -> dict[str, float]:
    out = {f"{prefix}_on_frac": 0.0, f"{prefix}_press_rate": 0.0,
           f"{prefix}_press_dur_log_mean": np.nan, f"{prefix}_value_mean": np.nan}
    if events is None or not len(events):
        return out
    t, v = events[:, 0].astype(np.float64), events[:, 1]
    starts, durs, values = [], [], []
    for i in np.flatnonzero(v > 0):
        release = t[i + 1] if i + 1 < len(t) else length_s
        starts.append(t[i])
        durs.append(min(release, length_s) - t[i])
        values.append(v[i])
    if not starts:
        return out
    durs_arr = np.clip(np.asarray(durs), 0.0, None)
    out[f"{prefix}_on_frac"] = float(durs_arr.sum() / length_s)
    out[f"{prefix}_press_rate"] = len(starts) / length_s
    out[f"{prefix}_press_dur_log_mean"] = float(np.mean(np.log(durs_arr + 1e-3)))
    out[f"{prefix}_value_mean"] = float(np.mean(values))
    return out


def segment_features(seg: Segment) -> dict[str, float]:
    """All features of :data:`FEATURE_NAMES` for one segment (NaN where undefined)."""
    notes = seg.notes.astype(np.float64)
    onset, offset, pitch, vel = notes.T
    dur = np.clip(offset - onset, 1e-4, None)
    f: dict[str, float] = {}

    p10, p50, p90 = np.percentile(vel, [10, 50, 90])
    f.update(vel_mean=vel.mean(), vel_std=vel.std(), vel_p10=p10, vel_p50=p50, vel_p90=p90,
             vel_iqr=float(np.subtract(*np.percentile(vel, [75, 25]))))
    f["vel_pitch_corr"] = (float(np.corrcoef(vel, pitch)[0, 1])
                           if vel.std() > 0 and pitch.std() > 0 else np.nan)

    cid = chord_ids(onset)
    n_chords = int(cid[-1]) + 1
    first = np.r_[0, np.flatnonzero(np.diff(cid)) + 1]
    chord_onset = onset[first]
    chord_vel = np.bincount(cid, weights=vel) / np.bincount(cid)
    chord_size = np.bincount(cid)
    f["vel_chord_diff_std"] = _nan_stat(np.std, np.diff(chord_vel))

    emph, top_loudest, spread, top_lead = [], [], [], []
    bounds = np.r_[first, len(onset)]
    for c in np.flatnonzero(chord_size >= 2):
        sl = slice(bounds[c], bounds[c + 1])
        p, v, o = pitch[sl], vel[sl], onset[sl]
        top = int(np.argmax(p))
        emph.append(v[top] - v.mean())
        top_loudest.append(float(v[top] >= v.max()))
        spread.append((o.max() - o.min()) * 1000.0)
        top_lead.append((o.mean() - o[top]) * 1000.0)
    emph_a, spread_a = np.asarray(emph), np.asarray(spread)
    f["mel_emph_mean"] = _nan_stat(np.mean, emph_a)
    f["mel_emph_std"] = _nan_stat(np.std, emph_a)
    f["mel_top_loudest_frac"] = _nan_stat(np.mean, np.asarray(top_loudest))
    f["async_mean_ms"] = _nan_stat(np.mean, spread_a)
    f["async_median_ms"] = _nan_stat(np.median, spread_a)
    f["async_std_ms"] = _nan_stat(np.std, spread_a)
    f["async_top_lead_ms"] = _nan_stat(np.mean, np.asarray(top_lead))

    f["note_rate"] = len(onset) / seg.length_s
    f["chord_rate"] = n_chords / seg.length_s
    ioi = np.diff(chord_onset)
    log_ioi = np.log(ioi[ioi > 0]) if len(ioi) else np.array([])
    f["ioi_log_mean"] = _nan_stat(np.mean, log_ioi)
    f["ioi_log_std"] = _nan_stat(np.std, log_ioi)

    # Equal, half-open sub-windows; a tail shorter than one sub-window is left out.
    n_sub = int(seg.length_s // TEMPO_SUBWINDOW_S)
    sub = (chord_onset // TEMPO_SUBWINDOW_S).astype(np.int64)
    rates = (np.bincount(sub[sub < n_sub], minlength=n_sub) / TEMPO_SUBWINDOW_S
             if n_sub >= 2 else np.array([]))
    f["tempo_rate_std"] = _nan_stat(np.std, rates)
    f["tempo_rate_cv"] = (float(rates.std() / rates.mean())
                          if len(rates) and rates.mean() > 0 else np.nan)
    logratio = np.diff(log_ioi)
    f["ioi_logratio_std"] = _nan_stat(np.std, logratio)
    f["ioi_logratio_absmean"] = _nan_stat(np.mean, np.abs(logratio))

    log_dur = np.log(dur)
    f["dur_log_mean"] = log_dur.mean()
    f["dur_log_std"] = log_dur.std()
    f["short_note_frac"] = float(np.mean(dur < 0.030))
    has_next = cid < n_chords - 1
    if has_next.any():
        gap = chord_onset[cid[has_next] + 1] - onset[has_next]
        ratio = dur[has_next] / np.clip(gap, 1e-3, None)
        f["art_ratio_log_median"] = float(np.median(np.log(ratio)))
        f["art_staccato_frac"] = float(np.mean(ratio < 0.5))
        f["art_overlap_frac"] = float(np.mean(ratio > 1.0))
    else:
        f.update(art_ratio_log_median=np.nan, art_staccato_frac=np.nan, art_overlap_frac=np.nan)

    for cc, name in PEDAL_CCS.items():
        f.update(_pedal_features(seg.pedals.get(cc), seg.length_s, name))

    f.update(pitch_min=pitch.min(), pitch_max=pitch.max(), pitch_range=pitch.max() - pitch.min(),
             pitch_mean=pitch.mean(), pitch_std=pitch.std(), chord_size_mean=chord_size.mean())
    return {k: float(f[k]) for k in FEATURE_NAMES}


def performance_feature_rows(perf_id: str, midi_path: str | Path, length_s: float,
                             hop_s: float, min_notes: int) -> list[dict[str, float | str]]:
    """Parse one MIDI file and return one feature row per segment."""
    perf = load_performance(midi_path)
    return [{"perf_id": perf_id, "seg_idx": s.seg_idx, "start_s": s.start_s,
             **segment_features(s)}
            for s in segment_performance(perf, length_s, hop_s, min_notes)]


def build_feature_table(midi_files: dict[str, Path], length_s: float, hop_s: float,
                        min_notes: int, progress_every: int = 500) -> pd.DataFrame:
    """Feature rows for every ``perf_id -> path`` in ``midi_files``.

    Files that fail to parse are skipped and listed in ``table.attrs["failed"]``.
    """
    rows: list[dict[str, float | str]] = []
    failed: dict[str, str] = {}
    for i, (perf_id, path) in enumerate(midi_files.items(), 1):
        try:
            rows.extend(performance_feature_rows(perf_id, path, length_s, hop_s, min_notes))
        except Exception as e:  # noqa: BLE001 - corrupt files are reported, not fatal
            failed[perf_id] = f"{type(e).__name__}: {e}"
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(midi_files)} performances, {len(rows)} segments")
    table = pd.DataFrame(rows, columns=INDEX_COLUMNS + FEATURE_NAMES)
    table["perf_id"] = table["perf_id"].astype(str)
    table.attrs["failed"] = failed
    return table
