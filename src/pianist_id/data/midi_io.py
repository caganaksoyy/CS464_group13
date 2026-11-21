"""MIDI file -> plain NumPy arrays.

A parsed performance is a :class:`Performance`:

* ``notes``  -- float32 array ``[n, 4]`` with columns (onset_s, offset_s, pitch,
  velocity), sorted by onset then pitch. All non-drum tracks are merged (ATEPP
  transcriptions are a single piano track, but we don't rely on it).
* ``pedals`` -- dict ``{cc_number: float32 array [m, 2]}`` of (time_s, value) for the
  pedal controllers 64 (sustain), 66 (sostenuto) and 67 (soft), when present.
* ``duration_s`` -- end time of the file.

M2 adds the cache format and a speed benchmark against ``symusic``; this module stays
the reference implementation used by the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi

PEDAL_CCS = (64, 66, 67)


@dataclass
class Performance:
    notes: np.ndarray                                   # [n, 4] float32
    pedals: dict[int, np.ndarray] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def n_notes(self) -> int:
        return int(self.notes.shape[0])


def load_performance(path: str | Path) -> Performance:
    """Parse one MIDI file. Raises on unreadable files (callers catch and log)."""
    pm = pretty_midi.PrettyMIDI(str(path))
    return performance_from_pretty_midi(pm)


def performance_from_pretty_midi(pm: pretty_midi.PrettyMIDI) -> Performance:
    note_rows: list[tuple[float, float, int, int]] = []
    cc_events: dict[int, list[tuple[float, int]]] = {cc: [] for cc in PEDAL_CCS}
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        note_rows.extend((n.start, n.end, n.pitch, n.velocity) for n in inst.notes)
        for cc in inst.control_changes:
            if cc.number in cc_events:
                cc_events[cc.number].append((cc.time, cc.value))
    notes = np.asarray(note_rows, dtype=np.float32).reshape(-1, 4)
    if len(notes):
        notes = notes[np.lexsort((notes[:, 2], notes[:, 0]))]
    pedals = {cc: np.asarray(sorted(ev), dtype=np.float32).reshape(-1, 2)
              for cc, ev in cc_events.items() if ev}
    return Performance(notes=notes, pedals=pedals, duration_s=float(pm.get_end_time()))


def summarize_midi(path: str | Path) -> dict[str, Any]:
    """Timing, dynamics and pedal statistics of one file, for the data audit.

    The timing statistics answer design questions for M2: are onsets quantised to a
    grid (which would hide chord asynchrony)? How spread are chord onsets? How short
    are the shortest notes (transcription artefacts)? Is the sustain pedal binary or
    continuous (half-pedalling)?
    """
    pm = pretty_midi.PrettyMIDI(str(path))
    perf = performance_from_pretty_midi(pm)
    all_ccs = sorted({cc.number for inst in pm.instruments for cc in inst.control_changes})
    out: dict[str, Any] = {
        "n_instruments": len(pm.instruments),
        "ticks_per_beat": pm.resolution,
        "n_tempo_changes": len(pm.get_tempo_changes()[0]),
        "duration_s": perf.duration_s,
        "n_notes": perf.n_notes,
        "cc_numbers": " ".join(map(str, all_ccs)),
    }
    if perf.n_notes:
        onset, offset, _, vel = perf.notes.T
        dur = offset - onset
        ioi = np.diff(np.unique(onset))           # gaps between distinct onset times
        on_ms = onset.astype(np.float64) * 1000.0
        out.update({
            "notes_per_s": perf.n_notes / max(perf.duration_s, 1e-6),
            "vel_min": int(vel.min()), "vel_max": int(vel.max()),
            "vel_n_unique": int(len(np.unique(vel))),
            "frac_onsets_on_1ms_grid": float(np.mean(np.abs(on_ms - np.round(on_ms)) < 1e-3)),
            "frac_onsets_on_10ms_grid": float(np.mean(np.abs(on_ms / 10 - np.round(on_ms / 10)) < 1e-3)),
            "min_ioi_ms": float(ioi.min() * 1000) if len(ioi) else np.nan,
            "frac_ioi_below_30ms": float(np.mean(ioi < 0.030)) if len(ioi) else np.nan,
            "frac_ioi_below_50ms": float(np.mean(ioi < 0.050)) if len(ioi) else np.nan,
            "dur_p01_ms": float(np.percentile(dur, 1) * 1000),
            "dur_median_ms": float(np.median(dur) * 1000),
            "frac_notes_below_30ms": float(np.mean(dur < 0.030)),
        })
    sustain = perf.pedals.get(64)
    out["has_cc64"] = sustain is not None
    if sustain is not None:
        values = sustain[:, 1]
        out["cc64_n_events"] = int(len(values))
        out["cc64_n_unique_values"] = int(len(np.unique(values)))
        out["cc64_frac_intermediate"] = float(np.mean((values > 0) & (values < 127)))
    out["has_cc66"] = 66 in perf.pedals
    out["has_cc67"] = 67 in perf.pedals
    return out
