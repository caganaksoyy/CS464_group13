"""Cut a parsed performance into fixed-length windows.

A :class:`Segment` holds the same arrays as :class:`~pianist_id.data.midi_io.Performance`
with times relative to the window start, so any model (handcrafted features, piano
rolls, token sequences) can be built from it. Notes belong to the window their onset
falls in; their offsets may run past the window end.

Windows are cut once with the training hop. Evaluation uses a coarser hop
(``hop_eval_s``, a multiple of ``hop_train_s``) and keeps only the windows whose start
lies on that grid (:func:`on_eval_grid`), so one segment table serves every split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pianist_id.data.midi_io import Performance


@dataclass
class Segment:
    seg_idx: int
    start_s: float
    length_s: float
    notes: np.ndarray                                   # [n, 4], times relative to start_s
    pedals: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def n_notes(self) -> int:
        return int(self.notes.shape[0])


def window_starts(duration_s: float, length_s: float, hop_s: float) -> np.ndarray:
    """Starts of full-length windows; a performance shorter than one window gets one."""
    if duration_s <= length_s:
        return np.zeros(1)
    n = int(np.floor((duration_s - length_s) / hop_s + 1e-9)) + 1
    return np.arange(n) * hop_s


def _clip_pedal(events: np.ndarray, start: float, end: float) -> np.ndarray:
    """Events inside [start, end), shifted to the window. A pedal already held at
    ``start`` gets a synthetic press at time 0 so the window knows it is down."""
    inside = events[(events[:, 0] >= start) & (events[:, 0] < end)].copy()
    inside[:, 0] -= start
    before = events[events[:, 0] < start]
    if len(before) and before[-1, 1] > 0:
        inside = np.vstack([np.array([[0.0, before[-1, 1]]], dtype=events.dtype), inside])
    return inside


def segment_performance(perf: Performance, length_s: float, hop_s: float,
                        min_notes: int) -> list[Segment]:
    """Windows of ``length_s`` every ``hop_s`` seconds with at least ``min_notes`` notes."""
    onsets = perf.notes[:, 0]
    segments = []
    for i, start in enumerate(window_starts(perf.duration_s, length_s, hop_s)):
        end = start + length_s
        lo, hi = np.searchsorted(onsets, [start, end], side="left")
        if hi - lo < min_notes:
            continue
        notes = perf.notes[lo:hi].copy()
        notes[:, :2] -= start
        pedals = {cc: _clip_pedal(ev, start, end) for cc, ev in perf.pedals.items()}
        segments.append(Segment(seg_idx=i, start_s=float(start), length_s=length_s,
                                notes=notes, pedals=pedals))
    return segments


def on_eval_grid(start_s: np.ndarray, hop_eval_s: float, hop_train_s: float) -> np.ndarray:
    """Boolean mask: which window starts lie on the evaluation hop grid."""
    ratio = hop_eval_s / hop_train_s
    if not np.isclose(ratio, round(ratio)) or ratio < 1:
        raise ValueError(f"hop_eval_s ({hop_eval_s}) must be a multiple of hop_train_s "
                         f"({hop_train_s})")
    k = np.asarray(start_s) / hop_eval_s
    return np.isclose(k, np.round(k))
