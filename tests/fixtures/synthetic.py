"""Synthetic stand-in for ATEPP, used by the tests and the notebook smoke runs.

It writes a zip that looks like the real archive (same metadata columns, same
``Composer/Work/Movement/NNNNN.mid`` layout) with 3 fake pianists whose styles differ
on purpose, so a working pipeline must beat chance:

* "Pianist Soft":   quiet (velocity ~45), tight chords (~2 ms spread), light pedal, legato
* "Pianist Bold":   loud (velocity ~90), late melody / spread chords (~25 ms), heavy pedal
* "Pianist Dry":    medium loudness, ~10 ms spread, almost no pedal, staccato,
                    half-pedal values when it does pedal

The *notes* of a movement are identical for every pianist (generated from the
movement's seed); only the expressive rendering differs, as in the real task.

Usage from the command line (writes ``<out>/raw/ATEPP-synthetic.zip``)::

    python -m tests.fixtures.synthetic <out_dir>
"""

from __future__ import annotations

import io
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

METADATA_COLUMNS = [
    "artist", "artist_id", "track", "track_duration", "composer", "composition_id",
    "score_path", "midi_path", "youtube_links", "quality", "perf_id", "album",
    "album_date", "repetition",
]


@dataclass(frozen=True)
class Style:
    velocity_mean: float
    velocity_std: float
    chord_spread_s: float      # std of onset offsets inside a chord
    melody_lead_s: float       # > 0: top note comes early
    articulation: float        # note duration / nominal duration
    pedal_duty: float          # fraction of beats with pedal down
    half_pedal: bool
    tempo: float               # global tempo factor


STYLES: dict[str, Style] = {
    "Pianist Soft": Style(45, 6, 0.002, 0.000, 1.10, 0.35, False, 1.00),
    "Pianist Bold": Style(90, 12, 0.025, 0.020, 0.90, 0.85, False, 0.92),
    "Pianist Dry":  Style(68, 8, 0.010, 0.005, 0.45, 0.05, True, 1.08),
}

# (composer, work, [movements]); one non-ASCII name to test path encoding.
WORKS = [
    ("Fake_Composer_A", "Sonata_No.1", ["I._Allegro", "II._Adagio", "III._Rondo"]),
    ("Fake_Composer_A", "Sonata_No.2", ["I._Moderato", "II._Presto"]),
    ("Fake_Composer_A", "Etude_Op.10", ["1._Étude"]),
    ("Fake_Composer_B", "Four_Pieces", ["1._Mélodie", "2._Gavotte"]),
    ("Fake_Composer_B", "Nocturne_Op.9", ["1._Nocturne"]),
    ("Fake_Composer_B", "Ballade_No.1", ["1._Ballade"]),
]


def movement_content(seed: int, n_beats: int = 40) -> list[tuple[float, list[int], int]]:
    """Score-like content: per beat (nominal time, accompaniment chord, melody pitch)."""
    rng = np.random.default_rng(seed)
    root = int(rng.integers(45, 57))
    events = []
    for b in range(n_beats):
        chord_root = root + int(rng.choice([0, 5, 7, 9]))
        chord = [chord_root, chord_root + 4, chord_root + 7]
        melody = chord_root + 12 + int(rng.choice([0, 2, 4, 5, 7, 9, 12]))
        events.append((b * 0.5, chord, melody))
    return events


def render_performance(content: list[tuple[float, list[int], int]], style: Style,
                       rng: np.random.Generator) -> pretty_midi.PrettyMIDI:
    """Turn score-like content into an expressive MIDI performance in ``style``."""
    pm = pretty_midi.PrettyMIDI(resolution=480)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    beat = 0.5 * style.tempo
    for t_nom, chord, melody in content:
        t = 0.2 + t_nom * style.tempo + rng.normal(0, 0.01)   # small tempo wobble
        nominal_dur = beat * style.articulation
        for pitch in chord:
            onset = t + rng.normal(0, style.chord_spread_s)
            vel = np.clip(rng.normal(style.velocity_mean - 10, style.velocity_std), 1, 127)
            piano.notes.append(pretty_midi.Note(int(vel), pitch, max(onset, 0.0),
                                                max(onset, 0.0) + nominal_dur))
        onset = t - style.melody_lead_s + rng.normal(0, style.chord_spread_s / 2)
        vel = np.clip(rng.normal(style.velocity_mean + 10, style.velocity_std), 1, 127)
        piano.notes.append(pretty_midi.Note(int(vel), melody, max(onset, 0.0),
                                            max(onset, 0.0) + nominal_dur))
        if rng.random() < style.pedal_duty:
            down = int(rng.integers(40, 100)) if style.half_pedal else 127
            piano.control_changes.append(pretty_midi.ControlChange(64, down, t + 0.01))
            piano.control_changes.append(pretty_midi.ControlChange(64, 0, t + beat * 0.95))
    pm.instruments.append(piano)
    return pm


def make_fake_atepp(out_dir: str | Path, seed: int = 0) -> Path:
    """Write ``<out_dir>/raw/ATEPP-synthetic.zip`` and return its path.

    Each pianist plays an overlapping subset of the works. A few rows get quality
    tags, one recording is duplicated on a second album (near-duplicate check) and
    ``repeats.csv`` lists one removed id.
    """
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    rows, files = [], {}
    perf_counter = 1
    pianists = list(STYLES)
    for p_idx, pianist in enumerate(pianists):
        style = STYLES[pianist]
        # every pianist skips one work, so overlap is partial
        works = [w for i, w in enumerate(WORKS) if i != (p_idx * 2) % len(WORKS)]
        for composer, work, movements in works:
            for movement in movements:
                comp_id = [f"{c}/{w}/{m}" for c, w, ms in WORKS for m in ms].index(
                    f"{composer}/{work}/{movement}")
                content = movement_content(seed=1000 + comp_id)
                pm = render_performance(content, style, rng)
                perf_id = f"{perf_counter:05d}"
                perf_counter += 1
                rel = f"{composer}/{work}/{movement}/{perf_id}.mid"
                buf = io.BytesIO()
                pm.write(buf)
                files[rel] = buf.getvalue()
                rows.append({
                    "artist": pianist, "artist_id": str(p_idx),
                    "track": f"{work.replace('_', ' ')}: {movement.replace('_', ' ')}",
                    "track_duration": round(pm.get_end_time(), 3),
                    "composer": composer.replace("_", " "),
                    "composition_id": str(comp_id), "score_path": "",
                    "midi_path": rel,
                    "youtube_links": f"https://www.youtube.com/watch?v=fake{perf_id}",
                    "quality": "", "perf_id": perf_id,
                    "album": f"{pianist} plays {composer.replace('_', ' ')}",
                    "album_date": f"{1960 + 10 * p_idx}-01-01", "repetition": "",
                })
    df = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    # quality tags on a few rows
    for i, tag in zip([1, 7, 12, 20, 25], ["low quality", "background noise", "applause",
                                           "high quality", "corrupted"]):
        df.loc[i % len(df), "quality"] = tag
    # near-duplicate: the same recording re-released on a compilation album
    dup = df.iloc[3].copy()
    dup["perf_id"] = f"{perf_counter:05d}"
    dup["album"] = "Greatest Hits (compilation)"
    dup["album_date"] = "1999"
    dup["track_duration"] = round(float(dup["track_duration"]) * 1.002, 3)
    dup_rel = dup["midi_path"].rsplit("/", 1)[0] + f"/{dup['perf_id']}.mid"
    files[dup_rel] = files[df.iloc[3]["midi_path"]]
    dup["midi_path"] = dup_rel
    df = pd.concat([df, dup.to_frame().T], ignore_index=True)

    repeats = pd.DataFrame({"perf_id": ["99999"], "midi_path": ["Removed/Work/1/99999.mid"]})

    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "ATEPP-synthetic.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ATEPP-1.2/ATEPP-metadata-1.2.csv", df.to_csv(index=False))
        zf.writestr("ATEPP-1.2/repeats.csv", repeats.to_csv(index=False))
        for rel, data in files.items():
            zf.writestr(f"ATEPP-1.2/{rel}", data)
    return zip_path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("workspace")
    print("Wrote", make_fake_atepp(target))
