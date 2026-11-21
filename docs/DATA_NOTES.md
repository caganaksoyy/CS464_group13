# Data notes: ATEPP v1.2

Status: **skeleton.** The column roles below come from the public metadata CSV header on
GitHub (`tangjjbetsy/ATEPP`, `ATEPP-metadata-1.2.csv`). Every `TBD (audit)` value gets
filled from the output of `notebooks/00_setup_and_data_audit.ipynb` run on the archive
that was actually downloaded (`results/audit/audit_report.md` on Drive).

Citation: Zhang, H., Tang, J., Rafee, S. R. M., Dixon, S., & Fazekas, G. (2022).
*ATEPP: A Dataset of Automatically Transcribed Expressive Piano Performance.*
Proceedings of ISMIR 2022. License: CC BY 4.0.

## 1. Files

| File | Content | Status |
|---|---|---|
| `ATEPP-metadata-1.2.csv` | one row per performance (the main table) | header seen on GitHub |
| `ATEPP-metadata-1.0.csv`, `-1.1.csv` | older versions | compared in audit §1 |
| `repeats.csv` | 65 duplicate audios removed in v1.1 (audio fingerprinting) | TBD (audit §6) |
| MIDI files | `Composer/Work/Movement/NNNNN.mid` relative to the dataset root | TBD (audit §5) |

## 2. Column roles

| Role | Column | Notes |
|---|---|---|
| performance id | `perf_id` | 5 digits, equals the MIDI file name. Read as string (leading zeros). |
| pianist (label) | `artist` | `artist_id` is a numeric id for the same thing |
| composer | `composer` | |
| movement / piece | `composition_id` | appears to be **per movement** (e.g. the four pieces of Rachmaninoff Op. 1 have different ids) |
| work | *derived* `work_key` | first two folders of `midi_path` (`Composer/Work`); cross-checked with the track-title prefix before `:` |
| title | `track` | `"Work title: movement title"` |
| duration | `track_duration` | seconds |
| MIDI path | `midi_path` | relative path |
| quality | `quality` | tags: `low quality`, `background noise`, `applause`, `high quality`, `corrupted`; mostly empty |
| repeats played | `repetition` | authors say "not thoroughly analysed"; appears mostly empty |
| album | `album` | album title |
| date | `album_date` | full date or year only; parsed to a 4-digit `year` |
| recording link | `youtube_links` | source video URL |
| score | `score_path` | not used by this project |

Verified against the archive: TBD (audit §2: "Configured column roles present?").

## 3. Counts after quality filtering

Filter: drop `low quality`, `background noise` and `corrupted` (see DECISIONS D-008).

| | Before | After |
|---|---|---|
| performances | TBD | TBD |
| hours | TBD | TBD |
| pianists | TBD | TBD |
| composition_ids (movements) | TBD | TBD |
| works (`work_key`) | TBD | TBD |

Per-pianist table (performances, distinct works, distinct movements, hours, albums,
year range): TBD (copy from `pianist_stats_filtered.csv`).

## 4. Composition hierarchy (this decides the split grouping)

- Does any `composition_id` occur in more than one work folder? TBD
- Movements per work (distribution): TBD
- Largest works by number of performances: TBD. Very large groups (e.g. a whole set of
  preludes in one folder) make a 70/15/15 split harder but are the safe choice.
- Do title prefixes and work folders agree? TBD

## 5. Duplicates

- `repeats.csv` ids still present in the v1.2 metadata: TBD
- Rows sharing a YouTube link: TBD
- Near-duplicates (same pianist + composition, duration within 1 %): TBD pairs.
  Decision on how to treat them: TBD (M2)
- Byte-identical MIDI files: TBD

## 6. Recordings, albums and era (recording-disjoint split, E8)

- Fraction of rows with album / year: TBD
- Albums per pianist, largest-album share per pianist: TBD
- (pianist, work) pairs that appear on several albums: TBD
- Recording-disjoint split feasible? TBD

## 7. MIDI properties (sample of 300 files)

- Ticks per beat / tempo changes: TBD
- Onsets on a grid (1 ms / 10 ms)? TBD. If they are not quantised, chord asynchrony of
  10–30 ms is visible and 100 fps piano-rolls are justified.
- Fraction of inter-onset gaps below 30 ms / 50 ms (chord spread): TBD
- Very short notes (< 30 ms, possible transcription artefacts): TBD
- CC64 binary or continuous (half-pedal)? CC66/CC67 present? TBD
- MIDI length vs `track_duration`: TBD

## 8. Contradictions with the original project brief

TBD after the audit.
