# Data notes: ATEPP v1.2

Status: **filled from the first audit run** (`notebooks/00_setup_and_data_audit.ipynb` on the
local ATEPP-1.2 archive, 2026-11-21; output in `results/audit/audit_report.md`). Values still
marked `TBD` are not answered by the audit.

Citation: Zhang, H., Tang, J., Rafee, S. R. M., Dixon, S., & Fazekas, G. (2022).
*ATEPP: A Dataset of Automatically Transcribed Expressive Piano Performance.*
Proceedings of ISMIR 2022. License: CC BY 4.0.

## 1. Files

| File | Content | Status |
|---|---|---|
| `ATEPP-metadata-1.2.csv` | one row per performance (the main table) | header seen on GitHub |
| `ATEPP-metadata-1.0.csv`, `-1.1.csv` | older versions | not in this archive |
| `repeats.csv` | 65 duplicate audios removed in v1.1 (audio fingerprinting) | not in this archive |
| MIDI files | `Composer/Work/Movement/NNNNN.mid` relative to the dataset root | 11839 files; every metadata row has a file (matched by `perf_id`); 165 files are not in the metadata; 98 % of relative paths resolve |

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

Verified against the archive: all 14 configured columns present, 11674 rows. Some `perf_id`s carry a suffix (`02152-1`, `01747-2`), so they are not always 5 digits.

## 3. Counts after quality filtering

Filter: drop `low quality`, `background noise` and `corrupted` (see DECISIONS D-008).

| | Before | After |
|---|---|---|
| performances | 11674 | 8927 |
| hours | 1001.8 | 749.9 |
| pianists | 49 | 49 |
| composition_ids (movements) | 1595 | 1525 |
| works (`work_key`) | 487 | 466 |

Per-pianist table (performances, distinct works, distinct movements, hours, albums,
year range): `results/audit/pianist_stats_filtered.csv`. Top 5 by distinct works: Ashkenazy (234), Richter (160), Arrau (124), Schiff (122), Barenboim (100). 10th: Perahia (67).

## 4. Composition hierarchy (this decides the split grouping)

- Does any `composition_id` occur in more than one work folder? No (0).
- Movements per work (distribution): median 2, mean 3.3, max 51 (Shostakovich Op. 87).
- Largest works by number of performances: Kinderszenen (351), Chopin Préludes Op. 28 (276), Appassionata (255). Very large groups (e.g. a whole set of
  preludes in one folder) make a 70/15/15 split harder but are the safe choice.
- Do title prefixes and work folders agree? Not reliably: 333 work folders have several title prefixes (spelling variants), so the folder is the work key.

## 5. Duplicates

- `repeats.csv` ids still present in the v1.2 metadata: `repeats.csv` not in this archive.
- Rows sharing a YouTube link: 14 (mostly a track and its `-1`/`-2` variant).
- Near-duplicates (same pianist + composition, duration within 1 %): 2402 pairs, mostly the
  same recording re-released on another album. They stay inside one work group (D-013) and
  are the reason for the random-vs-work split comparison (D-016).
- Byte-identical MIDI files: 20.

## 6. Recordings, albums and era (recording-disjoint split, E8)

- Fraction of rows with album / year: 100 % / 100 % (1928-2022); 1125 albums after filtering.
- Albums per pianist, largest-album share per pianist: TBD
- (pianist, work) pairs that appear on several albums: 866 of 2030.
- Recording-disjoint split feasible? TBD

## 7. MIDI properties (sample of 300 files)

- Ticks per beat / tempo changes: 384, a single tempo; one instrument per file.
- Onsets on a grid (1 ms / 10 ms)? No (1.5 % / 0.3 % of onsets); minimum inter-onset gap about 1.3 ms. If they are not quantised, chord asynchrony of
  10–30 ms is visible and 100 fps piano-rolls are justified.
- Fraction of inter-onset gaps below 30 ms / 50 ms (chord spread): median 39 % / 43 %.
- Very short notes (< 30 ms, possible transcription artefacts): median 1.8 % of notes.
- CC64 binary or continuous (half-pedal)? Every event is a (value > 0, then 0) pair, so
  the value is a per-press depth, not a continuous curve (D-017). CC67 in 298/300 files, CC66 never.
- MIDI length vs `track_duration`: ratio median 1.01 (0.80-1.19).

## 8. Contradictions with the original project brief

TBD after the audit.
