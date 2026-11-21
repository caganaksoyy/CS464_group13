# Decisions log

Small design decisions, with the reason, in the order they were made. Big or
expensive-to-reverse decisions (split granularity, class set, tokenisation) are asked
to the project owner first and recorded here once settled.

| ID | Decision | Reason |
|---|---|---|
| D-001 | The repository root is this folder. On GitHub it is `CS464_group13` (private). Colab clones it with a fine-grained personal access token stored in Colab Secrets (`GITHUB_USER`, `GITHUB_TOKEN`). | Private repo; keeps the token out of notebooks and output (the setup cell masks it in error messages). |
| D-002 | Local development uses Python 3.12 in `.venv`. | Stays close to Colab's Python version (the machine's default is 3.14), so local tests behave like Colab. |
| D-003 | Two storage roots: `PIANIST_ID_ROOT` (Drive, persistent) and `PIANIST_ID_LOCAL` (Colab VM disk, fast). The archive is re-extracted to local disk each session; the cache is copied there. | Reading ~11k small files from Drive is slow. Extracting 212 MB takes about a minute. |
| D-004 | `requirements.txt` pins only packages Colab does not preinstall. Preinstalled ones (torch, numpy, pandas, scikit-learn, ...) are left alone; `pip install -e . --no-deps`. Versions are printed by every setup cell and saved with every run. | Avoids breaking Colab's CUDA torch and its pinned numpy/pandas; still reproducible through the recorded versions. |
| D-005 | Notebooks (`.ipynb`) are the source of truth. The tests execute them top to bottom with `nbclient` in tiny mode on synthetic data, instead of converting them to scripts. | Same goal as a jupytext conversion (catch notebook breakage) with one less moving part. |
| D-006 | Column names come from the public `ATEPP-metadata-1.2.csv` header on GitHub and live in `configs/data.yaml`. Notebook 00 verifies them against the archive. | The audit can confirm instead of guess; the synthetic fixture mirrors the real schema. |
| D-007 | MIDI files are found by their 5-digit file name (`perf_id`), not by the relative `midi_path`. | Robust to how the zip encodes non-ASCII folder names (`Mélodie`, `Étude`). |
| D-008 | The quality filter drops `low quality`, `background noise` **and** `corrupted`. | The authors recommend the first two; `corrupted` marks confirmed-broken transcriptions. |
| D-009 | Provisional: the split group is the **work**, read from the second folder of `midi_path`. | Movements of one sonata must not straddle train and test. Will be confirmed or changed after the audit (DATA_NOTES §4). |
| D-010 | Logging goes to CSV/JSON files; no Weights & Biases. | Project owner's choice. |
| D-011 | The one-page proposal is deferred. | Project owner's choice; team details not yet available. |
| D-012 | pandas 2.2.3 locally; code avoids pandas-3-only behaviour. | Colab most likely ships pandas 2.2.x; to be checked against the first setup-cell output. |
| D-013 | The split group is the **work** (confirms D-009). | Audit: no `composition_id` occurs in two work folders, so work groups contain whole movements. Near-duplicate re-releases (same pianist + movement) stay inside one group. |
| D-014 | K (`min_compositions`) stays 8; it does not bind for N<=10. | Audit: the 10th pianist by distinct works has 67 works; 43 pianists have >= 8. |
| D-015 | Main result uses N=5 pianists, N=10 is an extra run from the same feature table; both ranked by distinct works. | Demo scope. N=5: 3716 performances, 434 works, 296 h. N=10: 5666 performances, 447 works, 469 h. |
| D-016 | The XGBoost demo also runs a performance-level random split next to the work-disjoint split (leakage analysis). | Audit: 2402 near-duplicate pairs (same pianist and movement, duration within 1 %, mostly re-releases); a random split puts copies on both sides and inflates accuracy. |
| D-017 | Pedal features use on-time and on-value, not a half-pedal fraction. | Audit: every CC64/CC67 event is a (value > 0, then 0) pair, so the fraction of intermediate values is exactly 0.5 in every file and carries no information. |

## Open

- Whether a recording-disjoint split (by album) is feasible.
- How to treat near-duplicate performances beyond keeping them in one work group (e.g. dropping them for the random split).
