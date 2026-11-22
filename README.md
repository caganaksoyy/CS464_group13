# Who Is Playing? Pianist identification from expressive piano MIDI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caganaksoyy/CS464_group13/blob/main/notebooks/01_xgboost_baseline.ipynb) notebook 01 (XGBoost demo)

Given a segment of a classical piano performance in MIDI format, predict which pianist
played it. Many pianists in the dataset recorded the same pieces, so the notes alone
cannot reveal the answer; models must learn expressive style (timing, dynamics,
articulation, pedalling). Test sets contain only compositions never seen in training.

Data: ATEPP v1.2 (Zhang et al., ISMIR 2022), performance MIDI transcribed from audio.

**Status: demo - XGBoost baseline on handcrafted expressive features** (notebook 01).

## Repository layout

```
configs/            YAML configs (data.yaml; tiny.yaml = CPU smoke-test overrides)
src/pianist_id/     the library the notebooks import
  paths.py          Drive vs local storage locations
  colab.py          session setup used by every notebook
  data/             archive extraction, metadata, MIDI parsing, splits, segments
  features/         handcrafted segment features, grouped by musical aspect
  models/           XGBoost baseline
  utils/            config loading, seeding, run logging
notebooks/          what you run on Colab (orchestration, plots, commentary)
tests/              pytest unit tests + notebook smoke runs on synthetic data
docs/               DATA_NOTES, DECISIONS, COLAB instructions
report/             report material (later milestones)
```

## Running on Colab

See [docs/COLAB.md](docs/COLAB.md) for the one-time setup (optional GitHub token as a
Colab secret, Drive layout; the ATEPP zip is downloaded automatically if it is not on
Drive). Then open a notebook, e.g. with the badge above, and *Run all*.

| Notebook | Purpose | Runtime |
|---|---|---|
| `00_setup_and_data_audit.ipynb` | schema, quality tags, duplicates, per-pianist stats, MIDI properties | CPU, ~2–5 min |
| `01_xgboost_baseline.ipynb` | segment features, XGBoost, work-disjoint vs random split, confusion matrix, feature-group importance | CPU, ~15–20 min |

## Local development (CPU)

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt -e .
.venv\Scripts\python -m pytest -q
```

The tests build a synthetic ATEPP-like archive (3 fake pianists with deliberately
different styles) and run every notebook top to bottom in tiny mode
(`pytest -m "not slow"` skips the notebook runs).

## Results

| Model | Split | N | Perf. acc | Macro-F1 |
|---|---|---|---|---|
| XGBoost, handcrafted features | work-disjoint | 5 | 0.754 ± 0.023 | 0.761 ± 0.010 |
| XGBoost, handcrafted features | random (leaky) | 5 | 0.845 ± 0.005 | 0.843 ± 0.005 |
| XGBoost, handcrafted features | work-disjoint | 10 | 0.651 ± 0.021 | 0.639 ± 0.022 |
| XGBoost, handcrafted features | random (leaky) | 10 | 0.790 ± 0.010 | 0.769 ± 0.016 |

Mean ± std over split seeds 0, 1, 2; chance is 1/N. Performance accuracy aggregates the
20 s segments of each test performance. The work-disjoint split is the real task; the
random split lets the same piece and re-released recordings appear in train and test, and
the gap (0.09 for N=5, 0.14 for N=10) is that leakage (D-016).

## Citation

Zhang, H., Tang, J., Rafee, S. R. M., Dixon, S., & Fazekas, G. (2022). ATEPP: A Dataset
of Automatically Transcribed Expressive Piano Performance. *Proceedings of the 23rd
International Society for Music Information Retrieval Conference (ISMIR).*
Dataset license: CC BY 4.0.
