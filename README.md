# Who Is Playing? Pianist identification from expressive piano MIDI

Given a segment of a classical piano performance in MIDI format, predict which pianist
played it. Many pianists in the dataset recorded the same pieces, so the notes alone
cannot reveal the answer; models must learn expressive style (timing, dynamics,
articulation, pedalling). Test sets contain only compositions never seen in training.

Data: ATEPP v1.2 (Zhang et al., ISMIR 2022), performance MIDI transcribed from audio.

**Status: Milestone 1 (data audit).** No results yet.

## Repository layout

```
configs/            YAML configs (data.yaml; tiny.yaml = CPU smoke-test overrides)
src/pianist_id/     the library the notebooks import
  paths.py          Drive vs local storage locations
  colab.py          session setup used by every notebook
  data/             archive extraction, metadata, MIDI parsing (cache/splits in M2)
  utils/            config loading, seeding, run logging
notebooks/          what you run on Colab (orchestration, plots, commentary)
tests/              pytest unit tests + notebook smoke runs on synthetic data
docs/               DATA_NOTES, DECISIONS, COLAB instructions
report/             report material (later milestones)
```

## Running on Colab

See [docs/COLAB.md](docs/COLAB.md) for the one-time setup (GitHub token as a Colab
secret, Drive layout, uploading the ATEPP zip). Then open a notebook and *Run all*.

| Notebook | Purpose | Runtime |
|---|---|---|
| `00_setup_and_data_audit.ipynb` | schema, quality tags, duplicates, per-pianist stats, MIDI properties | CPU, ~2–5 min |

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
| *placeholder, no results yet* | | | | |

## Citation

Zhang, H., Tang, J., Rafee, S. R. M., Dixon, S., & Fazekas, G. (2022). ATEPP: A Dataset
of Automatically Transcribed Expressive Piano Performance. *Proceedings of the 23rd
International Society for Music Information Retrieval Conference (ISMIR).*
Dataset license: CC BY 4.0.
