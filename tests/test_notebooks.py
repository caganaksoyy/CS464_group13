"""Run notebooks top to bottom in tiny mode on the synthetic archive.

Catches notebook breakage before anything is taken to Colab. Marked ``slow``; skip
with ``pytest -m "not slow"``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def run_notebook(name: str, timeout: int = 600) -> nbformat.NotebookNode:
    nb = nbformat.read(NOTEBOOKS / name, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(NOTEBOOKS)}})
    client.execute()   # environment variables are inherited by the kernel
    return nb


@pytest.mark.slow
def test_notebook_00_data_audit(storage: Path) -> None:
    run_notebook("00_setup_and_data_audit.ipynb")
    report = storage / "results" / "audit" / "audit_report.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    for heading in ("### Schema", "### Quality filtering", "### Per-pianist stats",
                    "### Duplicate summary", "### MIDI sample: distribution"):
        assert heading in text
    assert (storage / "results" / "audit" / "pianist_stats_filtered.csv").exists()


@pytest.mark.slow
def test_notebook_01_xgboost(storage: Path) -> None:
    run_notebook("01_xgboost_baseline.ipynb")
    run_dirs = sorted((storage / "results" / "xgb_baseline").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    for name in ("runs.csv", "summary_N3.csv", "confusion.png", "importance.png",
                 "importance_groups_N3.csv", "run_info.json", "summary.json"):
        assert (run_dir / name).exists(), name
    import pandas as pd
    runs = pd.read_csv(run_dir / "runs.csv")
    assert set(runs["split"]) == {"work", "random"} and len(runs) == 6
    work = runs[runs["split"] == "work"]
    assert work["perf_acc"].mean() > work["chance"].iloc[0]
