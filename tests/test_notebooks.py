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
