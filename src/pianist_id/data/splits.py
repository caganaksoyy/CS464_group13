"""Train / val / test splits at the performance level.

A split is a DataFrame with one row per performance and the columns ``perf_id``,
``pianist``, ``group`` and ``split`` (one of :data:`SPLIT_NAMES`). It says nothing about
segments or features, so feature-based and sequence models read the same split CSVs.

Two kinds:

* :func:`group_split` -- whole groups (works, D-013) go to one side, so the test set
  only contains compositions never seen in training. This is the real task.
* :func:`random_split` -- performances are shuffled per pianist, ignoring groups. The
  same piece (and re-releases of the same recording) can then appear on both sides;
  it exists to measure that leakage (D-016).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pianist_id import paths

SPLIT_NAMES = ("train", "val", "test")


def _check_fractions(fractions: list[float]) -> np.ndarray:
    f = np.asarray(fractions, dtype=float)
    if f.shape != (3,) or (f <= 0).any() or not np.isclose(f.sum(), 1.0):
        raise ValueError(f"fractions must be three positive numbers summing to 1, got {fractions}")
    return f


def _frame(df: pd.DataFrame, perf_col: str, label_col: str, group_col: str,
           split: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "perf_id": df[perf_col].to_numpy(),
        "pianist": df[label_col].to_numpy(),
        "group": df[group_col].to_numpy(),
        "split": split,
    })


def missing_pianists(split_df: pd.DataFrame) -> dict[str, list[str]]:
    """Pianists absent from each split (empty lists when every split has everyone)."""
    everyone = set(split_df["pianist"])
    return {s: sorted(everyone - set(split_df.loc[split_df["split"] == s, "pianist"]))
            for s in SPLIT_NAMES}


def _greedy_group_assignment(sizes: pd.Series, targets: np.ndarray,
                             rng: np.random.Generator) -> dict[object, str]:
    """Shuffle groups, then give each one to the split furthest below its target size."""
    order = rng.permutation(len(sizes))
    filled = np.zeros(3)
    assignment: dict[object, str] = {}
    for i in order:
        k = int(np.argmax((targets - filled) / targets))
        assignment[sizes.index[i]] = SPLIT_NAMES[k]
        filled[k] += sizes.iloc[i]
    return assignment


def group_split(df: pd.DataFrame, *, perf_col: str, label_col: str, group_col: str,
                fractions: list[float], seed: int, max_attempts: int = 100) -> pd.DataFrame:
    """Group-disjoint split in which every pianist appears in train, val and test.

    Fractions are by performance count. A draw that leaves a pianist out of any split
    is rejected with a warning and redrawn from ``(seed, attempt)``; after
    ``max_attempts`` failures a ``RuntimeError`` is raised. The attempt that succeeded is
    stored in ``split_df.attrs["attempt"]``.
    """
    targets = _check_fractions(fractions) * len(df)
    sizes = df.groupby(group_col).size()
    for attempt in range(max_attempts):
        rng = np.random.default_rng([seed, attempt])
        assignment = _greedy_group_assignment(sizes, targets, rng)
        out = _frame(df, perf_col, label_col, group_col,
                     df[group_col].map(assignment).to_numpy())
        missing = {s: p for s, p in missing_pianists(out).items() if p}
        if not missing:
            out.attrs["attempt"] = attempt
            return out
        warnings.warn(f"group_split seed={seed} attempt={attempt}: pianists missing "
                      f"from {missing}; retrying", stacklevel=2)
    raise RuntimeError(f"group_split seed={seed}: no draw in {max_attempts} attempts had "
                       "every pianist in train, val and test")


def random_split(df: pd.DataFrame, *, perf_col: str, label_col: str, group_col: str,
                 fractions: list[float], seed: int) -> pd.DataFrame:
    """Performance-level split, stratified by pianist, ignoring groups."""
    f = _check_fractions(fractions)
    rng = np.random.default_rng(seed)
    split = np.empty(len(df), dtype=object)
    for _, idx in df.groupby(label_col).indices.items():
        idx = rng.permutation(idx)
        n_val = max(1, int(round(f[1] * len(idx))))
        n_test = max(1, int(round(f[2] * len(idx))))
        if len(idx) < n_val + n_test + 1:
            raise ValueError("A pianist has too few performances for a 3-way split")
        split[idx[:n_val]] = "val"
        split[idx[n_val:n_val + n_test]] = "test"
        split[idx[n_val + n_test:]] = "train"
    out = _frame(df, perf_col, label_col, group_col, split)
    out.attrs["attempt"] = 0
    return out


def split_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    """Performances per pianist and split, plus the number of groups per split."""
    table = split_df.pivot_table(index="pianist", columns="split", values="perf_id",
                                 aggfunc="count", fill_value=0)[list(SPLIT_NAMES)]
    table.loc["(groups)"] = [split_df.loc[split_df["split"] == s, "group"].nunique()
                             for s in SPLIT_NAMES]
    return table


def save_split(split_df: pd.DataFrame, name: str, directory: Path | None = None) -> Path:
    path = paths.ensure_dir(directory or paths.splits_dir()) / f"{name}.csv"
    split_df.to_csv(path, index=False)
    return path


def load_split(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"perf_id": str})
