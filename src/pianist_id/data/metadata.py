"""Load, inspect and clean the ATEPP metadata table.

Column names come from :class:`pianist_id.utils.config.ColumnsConfig`, never hard-coded,
so a schema surprise only requires editing ``configs/data.yaml``.

Derived columns added by :func:`add_derived_columns`:

* ``work_key``     -- ``Composer/Work`` from the first two folders of ``midi_path``.
                      Movements of one sonata share it; it is the default split group.
* ``movement_key`` -- the ``composition_id`` (one per movement/piece).
* ``title_work``   -- the part of the track title before the first ':'; a second,
                      independent guess at the work, used only to cross-check.
* ``year``         -- 4-digit year parsed from ``album_date`` (NaN if missing).
* ``quality_tags`` -- tuple of normalised quality tags ('' -> empty tuple).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from pianist_id.utils.config import ColumnsConfig, DataConfig

GROUP_COLUMNS = {"work": "work_key", "movement": "movement_key"}


# ---- loading ----------------------------------------------------------------------

def load_metadata(path: str | Path, cols: ColumnsConfig | None = None) -> pd.DataFrame:
    """Read a metadata CSV. Id-like columns are read as strings (keeps leading zeros)."""
    cols = cols or ColumnsConfig()
    as_str = [cols.perf_id, cols.pianist_id, cols.composition_id, cols.midi_path,
              cols.quality, cols.album_date]
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: str for c in as_str if c in header}
    return pd.read_csv(path, dtype=dtypes)


def check_columns(df: pd.DataFrame, cols: ColumnsConfig) -> dict[str, bool]:
    """Which configured columns exist in ``df``."""
    return {role: (name in df.columns) for role, name in vars(cols).items()}


def _work_key_from_path(p: object) -> str | float:
    if not isinstance(p, str) or not p:
        return np.nan
    parts = p.replace("\\", "/").strip("/").split("/")
    return "/".join(parts[:2]) if len(parts) >= 3 else np.nan


def parse_quality_tags(value: object) -> tuple[str, ...]:
    """'Low Quality; applause' -> ('low quality', 'applause'). NaN/'' -> ()."""
    if not isinstance(value, str) or not value.strip():
        return ()
    return tuple(t.strip().lower() for t in re.split(r"[,;/|]", value) if t.strip())


def _year(value: object) -> float:
    if not isinstance(value, str):
        return np.nan
    m = re.search(r"(1[89]\d\d|20\d\d)", value)
    return float(m.group(1)) if m else np.nan


def add_derived_columns(df: pd.DataFrame, cols: ColumnsConfig) -> pd.DataFrame:
    """Return a copy of ``df`` with the derived columns described in the module doc."""
    out = df.copy()
    out["work_key"] = out[cols.midi_path].map(_work_key_from_path)
    out["movement_key"] = out[cols.composition_id].astype(str)
    if cols.title in out:
        out["title_work"] = out[cols.title].astype(str).str.split(":", n=1).str[0].str.strip()
    if cols.album_date in out:
        out["year"] = out[cols.album_date].map(_year)
    out["quality_tags"] = out[cols.quality].map(parse_quality_tags) if cols.quality in out \
        else [()] * len(out)
    out[cols.duration_s] = pd.to_numeric(out[cols.duration_s], errors="coerce")
    return out


# ---- generic inspection -----------------------------------------------------------

def schema_report(df: pd.DataFrame, n_examples: int = 3) -> pd.DataFrame:
    """One row per column: dtype, nulls, distinct values and a few example values."""
    rows = []
    for c in df.columns:
        s = df[c]
        non_null = s.dropna()
        examples = non_null.astype(str).unique()[:n_examples]
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "n_null": int(s.isna().sum()),
            "frac_null": round(float(s.isna().mean()), 4),
            "n_unique": int(s.nunique(dropna=True)),
            "examples": " | ".join(e[:60] for e in examples),
        })
    return pd.DataFrame(rows)


def value_counts_report(df: pd.DataFrame, top: int = 15) -> dict[str, pd.Series]:
    """Top-``top`` value counts (including NaN) for every column."""
    return {c: df[c].value_counts(dropna=False).head(top) for c in df.columns}


# ---- quality filtering ------------------------------------------------------------

def quality_tag_counts(df: pd.DataFrame) -> pd.Series:
    """How many performances carry each quality tag ('<none>' for untagged)."""
    tags = df["quality_tags"].map(lambda t: t if t else ("<none>",))
    return tags.explode().value_counts()


def quality_keep_mask(df: pd.DataFrame, exclude_tags: list[str]) -> pd.Series:
    """True for rows that carry none of ``exclude_tags``."""
    excl = {t.lower() for t in exclude_tags}
    return df["quality_tags"].map(lambda tags: not (set(tags) & excl))


def filter_quality(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    return df[quality_keep_mask(df, cfg.quality.exclude_tags)].copy()


# ---- per-pianist statistics and selection -----------------------------------------

def per_pianist_stats(df: pd.DataFrame, cols: ColumnsConfig) -> pd.DataFrame:
    """Performances, distinct works/movements, hours, albums and years per pianist."""
    g = df.groupby(cols.pianist)
    stats = pd.DataFrame({
        "n_perf": g.size(),
        "n_works": g["work_key"].nunique(),
        "n_movements": g["movement_key"].nunique(),
        "n_composers": g[cols.composer].nunique(),
        "hours": g[cols.duration_s].sum() / 3600.0,
        "median_perf_min": g[cols.duration_s].median() / 60.0,
    })
    if cols.album in df:
        stats["n_albums"] = g[cols.album].nunique()
    if "year" in df:
        stats["year_min"] = g["year"].min()
        stats["year_max"] = g["year"].max()
    return stats.sort_values(["n_works", "hours"], ascending=False)


def eligibility_table(stats: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """Number of pianists with >= K distinct works (and movements) for each K."""
    return pd.DataFrame({
        "K": k_values,
        "pianists_with_K_works": [int((stats["n_works"] >= k).sum()) for k in k_values],
        "pianists_with_K_movements": [int((stats["n_movements"] >= k).sum()) for k in k_values],
    })


def select_pianists(stats: pd.DataFrame, n: int, min_compositions: int,
                    group_level: str = "work") -> list[str]:
    """Top-``n`` pianists by number of distinct compositions (ties: more hours first).

    Fails loudly if fewer than ``n`` pianists reach ``min_compositions``.
    """
    count_col = "n_works" if group_level == "work" else "n_movements"
    eligible = stats[stats[count_col] >= min_compositions]
    ranked = eligible.sort_values([count_col, "hours"], ascending=False)
    if len(ranked) < n:
        raise ValueError(f"Only {len(ranked)} pianists have >= {min_compositions} "
                         f"distinct {group_level}s; cannot select {n}.")
    return list(ranked.index[:n])


# ---- structure checks used by the audit -------------------------------------------

def hierarchy_report(df: pd.DataFrame, cols: ColumnsConfig) -> dict[str, object]:
    """How composition_id, work folder and title prefix relate to each other.

    The split grouping depends on this: if one composition_id maps to several work
    folders (or vice versa in unexpected ways), grouping needs care.
    """
    works_per_comp = df.groupby("movement_key")["work_key"].nunique()
    comps_per_work = df.groupby("work_key")["movement_key"].nunique()
    perfs_per_work = df.groupby("work_key").size()
    report: dict[str, object] = {
        "n_composition_ids": int(df["movement_key"].nunique()),
        "n_work_keys": int(df["work_key"].nunique()),
        "n_rows_without_work_key": int(df["work_key"].isna().sum()),
        "composition_ids_in_several_work_folders": int((works_per_comp > 1).sum()),
        "movements_per_work": comps_per_work.describe().round(2).to_dict(),
        "performances_per_work": perfs_per_work.describe().round(1).to_dict(),
        "largest_works_by_performances": perfs_per_work.nlargest(10).to_dict(),
    }
    if "title_work" in df:
        title_per_work = df.groupby("work_key")["title_work"].nunique()
        work_per_title = df.groupby([cols.composer, "title_work"])["work_key"].nunique()
        report["work_folders_with_several_title_prefixes"] = int((title_per_work > 1).sum())
        report["title_prefixes_in_several_work_folders"] = int((work_per_title > 1).sum())
    return report


def near_duplicates(df: pd.DataFrame, cols: ColumnsConfig, tol: float = 0.01) -> pd.DataFrame:
    """Pairs with the same pianist and composition_id whose durations differ < ``tol``.

    These may be the same recording released twice (e.g. on a compilation).
    """
    rows = []
    keep = [cols.perf_id, cols.duration_s] + [c for c in (cols.album, cols.album_date,
                                                          cols.youtube) if c in df]
    for (pianist, comp), g in df.groupby([cols.pianist, "movement_key"]):
        if len(g) < 2:
            continue
        g = g.sort_values(cols.duration_s)
        recs = g[keep].to_dict("records")
        for a, b in zip(recs[:-1], recs[1:]):
            da, db = a[cols.duration_s], b[cols.duration_s]
            if pd.notna(da) and pd.notna(db) and abs(da - db) / max(da, db, 1e-9) < tol:
                rows.append({"pianist": pianist, "composition_id": comp,
                             **{f"a_{k}": v for k, v in a.items()},
                             **{f"b_{k}": v for k, v in b.items()}})
    return pd.DataFrame(rows)


def duplicated_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Rows whose (non-null) value in ``column`` occurs more than once."""
    s = df[column]
    mask = s.notna() & s.duplicated(keep=False)
    return df[mask].sort_values(column)


def composition_overlap(df: pd.DataFrame, cols: ColumnsConfig, pianists: list[str],
                        level: str = "work") -> pd.DataFrame:
    """Matrix [pianist x pianist] of the number of shared works (or movements)."""
    key = GROUP_COLUMNS[level]
    sets = {p: set(df.loc[df[cols.pianist] == p, key].dropna()) for p in pianists}
    return pd.DataFrame([[len(sets[a] & sets[b]) for b in pianists] for a in pianists],
                        index=pianists, columns=pianists)


def album_report(df: pd.DataFrame, cols: ColumnsConfig) -> dict[str, object]:
    """Can recordings/sessions be reconstructed from the album columns?"""
    if cols.album not in df:
        return {"album_column_present": False}
    perfs_per_album = df.groupby(cols.album).size()
    pianists_per_album = df.groupby(cols.album)[cols.pianist].nunique()
    return {
        "album_column_present": True,
        "frac_rows_with_album": round(float(df[cols.album].notna().mean()), 4),
        "n_albums": int(df[cols.album].nunique()),
        "performances_per_album": perfs_per_album.describe().round(1).to_dict(),
        "albums_with_several_pianists": int((pianists_per_album > 1).sum()),
        "frac_rows_with_year": round(float(df["year"].notna().mean()), 4) if "year" in df else None,
        "year_range": [df["year"].min(), df["year"].max()] if "year" in df else None,
    }
