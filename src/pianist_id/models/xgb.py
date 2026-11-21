"""XGBoost on segment features, evaluated per segment and per performance.

A performance's prediction is the class with the highest mean log-probability over its
evaluation segments. Classes are weighted ``balanced`` on train and val, so a pianist
with twice the recordings does not dominate the loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from pianist_id.data.segments import on_eval_grid


@dataclass
class ExperimentResult:
    metrics: dict[str, float]
    classes: list[str]
    perf_predictions: pd.DataFrame            # perf_id, true, pred, one column per class
    confusion: pd.DataFrame                   # performance level, rows = true
    feature_importance: pd.Series             # gain per feature
    model: XGBClassifier = field(repr=False)


def train_xgb(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              params: dict[str, Any], seed: int) -> XGBClassifier:
    model = XGBClassifier(**params, objective="multi:softprob", random_state=seed)
    model.fit(X_train, y_train,
              sample_weight=compute_sample_weight("balanced", y_train),
              eval_set=[(X_val, y_val)],
              sample_weight_eval_set=[compute_sample_weight("balanced", y_val)],
              verbose=False)
    return model


def aggregate_by_performance(proba: np.ndarray, perf_ids: np.ndarray) -> pd.DataFrame:
    """Mean log-probability per performance; rows indexed by perf_id."""
    logp = pd.DataFrame(np.log(np.clip(proba, 1e-9, 1.0)))
    logp["perf_id"] = perf_ids
    return logp.groupby("perf_id").mean()


def gain_importance(model: XGBClassifier, feature_names: list[str]) -> pd.Series:
    """Total gain per feature (0 for features never used in a split)."""
    scores = model.get_booster().get_score(importance_type="total_gain")
    return pd.Series({name: scores.get(f"f{i}", 0.0) for i, name in enumerate(feature_names)})


def group_importance(importance: pd.Series, groups: dict[str, str]) -> pd.Series:
    """Feature importances summed per group, as a share of the total."""
    summed = importance.groupby(importance.index.map(groups)).sum()
    return (summed / summed.sum()).sort_values(ascending=False)


def run_experiment(features: pd.DataFrame, split_df: pd.DataFrame, feature_names: list[str],
                   params: dict[str, Any], seed: int, hop_train_s: float,
                   hop_eval_s: float) -> ExperimentResult:
    """Train on the train split, early-stop on val, evaluate on test.

    ``features`` has one row per segment (``perf_id``, ``start_s``, features); val and
    test keep only the segments on the evaluation hop grid.
    """
    table = features.merge(split_df[["perf_id", "pianist", "split"]], on="perf_id",
                           how="inner")
    classes = sorted(split_df["pianist"].unique())
    label = {c: i for i, c in enumerate(classes)}
    table["y"] = table["pianist"].map(label)
    eval_grid = on_eval_grid(table["start_s"].to_numpy(), hop_eval_s, hop_train_s)
    parts = {
        "train": table[table["split"] == "train"],
        "val": table[(table["split"] == "val") & eval_grid],
        "test": table[(table["split"] == "test") & eval_grid],
    }
    X = {k: v[feature_names].to_numpy(np.float32) for k, v in parts.items()}
    y = {k: v["y"].to_numpy() for k, v in parts.items()}

    model = train_xgb(X["train"], y["train"], X["val"], y["val"], params, seed)
    proba = model.predict_proba(X["test"])

    perf = aggregate_by_performance(proba, parts["test"]["perf_id"].to_numpy())
    perf.columns = classes
    truth = parts["test"].groupby("perf_id")["y"].first().loc[perf.index].to_numpy()
    pred = perf.to_numpy().argmax(axis=1)
    seg_pred = proba.argmax(axis=1)

    metrics = {
        "seg_acc": float(accuracy_score(y["test"], seg_pred)),
        "seg_macro_f1": float(f1_score(y["test"], seg_pred, average="macro")),
        "perf_acc": float(accuracy_score(truth, pred)),
        "perf_macro_f1": float(f1_score(truth, pred, average="macro")),
        "chance": 1.0 / len(classes),
        "best_iteration": int(model.best_iteration),
        **{f"n_seg_{k}": int(len(v)) for k, v in parts.items()},
        "n_perf_test": int(len(perf)),
    }
    perf_predictions = perf.reset_index()
    perf_predictions.insert(1, "true", [classes[i] for i in truth])
    perf_predictions.insert(2, "pred", [classes[i] for i in pred])
    confusion = pd.DataFrame(confusion_matrix(truth, pred, labels=range(len(classes))),
                             index=classes, columns=classes)
    return ExperimentResult(metrics=metrics, classes=classes, perf_predictions=perf_predictions,
                            confusion=confusion,
                            feature_importance=gain_importance(model, feature_names),
                            model=model)


def per_class_f1(result: ExperimentResult) -> pd.Series:
    p = result.perf_predictions
    scores = f1_score(p["true"], p["pred"], labels=result.classes, average=None)
    return pd.Series(scores, index=result.classes)


def summarize_runs(rows: pd.DataFrame, by: list[str],
                   metrics: tuple[str, ...] = ("perf_acc", "perf_macro_f1", "seg_acc")) -> pd.DataFrame:
    """'mean ± std' over seeds for each combination of ``by`` columns."""
    g = rows.groupby(by)[list(metrics)]
    mean, std = g.mean(), g.std()
    return pd.DataFrame({m: [f"{a:.3f} ± {b:.3f}" for a, b in zip(mean[m], std[m])]
                         for m in metrics}, index=mean.index)
