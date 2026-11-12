"""Comparaison des modèles et pipeline sklearn sans fuite entre les plis."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from afrorisk.config import CLIP_COLUMNS, COST_FN, COST_FP, MODEL_NAMES, RANDOM_STATE
from afrorisk.features import category_list


class ColumnClipper(BaseEstimator, TransformerMixin):
    """Borne quelques montants aux percentiles appris sur le pli d'entraînement."""

    def __init__(self, columns: list[str] | None = None, lower: float = 0.5, upper: float = 99.5):
        self.columns = columns if columns is not None else list(CLIP_COLUMNS)
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.bounds_ = {}
        for column in self.columns:
            if column not in frame.columns:
                continue
            series = pd.to_numeric(frame[column], errors="coerce")
            if int(series.notna().sum()) < 20:
                continue
            low, high = np.nanpercentile(series.to_numpy(dtype=float), [self.lower, self.upper])
            if np.isfinite(low) and np.isfinite(high) and low < high:
                self.bounds_[column] = (float(low), float(high))
        return self

    def transform(self, X):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for column, (low, high) in self.bounds_.items():
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(low, high)
        return frame


def make_estimator(name: str, y: pd.Series, fast: bool):
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    scale_pos_weight = negatives / positives
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=5 if fast else 8,
            min_samples_leaf=8 if fast else 40,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=40 if fast else 120,
            max_depth=6 if fast else 12,
            min_samples_leaf=5 if fast else 20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=40 if fast else 160,
            max_depth=3 if fast else 5,
            learning_rate=0.1 if fast else 0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=40 if fast else 250,
            learning_rate=0.1 if fast else 0.05,
            num_leaves=15 if fast else 31,
            max_depth=4 if fast else -1,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    raise KeyError(name)


def make_pipeline(estimator, numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                categories=category_list(categorical),
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("clip", ColumnClipper()),
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )


def _scores_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    predicted = (proba >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "threshold": float(threshold),
    }


def threshold_study(y_true: np.ndarray, proba: np.ndarray) -> tuple[list[dict], float]:
    rows = []
    best: dict | None = None
    for threshold in np.round(np.linspace(0.05, 0.95, 19), 2):
        predicted = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        row = {
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "cost": float(COST_FN * fn + COST_FP * fp),
        }
        rows.append(row)
        if best is None or row["cost"] < best["cost"] or (
            row["cost"] == best["cost"] and row["f1"] > best["f1"]
        ):
            best = row
    assert best is not None
    return rows, float(best["threshold"])


def cross_validate(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    numeric: list[str],
    categorical: list[str],
    n_splits: int,
    fast: bool,
) -> dict:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)
    y_array = y.to_numpy()
    for train_index, test_index in splitter.split(X, y):
        estimator = make_estimator(name, y.iloc[train_index], fast=fast)
        pipeline = make_pipeline(estimator, numeric, categorical)
        pipeline.fit(X.iloc[train_index], y.iloc[train_index])
        oof[test_index] = pipeline.predict_proba(X.iloc[test_index])[:, 1]

    curve, threshold = threshold_study(y_array, oof)
    metrics = _scores_at_threshold(y_array, oof, threshold)
    metrics_at_half = _scores_at_threshold(y_array, oof, 0.5)
    return {
        "model": name,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "threshold": threshold,
        "confusion_matrix": metrics["confusion_matrix"],
        "metrics_at_05": metrics_at_half,
        "threshold_curve": curve,
        "oof_proba": oof,
    }


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    numeric: list[str],
    categorical: list[str],
    n_splits: int,
    fast: bool,
) -> list[dict]:
    results = []
    for name in MODEL_NAMES:
        print(f"  validation croisée : {name}")
        results.append(cross_validate(name, X, y, numeric, categorical, n_splits, fast))
    results.sort(key=lambda item: item["roc_auc"], reverse=True)
    return results


def fit_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    numeric: list[str],
    categorical: list[str],
    fast: bool,
) -> Pipeline:
    pipeline = make_pipeline(make_estimator(name, y, fast=fast), numeric, categorical)
    pipeline.fit(X, y)
    return pipeline


def public_comparison(results: list[dict]) -> list[dict]:
    exported = []
    for item in results:
        exported.append(
            {
                "model": item["model"],
                "roc_auc": item["roc_auc"],
                "pr_auc": item["pr_auc"],
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1"],
                "threshold": item["threshold"],
                "confusion_matrix": item["confusion_matrix"],
            }
        )
    return exported
