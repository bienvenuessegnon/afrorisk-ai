"""Comparaison des modèles et pipeline sklearn sans fuite entre les plis."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from afrorisk.config import (
    CLIP_COLUMNS,
    COST_FN,
    COST_FP,
    MODEL_NAMES,
    RANDOM_STATE,
    TARGET_ENCODE_COLUMNS,
)
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


class TargetEncodeAppender(BaseEstimator, TransformerMixin):
    """Ajoute des encodages de cible lissés, calculés uniquement sur l'échantillon d'ajustement."""

    def __init__(self, columns: list[str] | None = None, smoothing: float = 20.0):
        self.columns = columns if columns is not None else list(TARGET_ENCODE_COLUMNS)
        self.smoothing = smoothing

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TargetEncodeAppender requiert la cible.")
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        target = np.asarray(y, dtype=float)
        self.global_mean_ = float(np.mean(target))
        self.maps_ = {}
        self.feature_names_ = [f"{column}_TE" for column in self.columns if column in frame.columns]
        for column in self.columns:
            if column not in frame.columns:
                continue
            stats = pd.DataFrame({"x": frame[column], "y": target}).groupby("x")["y"].agg(["mean", "count"])
            encoded = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (
                stats["count"] + self.smoothing
            )
            self.maps_[column] = encoded.astype(float)
        return self

    def transform(self, X):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for column in self.columns:
            name = f"{column}_TE"
            if column in self.maps_:
                frame[name] = pd.to_numeric(frame[column].map(self.maps_[column]), errors="coerce")
                frame[name] = frame[name].fillna(self.global_mean_)
            elif column in frame.columns or name not in frame.columns:
                frame[name] = self.global_mean_
        return frame


class ProbabilityAverager(BaseEstimator):
    """Moyenne des probabilités de plusieurs pipelines déjà ajustés."""

    def __init__(self, pipelines, weights=None):
        self.pipelines = list(pipelines)
        self.weights = None if weights is None else list(weights)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        weights = self.weights or [1.0] * len(self.pipelines)
        total = float(sum(weights))
        acc = None
        for pipeline, weight in zip(self.pipelines, weights):
            part = pipeline.predict_proba(X) * (weight / total)
            acc = part if acc is None else acc + part
        return acc

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_estimator(name: str, y: pd.Series, fast: bool, params: dict | None = None):
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    scale_pos_weight = negatives / positives
    extra = dict(params or {})
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
            n_estimators=40 if fast else 100,
            max_depth=6 if fast else 10,
            min_samples_leaf=5 if fast else 20,
            class_weight="balanced_subsample",
            n_jobs=4,
            random_state=RANDOM_STATE,
        )
    if name == "xgboost":
        defaults = {
            "n_estimators": 60 if fast else 2500,
            "max_depth": 3 if fast else 5,
            "learning_rate": 0.1 if fast else 0.04,
            "subsample": 0.85,
            "colsample_bytree": 0.7,
            "min_child_weight": 4.0,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        if not fast:
            defaults["early_stopping_rounds"] = 50
        defaults.update(extra)
        return XGBClassifier(**defaults)
    if name == "lightgbm":
        defaults = {
            "n_estimators": 60 if fast else 3000,
            "learning_rate": 0.1 if fast else 0.03,
            "num_leaves": 15 if fast else 40,
            "max_depth": 4 if fast else -1,
            "min_child_samples": 20 if fast else 80,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        defaults.update(extra)
        return LGBMClassifier(**defaults)
    raise KeyError(name)


def _target_columns(frame: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if not requested:
        return []
    return [column for column in requested if column in frame.columns]


def make_pipeline(
    estimator,
    numeric: list[str],
    categorical: list[str],
    target_encode: list[str] | None = None,
) -> Pipeline:
    encode_cols = list(target_encode or [])
    te_names = [f"{column}_TE" for column in encode_cols]
    numeric_with_te = list(numeric) + [name for name in te_names if name not in numeric]
    steps: list[tuple] = [("clip", ColumnClipper())]
    if encode_cols:
        steps.append(("target_encode", TargetEncodeAppender(columns=encode_cols)))
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
                numeric_with_te,
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
    steps.extend([("preprocess", preprocess), ("model", estimator)])
    return Pipeline(steps=steps)


def _uses_early_stopping(name: str, fast: bool, n_rows: int) -> bool:
    return (not fast) and name in {"lightgbm", "xgboost"} and n_rows >= 800


def _split_inner(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def _fit_with_optional_stopping(pipeline: Pipeline, X, y, name: str, fast: bool) -> int | None:
    if not _uses_early_stopping(name, fast, len(X)):
        pipeline.fit(X, y)
        model = pipeline.named_steps["model"]
        best = getattr(model, "best_iteration_", None) or getattr(model, "best_iteration", None)
        return int(best) if best else None

    X_tr, X_va, y_tr, y_va = _split_inner(X, y)
    prepared = [step for step in pipeline.steps if step[0] != "model"]
    prefix = Pipeline(prepared)
    prefix.fit(X_tr, y_tr)
    X_tr_t = prefix.transform(X_tr)
    X_va_t = prefix.transform(X_va)
    # Recopie les étapes déjà ajustées dans le pipeline final.
    for step_name, transformer in prefix.steps:
        pipeline.named_steps[step_name] = transformer

    model = pipeline.named_steps["model"]
    if name == "lightgbm":
        model.fit(
            X_tr_t,
            y_tr,
            eval_X=X_va_t,
            eval_y=y_va,
            eval_metric="auc",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        best = getattr(model, "best_iteration_", None)
        return int(best) if best is not None else int(model.n_estimators)
    model.fit(X_tr_t, y_tr, eval_set=[(X_va_t, y_va)], verbose=False)
    best = getattr(model, "best_iteration", None)
    if best is None:
        best = getattr(model, "best_iteration_", None)
    return int(best) if best not in (None, 0) else None


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


def _pack_oof(name: str, y_array: np.ndarray, oof: np.ndarray, best_iterations: list[int]) -> dict:
    curve, threshold = threshold_study(y_array, oof)
    metrics = _scores_at_threshold(y_array, oof, threshold)
    metrics_at_half = _scores_at_threshold(y_array, oof, 0.5)
    mean_best = int(round(float(np.mean(best_iterations)))) if best_iterations else None
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
        "best_iteration": mean_best,
    }


def cross_validate(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    numeric: list[str],
    categorical: list[str],
    n_splits: int,
    fast: bool,
    params: dict | None = None,
    target_encode: list[str] | None = None,
    random_state: int | None = None,
) -> dict:
    seed = RANDOM_STATE if random_state is None else random_state
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    y_array = y.to_numpy()
    best_iterations: list[int] = []
    encode_cols = _target_columns(X, target_encode)
    for train_index, test_index in splitter.split(X, y):
        fold_params = dict(params or {})
        if random_state is not None and name in {"lightgbm", "xgboost"}:
            fold_params["random_state"] = random_state
        estimator = make_estimator(name, y.iloc[train_index], fast=fast, params=fold_params or None)
        pipeline = make_pipeline(estimator, numeric, categorical, target_encode=encode_cols)
        best = _fit_with_optional_stopping(
            pipeline, X.iloc[train_index], y.iloc[train_index], name, fast
        )
        if best:
            best_iterations.append(best)
        oof[test_index] = pipeline.predict_proba(X.iloc[test_index])[:, 1]
    return _pack_oof(name, y_array, oof, best_iterations)


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    numeric: list[str],
    categorical: list[str],
    n_splits: int,
    fast: bool,
    param_overrides: dict[str, dict] | None = None,
    target_encode: list[str] | None = None,
) -> list[dict]:
    results = []
    overrides = param_overrides or {}
    for name in MODEL_NAMES:
        print(f"  validation croisée : {name}")
        results.append(
            cross_validate(
                name,
                X,
                y,
                numeric,
                categorical,
                n_splits,
                fast,
                params=overrides.get(name),
                target_encode=target_encode,
            )
        )
    results.sort(key=lambda item: item["roc_auc"], reverse=True)
    return results


def fit_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    numeric: list[str],
    categorical: list[str],
    fast: bool,
    params: dict | None = None,
    target_encode: list[str] | None = None,
    best_iteration: int | None = None,
) -> Pipeline:
    fit_params = dict(params or {})
    if best_iteration and name in {"lightgbm", "xgboost"}:
        fit_params["n_estimators"] = max(int(best_iteration * 1.05), 50)
        fit_params.pop("early_stopping_rounds", None)
        # Ajustement sur tout l'échantillon, sans arrêt anticipé.
        local_fast = True if name == "xgboost" else fast
        estimator = make_estimator(name, y, fast=local_fast, params=fit_params)
        if name == "lightgbm":
            estimator.set_params(n_estimators=fit_params["n_estimators"])
        pipeline = make_pipeline(
            estimator, numeric, categorical, target_encode=_target_columns(X, target_encode)
        )
        pipeline.fit(X, y)
        return pipeline
    estimator = make_estimator(name, y, fast=fast, params=fit_params or None)
    if name == "xgboost":
        # Pas de jeu de validation ici : on retire l'arrêt anticipé.
        try:
            estimator.set_params(early_stopping_rounds=None)
        except ValueError:
            pass
    pipeline = make_pipeline(
        estimator, numeric, categorical, target_encode=_target_columns(X, target_encode)
    )
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


def _holdout_auc(name: str, X, y, numeric, categorical, fast, params, target_encode) -> float:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    estimator = make_estimator(name, y_tr, fast=fast, params=params)
    pipeline = make_pipeline(estimator, numeric, categorical, target_encode=_target_columns(X, target_encode))
    _fit_with_optional_stopping(pipeline, X_tr, y_tr, name, fast)
    proba = pipeline.predict_proba(X_va)[:, 1]
    return float(roc_auc_score(y_va, proba))


def tune_boosting(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    numeric: list[str],
    categorical: list[str],
    fast: bool,
    n_trials: int,
    timeout: int | None,
    target_encode: list[str] | None = None,
) -> dict:
    """Recherche d'hyperparamètres sur un hold-out stratifié, sans toucher aux plis finaux."""
    if fast or n_trials <= 0:
        return {}
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Optuna n'est pas installé : hyperparamètres par défaut.")
        return {}

    def objective(trial):
        if name == "lightgbm":
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 16, 64),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 8.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 4.0),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.4),
                "n_estimators": 3000,
            }
        else:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 4.0),
                "gamma": trial.suggest_float("gamma", 0.0, 3.0),
                "n_estimators": 2500,
                "early_stopping_rounds": 50,
            }
        return _holdout_auc(name, X, y, numeric, categorical, fast, params, target_encode)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    print(f"  réglage Optuna ({name}, {n_trials} essais, timeout={timeout}s)")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    print(f"  meilleur hold-out {name} : AUC {study.best_value:.4f}")
    return dict(study.best_params)


def optuna_enabled() -> bool:
    if os.environ.get("AFRORISK_FAST") == "1":
        return False
    return os.environ.get("AFRORISK_OPTUNA", "1") != "0"


def optuna_trials(default: int) -> int:
    raw = os.environ.get("AFRORISK_OPTUNA_TRIALS")
    return int(raw) if raw else default


def blend_oof(name: str, parts: list[np.ndarray], y: np.ndarray, weights: list[float] | None = None) -> dict:
    if weights is None:
        weights = [1.0] * len(parts)
    total = float(sum(weights))
    oof = sum(part * (weight / total) for part, weight in zip(parts, weights))
    return _pack_oof(name, y, np.asarray(oof, dtype=float), [])
