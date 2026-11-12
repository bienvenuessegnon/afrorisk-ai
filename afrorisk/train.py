"""Entraîne le modèle de recherche et le modèle servi, puis met à jour le rapport.

Usage : python -m afrorisk.train
"""

import json
import os
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd

from afrorisk.config import (
    COST_FP,
    COST_FN,
    FULL_EXTRA_CATEGORICAL,
    FULL_EXTRA_NUMERIC,
    METRICS_PATH,
    MODELS_RESEARCH,
    MODELS_SERVED,
    RANDOM_STATE,
    ROOT,
    SERVED_CATEGORICAL,
    SERVED_NUMERIC,
)
from afrorisk.eda import save_figures
from afrorisk.explain import background_matrix, feature_sources, group_mean_abs, shap_matrix
from afrorisk.features import build_modeling_table, research_feature_frame, served_feature_frame
from afrorisk.io import load_tables
from afrorisk.modeling import compare_models, fit_model, public_comparison
from afrorisk.report import update_readme, write_metrics


def _limit_rows(frame: pd.DataFrame) -> pd.DataFrame:
    raw = os.environ.get("AFRORISK_MAX_ROWS")
    if not raw:
        return frame
    limit = int(raw)
    if len(frame) <= limit:
        return frame
    shares = frame["TARGET"].value_counts(normalize=True)
    parts = []
    for label, group in frame.groupby("TARGET"):
        take = max(1, int(round(limit * float(shares.loc[label]))))
        parts.append(group.sample(n=min(len(group), take), random_state=RANDOM_STATE))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)


def _selected(results: list[dict]) -> dict:
    return max(results, key=lambda item: item["roc_auc"])


def _fit_scope(name, X, y, numeric, categorical, fast, destination):
    pipeline = fit_model(name, X, y, numeric, categorical, fast=fast)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination, compress=3)
    return pipeline


def _maybe_shrink(pipeline, X, y, numeric, categorical, fast, path):
    size = path.stat().st_size
    if size <= 12 * 1024 * 1024:
        return pipeline, size, False
    print("L'artefact dépasse 12 Mo : le modèle servi retenu est la régression logistique.")
    pipeline = _fit_scope("logistic_regression", X, y, numeric, categorical, fast, path)
    return pipeline, path.stat().st_size, True


def train() -> dict:
    tables, source = load_tables()
    print(f"Source des données : {source}")
    frame = _limit_rows(build_modeling_table(tables))
    if frame["TARGET"].nunique() < 2:
        raise RuntimeError("La cible ne contient qu'une classe : entraînement impossible.")

    y = frame["TARGET"].astype(int)
    X_served = served_feature_frame(frame)
    X_research = research_feature_frame(frame)
    fast = len(frame) < 8_000 or os.environ.get("AFRORISK_FAST") == "1"
    default_folds = 5 if len(frame) <= 20_000 else 3
    n_splits = int(os.environ.get("AFRORISK_CV_FOLDS", default_folds))
    placeholder = source != "home_credit"

    print(f"{len(frame)} demandes, taux de défaut {y.mean():.3f}, fast={fast}, plis={n_splits}")
    print("Modèle de recherche")
    research_numeric = SERVED_NUMERIC + FULL_EXTRA_NUMERIC
    research_categorical = SERVED_CATEGORICAL + FULL_EXTRA_CATEGORICAL
    research_results = compare_models(
        X_research, y, research_numeric, research_categorical, n_splits, fast
    )
    print("Modèle servi")
    served_results = compare_models(
        X_served, y, SERVED_NUMERIC, SERVED_CATEGORICAL, n_splits, fast
    )

    best_research = _selected(research_results)
    best_served = _selected(served_results)
    print(
        f"Recherche : {best_research['model']} (ROC-AUC {best_research['roc_auc']:.3f}) | "
        f"servi : {best_served['model']} (ROC-AUC {best_served['roc_auc']:.3f})"
    )

    MODELS_RESEARCH.mkdir(parents=True, exist_ok=True)
    MODELS_SERVED.mkdir(parents=True, exist_ok=True)
    _fit_scope(
        best_research["model"],
        X_research,
        y,
        research_numeric,
        research_categorical,
        fast,
        MODELS_RESEARCH / "model.joblib",
    )
    served_pipeline = _fit_scope(
        best_served["model"],
        X_served,
        y,
        SERVED_NUMERIC,
        SERVED_CATEGORICAL,
        fast,
        MODELS_SERVED / "model.joblib",
    )
    served_pipeline, artifact_bytes, shrunk = _maybe_shrink(
        served_pipeline,
        X_served,
        y,
        SERVED_NUMERIC,
        SERVED_CATEGORICAL,
        fast,
        MODELS_SERVED / "model.joblib",
    )
    served_name = "logistic_regression" if shrunk else best_served["model"]
    served_metrics_row = next(item for item in served_results if item["model"] == served_name)

    sample = X_served.sample(n=min(250, len(X_served)), random_state=RANDOM_STATE)
    background = background_matrix(served_pipeline, X_served, limit=200)
    np.save(MODELS_SERVED / "background.npy", background)
    sources = feature_sources(served_pipeline, SERVED_NUMERIC, SERVED_CATEGORICAL)
    values = shap_matrix(served_pipeline, sample, background=background)
    if values.shape[1] != len(sources):
        raise RuntimeError(
            f"SHAP a renvoyé {values.shape[1]} colonnes, le pipeline en attend {len(sources)}."
        )
    global_factors = group_mean_abs(values, sources)[:12]

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metadata = {
        "model_name": served_name,
        "placeholder": placeholder,
        "data_source": source,
        "n_rows": int(len(frame)),
        "trained_at": generated_at,
        "threshold": served_metrics_row["threshold"],
        "numeric_features": SERVED_NUMERIC,
        "categorical_features": SERVED_CATEGORICAL,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "artifact_bytes": int(artifact_bytes),
        "disclaimer": (
            "Modèle entraîné sur une fixture synthétique. Aucune valeur pour une décision de crédit."
            if placeholder
            else "Modèle entraîné sur Home Credit. Aide à la lecture uniquement, pas une décision d'octroi."
        ),
    }
    (MODELS_SERVED / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = {
        "data_source": source,
        "placeholder": placeholder,
        "n_rows": int(len(frame)),
        "positive_rate": float(y.mean()),
        "cv_folds": n_splits,
        "fast_mode": fast,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "generated_at": generated_at,
        "served": {
            "selected_model": served_name,
            "threshold": served_metrics_row["threshold"],
            "metrics": {
                "roc_auc": served_metrics_row["roc_auc"],
                "pr_auc": served_metrics_row["pr_auc"],
                "precision": served_metrics_row["precision"],
                "recall": served_metrics_row["recall"],
                "f1": served_metrics_row["f1"],
                "confusion_matrix": served_metrics_row["confusion_matrix"],
                "threshold": served_metrics_row["threshold"],
            },
            "metrics_at_05": served_metrics_row["metrics_at_05"],
            "comparison": public_comparison(served_results),
            "threshold_curve": served_metrics_row["threshold_curve"],
            "global_factors": global_factors,
        },
        "research": {
            "selected_model": best_research["model"],
            "threshold": best_research["threshold"],
            "metrics": {
                "roc_auc": best_research["roc_auc"],
                "pr_auc": best_research["pr_auc"],
                "precision": best_research["precision"],
                "recall": best_research["recall"],
                "f1": best_research["f1"],
                "confusion_matrix": best_research["confusion_matrix"],
                "threshold": best_research["threshold"],
            },
            "comparison": public_comparison(research_results),
        },
    }
    write_metrics(metrics, METRICS_PATH)
    update_readme(metrics)

    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    mlflow.set_experiment("afrorisk")
    for scope, results in (("research", research_results), ("served", served_results)):
        for item in results:
            with mlflow.start_run(run_name=f"{scope}-{item['model']}"):
                mlflow.log_params(
                    {
                        "scope": scope,
                        "model": item["model"],
                        "data_source": source,
                        "placeholder": placeholder,
                        "fast_mode": fast,
                        "cv_folds": n_splits,
                        "n_rows": int(len(frame)),
                    }
                )
                mlflow.log_metrics(
                    {
                        "roc_auc": item["roc_auc"],
                        "pr_auc": item["pr_auc"],
                        "precision": item["precision"],
                        "recall": item["recall"],
                        "f1": item["f1"],
                        "threshold": item["threshold"],
                    }
                )

    print("Figures EDA")
    save_figures(tables["application"], source)
    print(f"Métriques : {METRICS_PATH}")
    print(f"Modèle servi : {MODELS_SERVED / 'model.joblib'} ({artifact_bytes} octets)")
    return metrics


def main() -> None:
    train()


if __name__ == "__main__":
    main()
