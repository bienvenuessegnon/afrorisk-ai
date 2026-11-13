"""Explications SHAP globales et locales, regroupées par variable d'origine."""

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from afrorisk.labels import label_of


def feature_sources(pipeline, numeric: list[str], categorical: list[str]) -> list[str]:
    encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    sources = list(numeric)
    width = len(numeric)
    for column, levels in zip(categorical, encoder.categories_):
        sources.extend([column] * len(levels))
        width += len(levels)
    if len(sources) != width:
        raise RuntimeError(f"Décalage SHAP : {len(sources)} sources pour {width} colonnes.")
    return sources


def _positive_shap_values(values) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    return array


def shap_matrix(pipeline, frame: pd.DataFrame, background: np.ndarray | None = None) -> np.ndarray:
    model = pipeline.named_steps["model"]
    transformed = _transform(pipeline, frame)
    if isinstance(model, LogisticRegression):
        if background is None:
            raise ValueError("Le modèle linéaire a besoin d'un jeu de fond pour SHAP.")
        explainer = shap.LinearExplainer(model, background)
        values = explainer.shap_values(transformed)
    else:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
    return _positive_shap_values(values)


def _transform(pipeline, frame: pd.DataFrame) -> np.ndarray:
    clipped = pipeline.named_steps["clip"].transform(frame)
    return np.asarray(pipeline.named_steps["preprocess"].transform(clipped))


def background_matrix(pipeline, frame: pd.DataFrame, limit: int = 200) -> np.ndarray:
    sample = frame.sample(n=min(limit, len(frame)), random_state=42)
    return _transform(pipeline, sample)


def group_mean_abs(values: np.ndarray, sources: list[str]) -> list[dict]:
    totals: dict[str, float] = {}
    mean_abs = np.abs(values).mean(axis=0)
    for source, magnitude in zip(sources, mean_abs):
        totals[source] = totals.get(source, 0.0) + float(magnitude)
    ordered = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "feature": feature,
            "label": label_of(feature),
            "mean_abs_shap": round(magnitude, 6),
        }
        for feature, magnitude in ordered
    ]


def local_factors(
    pipeline,
    frame: pd.DataFrame,
    sources: list[str],
    background: np.ndarray | None,
    top_k: int = 6,
) -> list[dict]:
    values = shap_matrix(pipeline, frame, background=background)[0]
    totals: dict[str, float] = {}
    for source, contribution in zip(sources, values):
        totals[source] = totals.get(source, 0.0) + float(contribution)
    ordered = sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
    row = frame.iloc[0]
    factors = []
    for feature, contribution in ordered:
        raw_value = row[feature] if feature in row.index else None
        if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
            display = None
        elif isinstance(raw_value, (float, np.floating)):
            display = round(float(raw_value), 4)
        else:
            display = raw_value if isinstance(raw_value, str) else str(raw_value)
        factors.append(
            {
                "feature": feature,
                "label": label_of(feature),
                "value": display,
                "contribution": round(float(contribution), 6),
                "direction": "augmente" if contribution > 0 else "réduit",
            }
        )
    return factors
