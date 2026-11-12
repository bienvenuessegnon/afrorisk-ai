"""Chargement du modèle servi et calcul d'une prédiction expliquée."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from afrorisk.api.schemas import BorrowerInput
from afrorisk.config import EMPLOYMENT_SENTINEL, METRICS_PATH, MODELS_SERVED
from afrorisk.explain import feature_sources, local_factors
from afrorisk.features import add_application_features, served_feature_frame


def risk_level(probability: float, threshold: float) -> str:
    if probability < threshold / 2:
        return "faible"
    if probability < threshold:
        return "modéré"
    return "élevé"


class Predictor:
    def __init__(self, pipeline, metadata: dict, background: np.ndarray, sources: list[str]):
        self.pipeline = pipeline
        self.metadata = metadata
        self.background = background
        self.sources = sources

    @classmethod
    def load(cls, directory: Path | None = None) -> "Predictor":
        folder = directory or MODELS_SERVED
        model_path = folder / "model.joblib"
        meta_path = folder / "metadata.json"
        background_path = folder / "background.npy"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Modèle servi introuvable dans {folder}. Lancez python -m afrorisk.train.")
        pipeline = joblib.load(model_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        background = np.load(background_path) if background_path.exists() else None
        sources = feature_sources(
            pipeline,
            metadata["numeric_features"],
            metadata["categorical_features"],
        )
        return cls(pipeline, metadata, background, sources)

    def frame_from_input(self, payload: BorrowerInput) -> pd.DataFrame:
        employed = (
            EMPLOYMENT_SENTINEL
            if payload.employment_years is None
            else -float(payload.employment_years) * 365.25
        )
        raw = pd.DataFrame(
            [
                {
                    "AMT_INCOME_TOTAL": payload.amt_income_total,
                    "AMT_CREDIT": payload.amt_credit,
                    "AMT_ANNUITY": payload.amt_annuity,
                    "AMT_GOODS_PRICE": payload.amt_goods_price,
                    "DAYS_BIRTH": -float(payload.age_years) * 365.25,
                    "DAYS_EMPLOYED": employed,
                    "EXT_SOURCE_1": payload.ext_source_1,
                    "EXT_SOURCE_2": payload.ext_source_2,
                    "EXT_SOURCE_3": payload.ext_source_3,
                    "CNT_CHILDREN": payload.cnt_children,
                    "REGION_RATING_CLIENT": payload.region_rating_client,
                    "NAME_CONTRACT_TYPE": payload.name_contract_type,
                    "NAME_EDUCATION_TYPE": payload.name_education_type,
                    "NAME_FAMILY_STATUS": payload.name_family_status,
                    "NAME_INCOME_TYPE": payload.name_income_type,
                    "FLAG_OWN_REALTY": payload.flag_own_realty,
                    "FLAG_OWN_CAR": payload.flag_own_car,
                    "CODE_GENDER": payload.code_gender,
                }
            ]
        )
        return served_feature_frame(add_application_features(raw))

    def predict(self, payload: BorrowerInput) -> dict:
        frame = self.frame_from_input(payload)
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        threshold = float(self.metadata["threshold"])
        predicted = int(probability >= threshold)
        factors = local_factors(
            self.pipeline,
            frame,
            self.sources,
            self.background,
            top_k=6,
        )
        return {
            "probability_default": round(probability, 6),
            "predicted_class": predicted,
            "predicted_label": "défaut" if predicted else "non-défaut",
            "risk_level": risk_level(probability, threshold),
            "threshold": threshold,
            "factors": factors,
            "model": {
                "name": self.metadata["model_name"],
                "placeholder": bool(self.metadata["placeholder"]),
                "data_source": self.metadata["data_source"],
                "disclaimer": self.metadata["disclaimer"],
            },
        }


def metrics_summary() -> dict:
    if not METRICS_PATH.exists():
        raise FileNotFoundError("reports/metrics.json est absent. Lancez python -m afrorisk.train.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
