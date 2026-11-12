"""API FastAPI : prédiction, santé, métriques pour le tableau de bord."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from afrorisk import __version__
from afrorisk.api.schemas import BorrowerInput
from afrorisk.api.service import Predictor, metrics_summary
from afrorisk.schema import (
    CONTRACT_TYPES,
    EDUCATION_TYPES,
    FORM_FAMILY_STATUSES,
    FORM_INCOME_TYPES,
)


def _origins() -> list[str]:
    raw = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080",
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


def get_predictor():
    if getattr(app.state, "predictor_checked", False):
        return getattr(app.state, "predictor", None)
    try:
        app.state.predictor = Predictor.load()
    except FileNotFoundError:
        app.state.predictor = None
    app.state.predictor_checked = True
    return app.state.predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_predictor()
    yield


app = FastAPI(
    title="AfroRisk AI",
    version=__version__,
    description=(
        "Estimation du risque de défaut et facteurs explicatifs. "
        "Outil de démonstration : le score ne constitue pas une décision d'octroi."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "AfroRisk AI", "version": __version__, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    predictor = get_predictor()
    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "model_loaded": False},
        )
    return {
        "status": "ok",
        "model_loaded": True,
        "placeholder": bool(predictor.metadata["placeholder"]),
        "data_source": predictor.metadata["data_source"],
        "model": predictor.metadata["model_name"],
    }


@app.get("/options")
def options():
    return {
        "contract_types": CONTRACT_TYPES,
        "education_types": EDUCATION_TYPES,
        "family_statuses": FORM_FAMILY_STATUSES,
        "income_types": FORM_INCOME_TYPES,
        "yes_no": ["Y", "N"],
        "genders": ["F", "M"],
        "region_ratings": [1, 2, 3],
    }


@app.get("/metrics-summary")
def metrics():
    try:
        return metrics_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict")
def predict(payload: BorrowerInput):
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle servi indisponible.")
    return predictor.predict(payload)
