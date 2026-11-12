"""Chemins, constantes d'entraînement et colonnes du modèle servi."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_FIXTURES = ROOT / "data" / "fixtures"
MODELS_SERVED = ROOT / "models" / "served"
MODELS_RESEARCH = ROOT / "models" / "research"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS_PATH = REPORTS / "metrics.json"
README_PATH = ROOT / "README.md"

EMPLOYMENT_SENTINEL = 365243
RANDOM_STATE = 42
COST_FN = 5.0
COST_FP = 1.0

# Colonnes numériques demandées au formulaire, plus ratios calculés côté serveur.
SERVED_NUMERIC = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCE_MEAN",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "GOODS_CREDIT_RATIO",
    "CNT_CHILDREN",
    "REGION_RATING_CLIENT",
]

SERVED_CATEGORICAL = [
    "NAME_CONTRACT_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "FLAG_OWN_REALTY",
    "FLAG_OWN_CAR",
    "CODE_GENDER",
]

CLIP_COLUMNS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
]

FULL_EXTRA_NUMERIC = [
    "REGION_POPULATION_RELATIVE",
    "CNT_FAM_MEMBERS",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "BUREAU_COUNT",
    "BUREAU_ACTIVE_COUNT",
    "BUREAU_AMT_CREDIT_SUM_MEAN",
    "BUREAU_AMT_CREDIT_SUM_SUM",
    "BUREAU_AMT_CREDIT_DEBT_MEAN",
    "BUREAU_AMT_OVERDUE_MEAN",
    "BUREAU_DAYS_CREDIT_MIN",
    "BUREAU_CREDIT_DAY_OVERDUE_MAX",
    "BB_MONTHS_COUNT",
    "BB_STATUS_MEAN",
    "BB_LATE_SHARE",
    "PREV_COUNT",
    "PREV_APPROVED_COUNT",
    "PREV_REFUSED_COUNT",
    "PREV_AMT_CREDIT_MEAN",
    "PREV_DAYS_DECISION_MIN",
    "PREV_APP_CREDIT_RATIO_MEAN",
    "POS_COUNT",
    "POS_SK_DPD_MEAN",
    "POS_SK_DPD_MAX",
    "POS_INSTALMENT_FUTURE_MEAN",
    "INST_COUNT",
    "INST_PAYMENT_RATIO_MEAN",
    "INST_LATE_DAYS_MEAN",
    "INST_LATE_SHARE",
    "CC_COUNT",
    "CC_AMT_BALANCE_MEAN",
    "CC_UTILIZATION_MEAN",
    "CC_SK_DPD_MAX",
]

FULL_EXTRA_CATEGORICAL = [
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
]

MODEL_LABELS = {
    "logistic_regression": "Régression logistique",
    "decision_tree": "Arbre de décision",
    "random_forest": "Forêt aléatoire",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
