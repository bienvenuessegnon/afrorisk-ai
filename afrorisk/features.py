"""Construction des variables, agrégations et garde-fous anti-fuite.

Les tables secondaires décrivent l'historique connu à la date de la demande
(décalages DAYS_* et MONTHS_BALANCE relatifs à cette date dans Home Credit).
La cible TARGET n'entre dans aucune agrégation.
"""

import numpy as np
import pandas as pd

from afrorisk.config import (
    EMPLOYMENT_SENTINEL,
    FULL_EXTRA_CATEGORICAL,
    FULL_EXTRA_NUMERIC,
    SERVED_CATEGORICAL,
    SERVED_NUMERIC,
)
from afrorisk.schema import CATEGORY_LEVELS

COUNT_LIKE = (
    "BUREAU_COUNT",
    "BUREAU_ACTIVE_COUNT",
    "BB_MONTHS_COUNT",
    "PREV_COUNT",
    "PREV_APPROVED_COUNT",
    "PREV_REFUSED_COUNT",
    "POS_COUNT",
    "INST_COUNT",
    "CC_COUNT",
)


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["AGE_YEARS"] = (-pd.to_numeric(out["DAYS_BIRTH"], errors="coerce")) / 365.25
    employed = pd.to_numeric(out["DAYS_EMPLOYED"], errors="coerce")
    employed = employed.mask(employed == EMPLOYMENT_SENTINEL, np.nan)
    out["EMPLOYMENT_YEARS"] = (-employed) / 365.25

    income = pd.to_numeric(out["AMT_INCOME_TOTAL"], errors="coerce").replace(0, np.nan)
    credit = pd.to_numeric(out["AMT_CREDIT"], errors="coerce").replace(0, np.nan)
    out["CREDIT_INCOME_RATIO"] = pd.to_numeric(out["AMT_CREDIT"], errors="coerce") / income
    out["ANNUITY_INCOME_RATIO"] = pd.to_numeric(out["AMT_ANNUITY"], errors="coerce") / income
    out["GOODS_CREDIT_RATIO"] = pd.to_numeric(out["AMT_GOODS_PRICE"], errors="coerce") / credit

    ext = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].apply(pd.to_numeric, errors="coerce")
    out["EXT_SOURCE_MEAN"] = ext.mean(axis=1, skipna=True)

    if "CODE_GENDER" in out.columns:
        gender = out["CODE_GENDER"].astype("object")
        gender = gender.mask(gender.isin(["XNA", ""]), np.nan)
        out["CODE_GENDER"] = gender.fillna("Non renseigné")
    if "OCCUPATION_TYPE" in out.columns:
        out["OCCUPATION_TYPE"] = out["OCCUPATION_TYPE"].fillna("Inconnu")

    out = out.replace([np.inf, -np.inf], np.nan)
    return _keep_known_categories(out)


def _keep_known_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Ramène une modalité inattendue vers un niveau prévu, sans utiliser la cible."""
    fallbacks = {
        "CODE_GENDER": "Non renseigné",
        "OCCUPATION_TYPE": "Inconnu",
        "NAME_FAMILY_STATUS": "Unknown",
    }
    for column, levels in CATEGORY_LEVELS.items():
        if column not in df.columns:
            continue
        allowed = set(levels)
        unknown = df[column].notna() & ~df[column].isin(allowed)
        if unknown.any():
            df.loc[unknown, column] = fallbacks.get(column, levels[0])
    return df


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    df = bureau.copy()
    df["IS_ACTIVE"] = (df["CREDIT_ACTIVE"].astype(str) == "Active").astype(int)
    grouped = df.groupby("SK_ID_CURR")
    return pd.DataFrame(
        {
            "BUREAU_COUNT": grouped.size(),
            "BUREAU_ACTIVE_COUNT": grouped["IS_ACTIVE"].sum(),
            "BUREAU_AMT_CREDIT_SUM_MEAN": grouped["AMT_CREDIT_SUM"].mean(),
            "BUREAU_AMT_CREDIT_SUM_SUM": grouped["AMT_CREDIT_SUM"].sum(),
            "BUREAU_AMT_CREDIT_DEBT_MEAN": grouped["AMT_CREDIT_SUM_DEBT"].mean(),
            "BUREAU_AMT_OVERDUE_MEAN": grouped["AMT_CREDIT_SUM_OVERDUE"].mean(),
            "BUREAU_DAYS_CREDIT_MIN": grouped["DAYS_CREDIT"].min(),
            "BUREAU_CREDIT_DAY_OVERDUE_MAX": grouped["CREDIT_DAY_OVERDUE"].max(),
        }
    )


def aggregate_bureau_balance(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    status = bureau_balance["STATUS"].astype(str)
    frame = pd.DataFrame(
        {
            "SK_ID_BUREAU": bureau_balance["SK_ID_BUREAU"].values,
            "STATUS_NUM": pd.to_numeric(status, errors="coerce").values,
            "LATE": status.isin(list("12345")).astype(int).values,
        }
    )
    link = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates()
    frame = frame.merge(link, on="SK_ID_BUREAU", how="inner")
    grouped = frame.groupby("SK_ID_CURR")
    return pd.DataFrame(
        {
            "BB_MONTHS_COUNT": grouped.size(),
            "BB_STATUS_MEAN": grouped["STATUS_NUM"].mean(),
            "BB_LATE_SHARE": grouped["LATE"].mean(),
        }
    )


def aggregate_previous(previous: pd.DataFrame) -> pd.DataFrame:
    df = previous.copy()
    df["APPROVED"] = (df["NAME_CONTRACT_STATUS"].astype(str) == "Approved").astype(int)
    df["REFUSED"] = (df["NAME_CONTRACT_STATUS"].astype(str) == "Refused").astype(int)
    asked = pd.to_numeric(df["AMT_APPLICATION"], errors="coerce")
    granted = pd.to_numeric(df["AMT_CREDIT"], errors="coerce").replace(0, np.nan)
    df["APP_CREDIT_RATIO"] = asked / granted
    grouped = df.groupby("SK_ID_CURR")
    return pd.DataFrame(
        {
            "PREV_COUNT": grouped.size(),
            "PREV_APPROVED_COUNT": grouped["APPROVED"].sum(),
            "PREV_REFUSED_COUNT": grouped["REFUSED"].sum(),
            "PREV_AMT_CREDIT_MEAN": grouped["AMT_CREDIT"].mean(),
            "PREV_DAYS_DECISION_MIN": grouped["DAYS_DECISION"].min(),
            "PREV_APP_CREDIT_RATIO_MEAN": grouped["APP_CREDIT_RATIO"].mean(),
        }
    )


def aggregate_pos(pos: pd.DataFrame) -> pd.DataFrame:
    grouped = pos.groupby("SK_ID_CURR")
    return pd.DataFrame(
        {
            "POS_COUNT": grouped.size(),
            "POS_SK_DPD_MEAN": grouped["SK_DPD"].mean(),
            "POS_SK_DPD_MAX": grouped["SK_DPD"].max(),
            "POS_INSTALMENT_FUTURE_MEAN": grouped["CNT_INSTALMENT_FUTURE"].mean(),
        }
    )


def aggregate_installments(installments: pd.DataFrame) -> pd.DataFrame:
    df = installments.copy()
    due = pd.to_numeric(df["AMT_INSTALMENT"], errors="coerce").replace(0, np.nan)
    paid = pd.to_numeric(df["AMT_PAYMENT"], errors="coerce")
    df["PAYMENT_RATIO"] = paid / due
    # Un écart positif veut dire un paiement postérieur à l'échéance.
    df["LATE_DAYS"] = pd.to_numeric(df["DAYS_ENTRY_PAYMENT"], errors="coerce") - pd.to_numeric(
        df["DAYS_INSTALMENT"], errors="coerce"
    )
    df["IS_LATE"] = (df["LATE_DAYS"] > 0).astype(float)
    grouped = df.groupby("SK_ID_CURR")
    return pd.DataFrame(
        {
            "INST_COUNT": grouped.size(),
            "INST_PAYMENT_RATIO_MEAN": grouped["PAYMENT_RATIO"].mean(),
            "INST_LATE_DAYS_MEAN": grouped["LATE_DAYS"].mean(),
            "INST_LATE_SHARE": grouped["IS_LATE"].mean(),
        }
    )


def aggregate_credit_card(credit_card: pd.DataFrame) -> pd.DataFrame:
    df = credit_card.copy()
    limit = pd.to_numeric(df["AMT_CREDIT_LIMIT_ACTUAL"], errors="coerce").replace(0, np.nan)
    df["UTILIZATION"] = pd.to_numeric(df["AMT_BALANCE"], errors="coerce") / limit
    grouped = df.groupby("SK_ID_CURR")
    return pd.DataFrame(
        {
            "CC_COUNT": grouped.size(),
            "CC_AMT_BALANCE_MEAN": grouped["AMT_BALANCE"].mean(),
            "CC_UTILIZATION_MEAN": grouped["UTILIZATION"].mean(),
            "CC_SK_DPD_MAX": grouped["SK_DPD"].max(),
        }
    )


def build_modeling_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    application = tables["application"].copy()
    if application["SK_ID_CURR"].duplicated().any():
        application = application.drop_duplicates("SK_ID_CURR", keep="first")

    frame = add_application_features(application)
    aggregates = [
        aggregate_bureau(tables["bureau"]),
        aggregate_bureau_balance(tables["bureau"], tables["bureau_balance"]),
        aggregate_previous(tables["previous_application"]),
        aggregate_pos(tables["POS_CASH_balance"]),
        aggregate_installments(tables["installments_payments"]),
        aggregate_credit_card(tables["credit_card_balance"]),
    ]
    for aggregate in aggregates:
        frame = frame.merge(aggregate, left_on="SK_ID_CURR", right_index=True, how="left")

    for column in COUNT_LIKE:
        if column in frame.columns:
            frame[column] = frame[column].fillna(0.0)
    return frame.reset_index(drop=True)


def served_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = SERVED_NUMERIC + SERVED_CATEGORICAL
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Colonnes servies absentes : {missing}")
    return frame[columns].copy()


def research_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = SERVED_NUMERIC + FULL_EXTRA_NUMERIC + SERVED_CATEGORICAL + FULL_EXTRA_CATEGORICAL
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Colonnes de recherche absentes : {missing}")
    return frame[columns].copy()


def category_list(columns: list[str]) -> list[list[str]]:
    return [CATEGORY_LEVELS[column] for column in columns]
