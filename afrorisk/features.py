"""Construction des variables, agrégations et garde-fous anti-fuite.

Les tables secondaires décrivent l'historique connu à la date de la demande
(décalages DAYS_* et MONTHS_BALANCE relatifs à cette date dans Home Credit).
La cible TARGET n'entre dans aucune agrégation.
"""

import gc

import numpy as np
import pandas as pd

from afrorisk.config import (
    EMPLOYMENT_SENTINEL,
    RESEARCH_CATEGORICAL,
    RESEARCH_NUMERIC,
    SERVED_CATEGORICAL,
    SERVED_NUMERIC,
    TARGET_ENCODE_COLUMNS,
)
from afrorisk.schema import CATEGORY_LEVELS

COUNT_PREFIXES = (
    "BUREAU_",
    "BB_",
    "PREV_",
    "POS_",
    "INST_",
    "CC_",
)
COUNT_LIKE_SUFFIXES = (
    "_COUNT",
    "_SUM",
)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _mask_sentinel(series: pd.Series) -> pd.Series:
    return series.mask(series == EMPLOYMENT_SENTINEL, np.nan)


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["AGE_YEARS"] = (-_num(out, "DAYS_BIRTH")) / 365.25
    employed = _mask_sentinel(_num(out, "DAYS_EMPLOYED"))
    out["EMPLOYMENT_YEARS"] = (-employed) / 365.25

    income = _num(out, "AMT_INCOME_TOTAL").replace(0, np.nan)
    credit = _num(out, "AMT_CREDIT").replace(0, np.nan)
    annuity = _num(out, "AMT_ANNUITY")
    goods = _num(out, "AMT_GOODS_PRICE")
    out["CREDIT_INCOME_RATIO"] = _num(out, "AMT_CREDIT") / income
    out["ANNUITY_INCOME_RATIO"] = annuity / income
    out["GOODS_CREDIT_RATIO"] = goods / credit
    out["GOODS_INCOME_RATIO"] = goods / income
    out["PAYMENT_RATE"] = annuity / credit
    out["CREDIT_GOODS_DIFF"] = _num(out, "AMT_CREDIT") - goods

    children = _num(out, "CNT_CHILDREN")
    out["INCOME_PER_CHILD"] = income / (children + 1.0)
    family = _num(out, "CNT_FAM_MEMBERS").replace(0, np.nan)
    out["INCOME_PER_PERSON"] = income / family
    out["CHILDREN_RATIO"] = children / family
    out["EMPLOYMENT_AGE_RATIO"] = out["EMPLOYMENT_YEARS"] / out["AGE_YEARS"].replace(0, np.nan)
    birth_days = _num(out, "DAYS_BIRTH").replace(0, np.nan)
    out["DAYS_EMPLOYED_PERC"] = employed / birth_days

    ext = pd.DataFrame(
        {
            "EXT_SOURCE_1": _num(out, "EXT_SOURCE_1"),
            "EXT_SOURCE_2": _num(out, "EXT_SOURCE_2"),
            "EXT_SOURCE_3": _num(out, "EXT_SOURCE_3"),
        },
        index=out.index,
    )
    out["EXT_SOURCE_MEAN"] = ext.mean(axis=1, skipna=True)
    out["EXT_SOURCE_MIN"] = ext.min(axis=1, skipna=True)
    out["EXT_SOURCE_MAX"] = ext.max(axis=1, skipna=True)
    out["EXT_SOURCE_STD"] = ext.std(axis=1, skipna=True)
    observed = ext.notna().sum(axis=1)
    # Produit des scores observés uniquement ; manquant si aucun score n'est présent.
    log_prod = np.log(ext.clip(lower=1e-6)).sum(axis=1, skipna=True)
    out["EXT_SOURCE_PROD"] = np.exp(log_prod).where(observed > 0, np.nan)
    out["EXT_SOURCE_MISSING"] = (3 - observed).astype(float)
    age = out["AGE_YEARS"]
    emp = out["EMPLOYMENT_YEARS"]
    out["EXT_SOURCE_1_AGE"] = ext["EXT_SOURCE_1"] * age
    out["EXT_SOURCE_2_AGE"] = ext["EXT_SOURCE_2"] * age
    out["EXT_SOURCE_3_AGE"] = ext["EXT_SOURCE_3"] * age
    out["EXT_SOURCE_MEAN_EMPLOYMENT"] = out["EXT_SOURCE_MEAN"] * emp

    phone = _mask_sentinel(_num(out, "DAYS_LAST_PHONE_CHANGE"))
    out["PHONE_TO_BIRTH_RATIO"] = phone / birth_days
    out["PHONE_TO_EMPLOY_RATIO"] = phone / employed.replace(0, np.nan)
    car_age = _num(out, "OWN_CAR_AGE")
    out["CAR_TO_BIRTH_RATIO"] = car_age / out["AGE_YEARS"].replace(0, np.nan)
    out["CAR_TO_EMPLOY_RATIO"] = car_age / out["EMPLOYMENT_YEARS"].replace(0, np.nan)

    if "REGION_RATING_CLIENT_W_CITY" in out.columns:
        out["REGION_RATING_DIFF"] = _num(out, "REGION_RATING_CLIENT") - _num(
            out, "REGION_RATING_CLIENT_W_CITY"
        )
    else:
        out["REGION_RATING_DIFF"] = np.nan

    document_cols = [column for column in out.columns if column.startswith("FLAG_DOCUMENT_")]
    if document_cols:
        docs = out[document_cols].apply(pd.to_numeric, errors="coerce")
        out["FLAG_DOCUMENT_SUM"] = docs.fillna(0).sum(axis=1)
    else:
        out["FLAG_DOCUMENT_SUM"] = 0.0

    request_cols = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    present_requests = [column for column in request_cols if column in out.columns]
    if present_requests:
        requests = out[present_requests].apply(pd.to_numeric, errors="coerce")
        out["AMT_REQ_CREDIT_BUREAU_TOTAL"] = requests.sum(axis=1, skipna=True)
        out.loc[requests.isna().all(axis=1), "AMT_REQ_CREDIT_BUREAU_TOTAL"] = np.nan
    else:
        out["AMT_REQ_CREDIT_BUREAU_TOTAL"] = np.nan

    if "CODE_GENDER" in out.columns:
        gender = out["CODE_GENDER"].astype("object")
        gender = gender.mask(gender.isin(["XNA", ""]), np.nan)
        out["CODE_GENDER"] = gender.fillna("Non renseigné")
    if "OCCUPATION_TYPE" in out.columns:
        out["OCCUPATION_TYPE"] = out["OCCUPATION_TYPE"].fillna("Inconnu")
    if "NAME_TYPE_SUITE" in out.columns:
        out["NAME_TYPE_SUITE"] = out["NAME_TYPE_SUITE"].fillna("Unknown")
    if "ORGANIZATION_TYPE" in out.columns:
        out["ORGANIZATION_TYPE"] = out["ORGANIZATION_TYPE"].fillna("Other")

    out = out.replace([np.inf, -np.inf], np.nan)
    return _keep_known_categories(out)


def _keep_known_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Ramène une modalité inattendue vers un niveau prévu, sans utiliser la cible."""
    fallbacks = {
        "CODE_GENDER": "Non renseigné",
        "OCCUPATION_TYPE": "Inconnu",
        "NAME_FAMILY_STATUS": "Unknown",
        "NAME_TYPE_SUITE": "Unknown",
        "ORGANIZATION_TYPE": "Other",
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
    }
    for column, levels in CATEGORY_LEVELS.items():
        if column not in df.columns:
            continue
        allowed = set(levels)
        unknown = df[column].notna() & ~df[column].isin(allowed)
        if unknown.any():
            df.loc[unknown, column] = fallbacks.get(column, levels[0])
    return df


def _grouped(frame: pd.DataFrame, aggregations: dict, prefix: str) -> pd.DataFrame:
    if frame.empty or "SK_ID_CURR" not in frame.columns:
        return pd.DataFrame()
    grouped = frame.groupby("SK_ID_CURR", sort=False).agg(aggregations)
    grouped.columns = [f"{prefix}_{name}" for name in grouped.columns]
    return grouped


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    df = bureau.copy()
    df["CREDIT_ACTIVE"] = df["CREDIT_ACTIVE"].astype(str)
    df["IS_ACTIVE"] = (df["CREDIT_ACTIVE"] == "Active").astype(np.int8)
    df["IS_CLOSED"] = (df["CREDIT_ACTIVE"] == "Closed").astype(np.int8)
    df["IS_SOLD"] = (df["CREDIT_ACTIVE"] == "Sold").astype(np.int8)
    for column in (
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_OVERDUE",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_MAX_OVERDUE",
        "AMT_ANNUITY",
        "DAYS_CREDIT",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "DAYS_CREDIT_UPDATE",
        "CREDIT_DAY_OVERDUE",
        "CNT_CREDIT_PROLONG",
    ):
        if column in df.columns:
            df[column] = _num(df, column)
            if column.startswith("DAYS_"):
                df[column] = _mask_sentinel(df[column])

    aggregations = {
        "SK_ID_BUREAU": "size",
        "IS_ACTIVE": "sum",
        "IS_CLOSED": "sum",
        "IS_SOLD": "sum",
        "AMT_CREDIT_SUM": ["mean", "sum", "max"],
        "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "max"],
        "DAYS_CREDIT": ["min", "max", "mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
    }
    if "AMT_CREDIT_SUM_LIMIT" in df.columns:
        aggregations["AMT_CREDIT_SUM_LIMIT"] = "mean"
    if "AMT_CREDIT_MAX_OVERDUE" in df.columns:
        aggregations["AMT_CREDIT_MAX_OVERDUE"] = "max"
    if "CNT_CREDIT_PROLONG" in df.columns:
        aggregations["CNT_CREDIT_PROLONG"] = "sum"

    raw = df.groupby("SK_ID_CURR", sort=False).agg(aggregations)
    raw.columns = [
        "BUREAU_COUNT" if col == ("SK_ID_BUREAU", "size") else
        "BUREAU_ACTIVE_COUNT" if col == ("IS_ACTIVE", "sum") else
        "BUREAU_CLOSED_COUNT" if col == ("IS_CLOSED", "sum") else
        "BUREAU_SOLD_COUNT" if col == ("IS_SOLD", "sum") else
        "BUREAU_AMT_CREDIT_SUM_MEAN" if col == ("AMT_CREDIT_SUM", "mean") else
        "BUREAU_AMT_CREDIT_SUM_SUM" if col == ("AMT_CREDIT_SUM", "sum") else
        "BUREAU_AMT_CREDIT_SUM_MAX" if col == ("AMT_CREDIT_SUM", "max") else
        "BUREAU_AMT_CREDIT_SUM_DEBT_MEAN" if col == ("AMT_CREDIT_SUM_DEBT", "mean") else
        "BUREAU_AMT_CREDIT_SUM_DEBT_SUM" if col == ("AMT_CREDIT_SUM_DEBT", "sum") else
        "BUREAU_AMT_OVERDUE_MEAN" if col == ("AMT_CREDIT_SUM_OVERDUE", "mean") else
        "BUREAU_AMT_OVERDUE_MAX" if col == ("AMT_CREDIT_SUM_OVERDUE", "max") else
        "BUREAU_DAYS_CREDIT_MIN" if col == ("DAYS_CREDIT", "min") else
        "BUREAU_DAYS_CREDIT_MAX" if col == ("DAYS_CREDIT", "max") else
        "BUREAU_DAYS_CREDIT_MEAN" if col == ("DAYS_CREDIT", "mean") else
        "BUREAU_CREDIT_DAY_OVERDUE_MAX" if col == ("CREDIT_DAY_OVERDUE", "max") else
        "BUREAU_CREDIT_DAY_OVERDUE_MEAN" if col == ("CREDIT_DAY_OVERDUE", "mean") else
        "BUREAU_AMT_CREDIT_SUM_LIMIT_MEAN" if col == ("AMT_CREDIT_SUM_LIMIT", "mean") else
        "BUREAU_AMT_MAX_OVERDUE_MAX" if col == ("AMT_CREDIT_MAX_OVERDUE", "max") else
        "BUREAU_CNT_PROLONG_SUM" if col == ("CNT_CREDIT_PROLONG", "sum") else
        f"BUREAU_{col[0]}_{col[1]}".upper()
        for col in raw.columns
    ]
    out = raw
    credit_sum = out.get("BUREAU_AMT_CREDIT_SUM_SUM")
    debt_sum = out.get("BUREAU_AMT_CREDIT_SUM_DEBT_SUM")
    if credit_sum is not None and debt_sum is not None:
        out["BUREAU_DEBT_CREDIT_RATIO"] = debt_sum / credit_sum.replace(0, np.nan)

    if "AMT_CREDIT_SUM" in df.columns and "AMT_CREDIT_SUM_DEBT" in df.columns:
        active = df[df["IS_ACTIVE"] == 1]
        if not active.empty:
            active_stats = active.groupby("SK_ID_CURR", sort=False).agg(
                BUREAU_ACTIVE_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
                BUREAU_ACTIVE_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
            )
            out = out.join(active_stats, how="left")
            out["BUREAU_ACTIVE_DEBT_CREDIT_RATIO"] = out["BUREAU_ACTIVE_DEBT_SUM"] / out[
                "BUREAU_ACTIVE_CREDIT_SUM"
            ].replace(0, np.nan)
            out = out.drop(columns=["BUREAU_ACTIVE_CREDIT_SUM", "BUREAU_ACTIVE_DEBT_SUM"], errors="ignore")

    if "CREDIT_TYPE" in df.columns:
        mapping = {
            "Consumer credit": "BUREAU_CONSUMER_COUNT",
            "Credit card": "BUREAU_CARD_COUNT",
            "Car loan": "BUREAU_CAR_COUNT",
            "Mortgage": "BUREAU_MORTGAGE_COUNT",
            "Microloan": "BUREAU_MICRO_COUNT",
        }
        types = df["CREDIT_TYPE"].astype(str)
        for label, column in mapping.items():
            counts = df.loc[types == label].groupby("SK_ID_CURR", sort=False).size()
            out[column] = counts

    if "DAYS_CREDIT" in df.columns:
        recent = df[df["DAYS_CREDIT"] >= -365]
        if not recent.empty:
            recent_stats = recent.groupby("SK_ID_CURR", sort=False).agg(
                BUREAU_RECENT_YEAR_COUNT=("SK_ID_CURR", "size"),
                BUREAU_RECENT_YEAR_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
                BUREAU_RECENT_YEAR_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum")
                if "AMT_CREDIT_SUM_DEBT" in recent.columns
                else ("SK_ID_CURR", "size"),
            )
            if "AMT_CREDIT_SUM_DEBT" not in recent.columns:
                recent_stats = recent_stats.drop(columns=["BUREAU_RECENT_YEAR_DEBT_SUM"], errors="ignore")
                recent_stats["BUREAU_RECENT_YEAR_DEBT_SUM"] = np.nan
            out = out.join(recent_stats, how="left")
    return out


def aggregate_bureau_balance(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    status = bureau_balance["STATUS"].astype(str)
    frame = pd.DataFrame(
        {
            "SK_ID_BUREAU": bureau_balance["SK_ID_BUREAU"].to_numpy(),
            "MONTHS_BALANCE": pd.to_numeric(bureau_balance["MONTHS_BALANCE"], errors="coerce").to_numpy(),
            "STATUS_NUM": pd.to_numeric(status, errors="coerce").to_numpy(),
            "LATE": status.isin(list("12345")).to_numpy(dtype=np.int8),
            "CLOSED": (status == "C").to_numpy(dtype=np.int8),
            "UNKNOWN": (status == "X").to_numpy(dtype=np.int8),
        }
    )
    link = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates()
    frame = frame.merge(link, on="SK_ID_BUREAU", how="inner")
    del link
    grouped = frame.groupby("SK_ID_CURR", sort=False)
    out = pd.DataFrame(
        {
            "BB_MONTHS_COUNT": grouped.size(),
            "BB_STATUS_MEAN": grouped["STATUS_NUM"].mean(),
            "BB_STATUS_MAX": grouped["STATUS_NUM"].max(),
            "BB_LATE_SHARE": grouped["LATE"].mean(),
            "BB_CLOSED_SHARE": grouped["CLOSED"].mean(),
            "BB_UNKNOWN_SHARE": grouped["UNKNOWN"].mean(),
        }
    )
    recent_12 = frame[frame["MONTHS_BALANCE"] >= -12]
    if not recent_12.empty:
        g12 = recent_12.groupby("SK_ID_CURR", sort=False)
        out = out.join(
            pd.DataFrame(
                {
                    "BB_RECENT_12_LATE_SHARE": g12["LATE"].mean(),
                    "BB_RECENT_12_STATUS_MEAN": g12["STATUS_NUM"].mean(),
                }
            ),
            how="left",
        )
    recent_36 = frame[frame["MONTHS_BALANCE"] >= -36]
    if not recent_36.empty:
        out = out.join(
            recent_36.groupby("SK_ID_CURR", sort=False)["LATE"].mean().rename("BB_RECENT_36_LATE_SHARE"),
            how="left",
        )
    del frame
    gc.collect()
    return out


def aggregate_previous(previous: pd.DataFrame) -> pd.DataFrame:
    df = previous.copy()
    status = df["NAME_CONTRACT_STATUS"].astype(str)
    df["APPROVED"] = (status == "Approved").astype(np.int8)
    df["REFUSED"] = (status == "Refused").astype(np.int8)
    df["CANCELED"] = (status == "Canceled").astype(np.int8)
    for column in (
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "RATE_DOWN_PAYMENT",
        "DAYS_DECISION",
        "CNT_PAYMENT",
        "NFLAG_INSURED_ON_APPROVAL",
    ):
        if column in df.columns:
            df[column] = _num(df, column)
            if column.startswith("DAYS_"):
                df[column] = _mask_sentinel(df[column])
    asked = _num(df, "AMT_APPLICATION")
    granted = _num(df, "AMT_CREDIT").replace(0, np.nan)
    df["APP_CREDIT_RATIO"] = asked / granted

    grouped = df.groupby("SK_ID_CURR", sort=False)
    out = pd.DataFrame(
        {
            "PREV_COUNT": grouped.size(),
            "PREV_APPROVED_COUNT": grouped["APPROVED"].sum(),
            "PREV_REFUSED_COUNT": grouped["REFUSED"].sum(),
            "PREV_CANCELED_COUNT": grouped["CANCELED"].sum(),
            "PREV_AMT_CREDIT_MEAN": grouped["AMT_CREDIT"].mean(),
            "PREV_AMT_CREDIT_MAX": grouped["AMT_CREDIT"].max(),
            "PREV_AMT_CREDIT_SUM": grouped["AMT_CREDIT"].sum(),
            "PREV_DAYS_DECISION_MIN": grouped["DAYS_DECISION"].min(),
            "PREV_DAYS_DECISION_MAX": grouped["DAYS_DECISION"].max(),
            "PREV_APP_CREDIT_RATIO_MEAN": grouped["APP_CREDIT_RATIO"].mean(),
        }
    )
    out["PREV_REFUSED_RATE"] = out["PREV_REFUSED_COUNT"] / out["PREV_COUNT"].replace(0, np.nan)
    if "AMT_ANNUITY" in df.columns:
        out["PREV_AMT_ANNUITY_MEAN"] = grouped["AMT_ANNUITY"].mean()
    if "AMT_DOWN_PAYMENT" in df.columns:
        out["PREV_AMT_DOWN_PAYMENT_MEAN"] = grouped["AMT_DOWN_PAYMENT"].mean()
    if "CNT_PAYMENT" in df.columns:
        out["PREV_CNT_PAYMENT_MEAN"] = grouped["CNT_PAYMENT"].mean()
    if "RATE_DOWN_PAYMENT" in df.columns:
        out["PREV_RATE_DOWN_PAYMENT_MEAN"] = grouped["RATE_DOWN_PAYMENT"].mean()

    approved = df[df["APPROVED"] == 1]
    if not approved.empty:
        out = out.join(
            approved.groupby("SK_ID_CURR", sort=False)["AMT_CREDIT"].mean().rename("PREV_APPROVED_CREDIT_MEAN"),
            how="left",
        )
        if "NFLAG_INSURED_ON_APPROVAL" in approved.columns:
            out = out.join(
                approved.groupby("SK_ID_CURR", sort=False)["NFLAG_INSURED_ON_APPROVAL"]
                .mean()
                .rename("PREV_INSURED_SHARE"),
                how="left",
            )

    if "NAME_YIELD_GROUP" in df.columns:
        high = (df["NAME_YIELD_GROUP"].astype(str) == "high").astype(np.int8)
        out = out.join(
            high.groupby(df["SK_ID_CURR"], sort=False).mean().rename("PREV_YIELD_HIGH_SHARE"),
            how="left",
        )
    if "NAME_CONTRACT_TYPE" in df.columns:
        contract = df["NAME_CONTRACT_TYPE"].astype(str)
        for label, column in (
            ("Cash loans", "PREV_CASH_COUNT"),
            ("Consumer loans", "PREV_CONSUMER_COUNT"),
            ("Revolving loans", "PREV_REVOLVING_COUNT"),
        ):
            out[column] = contract[contract == label].groupby(df.loc[contract == label, "SK_ID_CURR"], sort=False).size()

    recent = df[df["DAYS_DECISION"] >= -365]
    if not recent.empty:
        recent_stats = recent.groupby("SK_ID_CURR", sort=False).agg(
            PREV_RECENT_YEAR_COUNT=("SK_ID_CURR", "size"),
            PREV_RECENT_YEAR_REFUSED_COUNT=("REFUSED", "sum"),
        )
        out = out.join(recent_stats, how="left")
    return out


def aggregate_pos(pos: pd.DataFrame) -> pd.DataFrame:
    df = pos.copy()
    for column in ("SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "MONTHS_BALANCE"):
        if column in df.columns:
            df[column] = _num(df, column)
    grouped = df.groupby("SK_ID_CURR", sort=False)
    out = pd.DataFrame(
        {
            "POS_COUNT": grouped.size(),
            "POS_SK_DPD_MEAN": grouped["SK_DPD"].mean(),
            "POS_SK_DPD_MAX": grouped["SK_DPD"].max(),
            "POS_INSTALMENT_FUTURE_MEAN": grouped["CNT_INSTALMENT_FUTURE"].mean(),
        }
    )
    if "SK_DPD_DEF" in df.columns:
        out["POS_SK_DPD_DEF_MEAN"] = grouped["SK_DPD_DEF"].mean()
        out["POS_SK_DPD_DEF_MAX"] = grouped["SK_DPD_DEF"].max()
    if "NAME_CONTRACT_STATUS" in df.columns:
        completed = (df["NAME_CONTRACT_STATUS"].astype(str) == "Completed").astype(np.int8)
        out["POS_COMPLETED_SHARE"] = completed.groupby(df["SK_ID_CURR"], sort=False).mean()
    if "MONTHS_BALANCE" in df.columns:
        recent = df[df["MONTHS_BALANCE"] >= -12]
        if not recent.empty:
            recent_stats = recent.groupby("SK_ID_CURR", sort=False).agg(
                POS_RECENT_12_COUNT=("SK_ID_CURR", "size"),
                POS_RECENT_12_DPD_MAX=("SK_DPD", "max"),
                POS_RECENT_12_DPD_MEAN=("SK_DPD", "mean"),
            )
            out = out.join(recent_stats, how="left")
    return out


def aggregate_installments(installments: pd.DataFrame) -> pd.DataFrame:
    df = installments.copy()
    due = _num(df, "AMT_INSTALMENT")
    paid = _num(df, "AMT_PAYMENT")
    df["PAYMENT_RATIO"] = paid / due.replace(0, np.nan)
    df["PAYMENT_DIFF"] = paid - due
    df["LATE_DAYS"] = _num(df, "DAYS_ENTRY_PAYMENT") - _num(df, "DAYS_INSTALMENT")
    df["IS_LATE"] = (df["LATE_DAYS"] > 0).astype(np.float32)
    df["IS_UNDERPAY"] = (paid < due).astype(np.float32)
    days = _num(df, "DAYS_ENTRY_PAYMENT")
    grouped = df.groupby("SK_ID_CURR", sort=False)
    out = pd.DataFrame(
        {
            "INST_COUNT": grouped.size(),
            "INST_PAYMENT_RATIO_MEAN": grouped["PAYMENT_RATIO"].mean(),
            "INST_PAYMENT_RATIO_MIN": grouped["PAYMENT_RATIO"].min(),
            "INST_LATE_DAYS_MEAN": grouped["LATE_DAYS"].mean(),
            "INST_LATE_DAYS_MAX": grouped["LATE_DAYS"].max(),
            "INST_LATE_SHARE": grouped["IS_LATE"].mean(),
            "INST_UNDERPAY_SHARE": grouped["IS_UNDERPAY"].mean(),
            "INST_PAYMENT_DIFF_MEAN": grouped["PAYMENT_DIFF"].mean(),
            "INST_PAYMENT_DIFF_SUM": grouped["PAYMENT_DIFF"].sum(),
        }
    )
    for days_back, label in ((365, "YEAR"), (730, "2Y")):
        recent = df[days >= -days_back]
        if recent.empty:
            continue
        stats = recent.groupby("SK_ID_CURR", sort=False).agg(
            **{
                f"INST_RECENT_{label}_COUNT": ("SK_ID_CURR", "size"),
                f"INST_RECENT_{label}_PAYMENT_RATIO_MEAN": ("PAYMENT_RATIO", "mean"),
                f"INST_RECENT_{label}_LATE_SHARE": ("IS_LATE", "mean"),
            }
        )
        if label == "YEAR":
            stats[f"INST_RECENT_{label}_LATE_DAYS_MEAN"] = recent.groupby("SK_ID_CURR", sort=False)[
                "LATE_DAYS"
            ].mean()
        out = out.join(stats, how="left")
    return out


def aggregate_credit_card(credit_card: pd.DataFrame) -> pd.DataFrame:
    df = credit_card.copy()
    for column in (
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_CURRENT",
        "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "CNT_DRAWINGS_CURRENT",
        "CNT_DRAWINGS_ATM_CURRENT",
        "SK_DPD",
        "SK_DPD_DEF",
        "MONTHS_BALANCE",
    ):
        if column in df.columns:
            df[column] = _num(df, column)
    limit = _num(df, "AMT_CREDIT_LIMIT_ACTUAL").replace(0, np.nan)
    df["UTILIZATION"] = _num(df, "AMT_BALANCE") / limit
    grouped = df.groupby("SK_ID_CURR", sort=False)
    out = pd.DataFrame(
        {
            "CC_COUNT": grouped.size(),
            "CC_AMT_BALANCE_MEAN": grouped["AMT_BALANCE"].mean(),
            "CC_AMT_BALANCE_MAX": grouped["AMT_BALANCE"].max(),
            "CC_UTILIZATION_MEAN": grouped["UTILIZATION"].mean(),
            "CC_UTILIZATION_MAX": grouped["UTILIZATION"].max(),
            "CC_SK_DPD_MAX": grouped["SK_DPD"].max(),
            "CC_SK_DPD_MEAN": grouped["SK_DPD"].mean(),
        }
    )
    if "SK_DPD_DEF" in df.columns:
        out["CC_SK_DPD_DEF_MAX"] = grouped["SK_DPD_DEF"].max()
    if "AMT_DRAWINGS_CURRENT" in df.columns:
        out["CC_DRAWINGS_MEAN"] = grouped["AMT_DRAWINGS_CURRENT"].mean()
    if "AMT_DRAWINGS_ATM_CURRENT" in df.columns:
        out["CC_DRAWINGS_ATM_MEAN"] = grouped["AMT_DRAWINGS_ATM_CURRENT"].mean()
    if "AMT_PAYMENT_TOTAL_CURRENT" in df.columns:
        out["CC_PAYMENT_MEAN"] = grouped["AMT_PAYMENT_TOTAL_CURRENT"].mean()
    if "CNT_DRAWINGS_CURRENT" in df.columns:
        out["CC_CNT_DRAWINGS_MEAN"] = grouped["CNT_DRAWINGS_CURRENT"].mean()

    if "MONTHS_BALANCE" in df.columns:
        for months, label in ((12, "12"), (6, "6")):
            recent = df[df["MONTHS_BALANCE"] >= -months]
            if recent.empty:
                continue
            g = recent.groupby("SK_ID_CURR", sort=False)
            stats = pd.DataFrame(
                {
                    f"CC_RECENT_{label}_UTILIZATION_MEAN": g["UTILIZATION"].mean(),
                }
            )
            if label == "12":
                stats[f"CC_RECENT_{label}_UTILIZATION_MAX"] = g["UTILIZATION"].max()
                stats[f"CC_RECENT_{label}_DPD_MAX"] = g["SK_DPD"].max()
                stats[f"CC_RECENT_{label}_BALANCE_MEAN"] = g["AMT_BALANCE"].mean()
            out = out.join(stats, how="left")
    return out


def _zero_fill_counts(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        if not column.startswith(COUNT_PREFIXES) and column not in {
            "HAS_BUREAU",
            "HAS_PREV",
            "HAS_POS",
            "HAS_INST",
            "HAS_CC",
            "FLAG_DOCUMENT_SUM",
        }:
            continue
        if column.endswith(COUNT_LIKE_SUFFIXES) or column.startswith("HAS_"):
            frame[column] = frame[column].fillna(0.0)
    return frame


def build_modeling_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    application = tables["application"].copy()
    if application["SK_ID_CURR"].duplicated().any():
        application = application.drop_duplicates("SK_ID_CURR", keep="first")

    frame = add_application_features(application)
    builders = [
        ("bureau", lambda: aggregate_bureau(tables["bureau"])),
        ("bureau_balance", lambda: aggregate_bureau_balance(tables["bureau"], tables["bureau_balance"])),
        ("previous", lambda: aggregate_previous(tables["previous_application"])),
        ("pos", lambda: aggregate_pos(tables["POS_CASH_balance"])),
        ("installments", lambda: aggregate_installments(tables["installments_payments"])),
        ("credit_card", lambda: aggregate_credit_card(tables["credit_card_balance"])),
    ]
    for _, builder in builders:
        aggregate = builder()
        if aggregate is not None and not aggregate.empty:
            frame = frame.merge(aggregate, left_on="SK_ID_CURR", right_index=True, how="left")
        del aggregate
        gc.collect()

    frame["HAS_BUREAU"] = (frame.get("BUREAU_COUNT", 0).fillna(0) > 0).astype(float)
    frame["HAS_PREV"] = (frame.get("PREV_COUNT", 0).fillna(0) > 0).astype(float)
    frame["HAS_POS"] = (frame.get("POS_COUNT", 0).fillna(0) > 0).astype(float)
    frame["HAS_INST"] = (frame.get("INST_COUNT", 0).fillna(0) > 0).astype(float)
    frame["HAS_CC"] = (frame.get("CC_COUNT", 0).fillna(0) > 0).astype(float)
    frame = _zero_fill_counts(frame)
    return frame.reset_index(drop=True)


def _select_columns(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        # Les agrégats absents (historique vide sur une fixture courte) sont remplacés par des manquants.
        for column in missing:
            frame[column] = np.nan
        still = [column for column in columns if column not in frame.columns]
        if still:
            raise KeyError(f"{label} absentes : {still}")
    return frame[columns].copy()


def served_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _select_columns(frame, SERVED_NUMERIC + SERVED_CATEGORICAL, "Colonnes servies")


def research_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(dict.fromkeys(RESEARCH_NUMERIC + RESEARCH_CATEGORICAL + TARGET_ENCODE_COLUMNS))
    return _select_columns(frame, columns, "Colonnes de recherche")


def category_list(columns: list[str]) -> list[list[str]]:
    return [CATEGORY_LEVELS[column] for column in columns]
