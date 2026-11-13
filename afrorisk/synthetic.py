"""Fixture synthétique au schéma des colonnes utilisées.

Le signal est volontairement simple pour que les tests et l'interface
fonctionnent sans le jeu Kaggle. Ce n'est pas Home Credit.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from afrorisk.config import DATA_FIXTURES, EMPLOYMENT_SENTINEL, RANDOM_STATE
from afrorisk.schema import (
    CONTRACT_TYPES,
    EDUCATION_TYPES,
    HOUSING_TYPES,
    OCCUPATION_TYPES,
    SUITE_TYPES,
    WEEKDAYS,
    YN,
)

FAMILY_SAMPLE = [
    "Single / not married",
    "Married",
    "Civil marriage",
    "Widow",
    "Separated",
]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate(n_applications: int = 800, seed: int = RANDOM_STATE) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(100_000, 100_000 + n_applications)

    ext1 = rng.uniform(0.01, 0.99, n_applications)
    ext2 = rng.uniform(0.01, 0.99, n_applications)
    ext3 = rng.uniform(0.01, 0.99, n_applications)
    ext1_obs = ext1.copy()
    ext1_obs[rng.random(n_applications) < 0.40] = np.nan
    ext3_obs = ext3.copy()
    ext3_obs[rng.random(n_applications) < 0.18] = np.nan

    income = rng.lognormal(mean=11.55, sigma=0.35, size=n_applications).clip(25_000, 800_000)
    credit_income = rng.uniform(1.2, 6.5, n_applications)
    credit = income * credit_income
    annuity = credit * rng.uniform(0.03, 0.09, n_applications)
    goods = credit * rng.uniform(0.82, 1.08, n_applications)
    annuity_missing = rng.random(n_applications) < 0.02
    annuity = annuity.astype(float)
    annuity[annuity_missing] = np.nan

    age = rng.uniform(21, 67, n_applications)
    employment_years = np.minimum(rng.uniform(0, 35, n_applications), np.maximum(age - 18, 0))
    pensioner = (rng.random(n_applications) < 0.16) | (age > 60)
    days_employed = np.where(pensioner, EMPLOYMENT_SENTINEL, -employment_years * 365.25)

    late_share = np.where(
        rng.random(n_applications) < 0.22,
        rng.uniform(0.25, 0.85, n_applications),
        rng.uniform(0.0, 0.08, n_applications),
    )
    refused_rate = np.clip(0.15 + 0.55 * late_share + rng.normal(0, 0.05, n_applications), 0.02, 0.9)

    logit = (
        -0.75
        - 1.7 * ext3
        - 1.15 * ext2
        - 0.45 * np.nan_to_num(ext1, nan=0.5)
        + 0.28 * (credit_income - 3.0)
        + 1.35 * late_share
        + 0.22 * pensioner.astype(float)
        + rng.normal(0, 0.45, n_applications)
    )
    target = rng.binomial(1, _sigmoid(logit))

    gender = rng.choice(["F", "M"], size=n_applications, p=[0.66, 0.34])
    gender[rng.random(n_applications) < 0.01] = "XNA"
    occupation_pool = [item for item in OCCUPATION_TYPES if item != "Inconnu"]
    occupation = rng.choice(occupation_pool, size=n_applications)
    occupation = occupation.astype(object)
    occupation[rng.random(n_applications) < 0.30] = np.nan

    application = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "TARGET": target.astype(int),
            "NAME_CONTRACT_TYPE": rng.choice(CONTRACT_TYPES, size=n_applications, p=[0.9, 0.1]),
            "CODE_GENDER": gender,
            "FLAG_OWN_CAR": rng.choice(YN, size=n_applications, p=[0.34, 0.66]),
            "FLAG_OWN_REALTY": rng.choice(YN, size=n_applications, p=[0.69, 0.31]),
            "CNT_CHILDREN": rng.choice([0, 1, 2, 3, 4], size=n_applications, p=[0.7, 0.18, 0.08, 0.03, 0.01]),
            "AMT_INCOME_TOTAL": np.round(income, 2),
            "AMT_CREDIT": np.round(credit, 2),
            "AMT_ANNUITY": np.round(annuity, 2),
            "AMT_GOODS_PRICE": np.round(goods, 2),
            "NAME_INCOME_TYPE": np.where(
                pensioner,
                "Pensioner",
                rng.choice(
                    ["Working", "Commercial associate", "State servant", "Unemployed"],
                    size=n_applications,
                    p=[0.62, 0.24, 0.12, 0.02],
                ),
            ),
            "NAME_EDUCATION_TYPE": rng.choice(
                EDUCATION_TYPES, size=n_applications, p=[0.71, 0.22, 0.04, 0.025, 0.005]
            ),
            "NAME_FAMILY_STATUS": rng.choice(
                FAMILY_SAMPLE, size=n_applications, p=[0.14, 0.64, 0.1, 0.05, 0.07]
            ),
            "NAME_HOUSING_TYPE": rng.choice(
                HOUSING_TYPES, size=n_applications, p=[0.88, 0.04, 0.045, 0.02, 0.01, 0.005]
            ),
            "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.07, n_applications),
            "DAYS_BIRTH": np.round(-age * 365.25).astype(int),
            "DAYS_EMPLOYED": np.round(days_employed).astype(int),
            "DAYS_REGISTRATION": rng.integers(-15000, -1, n_applications),
            "DAYS_ID_PUBLISH": rng.integers(-7000, -1, n_applications),
            "CNT_FAM_MEMBERS": None,
            "REGION_RATING_CLIENT": rng.choice([1, 2, 3], size=n_applications, p=[0.1, 0.74, 0.16]),
            "REGION_RATING_CLIENT_W_CITY": None,
            "OCCUPATION_TYPE": occupation,
            "EXT_SOURCE_1": ext1_obs,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3_obs,
            "NAME_TYPE_SUITE": rng.choice(
                [item for item in SUITE_TYPES if item != "Unknown"],
                size=n_applications,
                p=[0.72, 0.16, 0.07, 0.025, 0.015, 0.007, 0.003],
            ),
            "OWN_CAR_AGE": rng.integers(0, 25, n_applications).astype(float),
            "FLAG_EMP_PHONE": rng.integers(0, 2, n_applications),
            "FLAG_WORK_PHONE": rng.integers(0, 2, n_applications),
            "FLAG_PHONE": rng.integers(0, 2, n_applications),
            "FLAG_EMAIL": rng.choice([0, 1], size=n_applications, p=[0.94, 0.06]),
            "WEEKDAY_APPR_PROCESS_START": rng.choice(WEEKDAYS, size=n_applications),
            "HOUR_APPR_PROCESS_START": rng.integers(8, 20, n_applications),
            "REG_CITY_NOT_LIVE_CITY": rng.choice([0, 1], size=n_applications, p=[0.92, 0.08]),
            "REG_CITY_NOT_WORK_CITY": rng.choice([0, 1], size=n_applications, p=[0.78, 0.22]),
            "LIVE_CITY_NOT_WORK_CITY": rng.choice([0, 1], size=n_applications, p=[0.82, 0.18]),
            "ORGANIZATION_TYPE": rng.choice(
                ["Business Entity Type 3", "XNA", "Self-employed", "Other", "Medicine", "Government", "School"],
                size=n_applications,
                p=[0.28, 0.18, 0.16, 0.14, 0.1, 0.08, 0.06],
            ),
            "OBS_30_CNT_SOCIAL_CIRCLE": rng.choice([0, 1, 2, 4, 8], size=n_applications, p=[0.55, 0.2, 0.15, 0.07, 0.03]).astype(float),
            "DEF_30_CNT_SOCIAL_CIRCLE": rng.choice([0, 1, 2], size=n_applications, p=[0.9, 0.08, 0.02]).astype(float),
            "OBS_60_CNT_SOCIAL_CIRCLE": rng.choice([0, 1, 2, 4, 8], size=n_applications, p=[0.55, 0.2, 0.15, 0.07, 0.03]).astype(float),
            "DEF_60_CNT_SOCIAL_CIRCLE": rng.choice([0, 1, 2], size=n_applications, p=[0.92, 0.07, 0.01]).astype(float),
            "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 3000, n_applications),
            "AMT_REQ_CREDIT_BUREAU_HOUR": rng.choice([0, 1], size=n_applications, p=[0.99, 0.01]).astype(float),
            "AMT_REQ_CREDIT_BUREAU_DAY": rng.choice([0, 1], size=n_applications, p=[0.98, 0.02]).astype(float),
            "AMT_REQ_CREDIT_BUREAU_WEEK": rng.choice([0, 1, 2], size=n_applications, p=[0.9, 0.08, 0.02]).astype(float),
            "AMT_REQ_CREDIT_BUREAU_MON": rng.choice([0, 1, 2, 4], size=n_applications, p=[0.7, 0.18, 0.08, 0.04]).astype(float),
            "AMT_REQ_CREDIT_BUREAU_QRT": rng.choice([0, 1, 2, 3], size=n_applications, p=[0.75, 0.15, 0.07, 0.03]).astype(float),
            "AMT_REQ_CREDIT_BUREAU_YEAR": rng.choice(
                [0, 1, 2, 3, 5, 8], size=n_applications, p=[0.35, 0.28, 0.18, 0.1, 0.06, 0.03]
            ).astype(float),
        }
    )
    rating = application["REGION_RATING_CLIENT"].to_numpy()
    application["REGION_RATING_CLIENT_W_CITY"] = np.clip(rating + rng.choice([-1, 0, 0, 0, 1], size=n_applications), 1, 3)
    owns_car = application["FLAG_OWN_CAR"].to_numpy()
    application.loc[owns_car == "N", "OWN_CAR_AGE"] = np.nan
    for doc in range(2, 22):
        col = f"FLAG_DOCUMENT_{doc}"
        if doc == 3:
            application[col] = rng.choice([0, 1], size=n_applications, p=[0.28, 0.72])
        elif doc in {6, 8}:
            application[col] = rng.choice([0, 1], size=n_applications, p=[0.9, 0.1])
        else:
            application[col] = 0
    children = application["CNT_CHILDREN"].to_numpy()
    partner = application["NAME_FAMILY_STATUS"].isin(["Married", "Civil marriage"]).to_numpy()
    application["CNT_FAM_MEMBERS"] = children + 1 + partner.astype(int)
    bureau_missing = rng.random(n_applications) < 0.08
    for column in (
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ):
        application.loc[bureau_missing, column] = np.nan

    bureau_rows = []
    balance_rows = []
    previous_rows = []
    pos_rows = []
    installment_rows = []
    card_rows = []
    next_bureau = 1
    next_prev = 1

    for index, current_id in enumerate(ids):
        n_bureau = int(rng.integers(0, 5))
        if late_share[index] > 0.2 and n_bureau == 0:
            n_bureau = 1
        for ordinal in range(n_bureau):
            bureau_id = next_bureau
            next_bureau += 1
            active = ordinal == 0 or rng.random() < 0.45
            overdue_days = 0
            overdue_amt = 0.0
            if late_share[index] > 0.2 and ordinal == 0:
                overdue_days = int(rng.integers(5, 90))
                overdue_amt = float(rng.uniform(200, 8000))
            credit_sum = float(rng.uniform(5_000, 250_000))
            debt = credit_sum * float(rng.uniform(0, 0.8)) if active else 0.0
            days_credit = int(-rng.integers(30, 2500))
            bureau_rows.append(
                {
                    "SK_ID_CURR": int(current_id),
                    "SK_ID_BUREAU": bureau_id,
                    "CREDIT_ACTIVE": "Active" if active else "Closed",
                    "DAYS_CREDIT": days_credit,
                    "CREDIT_DAY_OVERDUE": overdue_days,
                    "DAYS_CREDIT_ENDDATE": days_credit + int(rng.integers(180, 1500)),
                    "DAYS_ENDDATE_FACT": None if active else days_credit + int(rng.integers(60, 800)),
                    "AMT_CREDIT_MAX_OVERDUE": round(overdue_amt * float(rng.uniform(1.0, 2.0)), 2),
                    "CNT_CREDIT_PROLONG": int(rng.integers(0, 3)) if rng.random() < 0.1 else 0,
                    "AMT_CREDIT_SUM": round(credit_sum, 2),
                    "AMT_CREDIT_SUM_DEBT": round(debt, 2),
                    "AMT_CREDIT_SUM_LIMIT": round(credit_sum * float(rng.uniform(0.0, 0.3)), 2),
                    "AMT_CREDIT_SUM_OVERDUE": round(overdue_amt, 2),
                    "CREDIT_TYPE": str(
                        rng.choice(
                            ["Consumer credit", "Credit card", "Car loan", "Mortgage", "Microloan"],
                            p=[0.7, 0.18, 0.06, 0.04, 0.02],
                        )
                    ),
                    "DAYS_CREDIT_UPDATE": days_credit + int(rng.integers(0, 200)),
                    "AMT_ANNUITY": round(credit_sum * float(rng.uniform(0.02, 0.08)), 2),
                }
            )
            n_months = int(rng.integers(4, 10))
            for month in range(1, n_months + 1):
                if late_share[index] > 0.2 and rng.random() < late_share[index]:
                    status = str(rng.choice(["1", "2", "3"]))
                elif rng.random() < 0.15:
                    status = "C" if not active else "0"
                elif rng.random() < 0.08:
                    status = "X"
                else:
                    status = "0"
                balance_rows.append(
                    {
                        "SK_ID_BUREAU": bureau_id,
                        "MONTHS_BALANCE": -month,
                        "STATUS": status,
                    }
                )

        n_prev = int(rng.integers(0, 4))
        for _ in range(n_prev):
            prev_id = next_prev
            next_prev += 1
            refused = rng.random() < refused_rate[index]
            status = "Refused" if refused else rng.choice(["Approved", "Canceled", "Unused offer"], p=[0.8, 0.12, 0.08])
            applied = float(rng.uniform(10_000, 300_000))
            granted = 0.0 if status == "Refused" else applied * float(rng.uniform(0.7, 1.05))
            days_decision = int(-rng.integers(20, 1800))
            previous_rows.append(
                {
                    "SK_ID_PREV": prev_id,
                    "SK_ID_CURR": int(current_id),
                    "NAME_CONTRACT_TYPE": str(rng.choice(["Cash loans", "Consumer loans", "Revolving loans"], p=[0.45, 0.4, 0.15])),
                    "AMT_ANNUITY": round(max(granted, applied) * float(rng.uniform(0.03, 0.1)), 2),
                    "AMT_APPLICATION": round(applied, 2),
                    "AMT_CREDIT": round(granted, 2),
                    "AMT_DOWN_PAYMENT": round(applied * float(rng.uniform(0.0, 0.2)), 2),
                    "AMT_GOODS_PRICE": round(applied * float(rng.uniform(0.85, 1.05)), 2),
                    "RATE_DOWN_PAYMENT": float(rng.uniform(0.0, 0.2)),
                    "NAME_CONTRACT_STATUS": status,
                    "DAYS_DECISION": days_decision,
                    "CODE_REJECT_REASON": "XAP" if status != "Refused" else str(rng.choice(["HC", "LIMIT", "SCO", "VERIF"])),
                    "NAME_CLIENT_TYPE": str(rng.choice(["Repeater", "New", "Refreshed"], p=[0.7, 0.2, 0.1])),
                    "NAME_PORTFOLIO": str(rng.choice(["POS", "Cash", "Cards"], p=[0.45, 0.4, 0.15])),
                    "CNT_PAYMENT": int(rng.choice([6, 12, 18, 24, 36])),
                    "NAME_YIELD_GROUP": str(rng.choice(["high", "middle", "low_normal", "low_action", "XNA"], p=[0.25, 0.3, 0.25, 0.1, 0.1])),
                    "DAYS_FIRST_DUE": None if status != "Approved" else days_decision + 30,
                    "DAYS_LAST_DUE": None if status != "Approved" else days_decision + int(rng.integers(180, 900)),
                    "DAYS_TERMINATION": None if status != "Approved" else days_decision + int(rng.integers(200, 1000)),
                    "NFLAG_INSURED_ON_APPROVAL": None if status != "Approved" else int(rng.integers(0, 2)),
                }
            )
            if status != "Approved":
                continue
            n_pos = int(rng.integers(3, 8))
            for month in range(1, n_pos + 1):
                dpd = int(rng.integers(10, 60)) if late_share[index] > 0.2 and rng.random() < late_share[index] else 0
                pos_rows.append(
                    {
                        "SK_ID_PREV": prev_id,
                        "SK_ID_CURR": int(current_id),
                        "MONTHS_BALANCE": -month,
                        "CNT_INSTALMENT": 12,
                        "CNT_INSTALMENT_FUTURE": max(12 - month, 0),
                        "NAME_CONTRACT_STATUS": "Active" if month < 3 else "Completed",
                        "SK_DPD": dpd,
                        "SK_DPD_DEF": dpd if dpd > 5 else 0,
                    }
                )
            n_inst = int(rng.integers(4, 10))
            for payment_number in range(n_inst):
                due_amount = float(rng.uniform(800, 8000))
                late = late_share[index] > 0.2 and rng.random() < late_share[index]
                paid_amount = due_amount * float(rng.uniform(0.4, 0.9) if late else rng.uniform(0.98, 1.02))
                due_day = -int(rng.integers(40, 900))
                pay_day = due_day + int(rng.integers(5, 40)) if late else due_day - int(rng.integers(0, 8))
                installment_rows.append(
                    {
                        "SK_ID_PREV": prev_id,
                        "SK_ID_CURR": int(current_id),
                        "NUM_INSTALMENT_VERSION": 1,
                        "DAYS_INSTALMENT": due_day,
                        "DAYS_ENTRY_PAYMENT": pay_day,
                        "AMT_INSTALMENT": round(due_amount, 2),
                        "AMT_PAYMENT": round(paid_amount, 2),
                    }
                )
            if rng.random() < 0.45:
                limit = float(rng.choice([50_000, 100_000, 150_000, 300_000]))
                utilization = float(rng.uniform(0.55, 0.98) if late_share[index] > 0.2 else rng.uniform(0.05, 0.55))
                for month in range(1, 5):
                    dpd = int(rng.integers(8, 45)) if late_share[index] > 0.2 and month == 1 else 0
                    card_rows.append(
                        {
                            "SK_ID_PREV": prev_id,
                            "SK_ID_CURR": int(current_id),
                            "MONTHS_BALANCE": -month,
                            "AMT_BALANCE": round(limit * utilization, 2),
                            "AMT_CREDIT_LIMIT_ACTUAL": limit,
                            "AMT_DRAWINGS_ATM_CURRENT": round(limit * utilization * 0.15, 2),
                            "AMT_DRAWINGS_CURRENT": round(limit * utilization * 0.2, 2),
                            "AMT_INST_MIN_REGULARITY": round(limit * 0.03, 2),
                            "AMT_PAYMENT_TOTAL_CURRENT": round(limit * utilization * 0.1, 2),
                            "CNT_DRAWINGS_ATM_CURRENT": int(rng.integers(0, 4)),
                            "CNT_DRAWINGS_CURRENT": int(rng.integers(0, 8)),
                            "CNT_INSTALMENT_MATURE_CUM": month,
                            "NAME_CONTRACT_STATUS": "Active",
                            "SK_DPD": dpd,
                            "SK_DPD_DEF": dpd if dpd > 5 else 0,
                        }
                    )

    def _frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows)[columns]

    from afrorisk.schema import (
        APPLICATION_COLUMNS,
        BUREAU_BALANCE_COLUMNS,
        BUREAU_COLUMNS,
        CREDIT_CARD_COLUMNS,
        INSTALLMENTS_COLUMNS,
        POS_CASH_COLUMNS,
        PREVIOUS_COLUMNS,
    )

    return {
        "application": application[APPLICATION_COLUMNS],
        "bureau": _frame(bureau_rows, BUREAU_COLUMNS),
        "bureau_balance": _frame(balance_rows, BUREAU_BALANCE_COLUMNS),
        "previous_application": _frame(previous_rows, PREVIOUS_COLUMNS),
        "POS_CASH_balance": _frame(pos_rows, POS_CASH_COLUMNS),
        "installments_payments": _frame(installment_rows, INSTALLMENTS_COLUMNS),
        "credit_card_balance": _frame(card_rows, CREDIT_CARD_COLUMNS),
    }


def write_fixture(directory: Path | None = None, n_applications: int = 800) -> Path:
    from afrorisk.schema import TABLE_FILES

    destination = directory or DATA_FIXTURES
    destination.mkdir(parents=True, exist_ok=True)
    tables = generate(n_applications=n_applications)
    for key, frame in tables.items():
        frame.to_csv(destination / TABLE_FILES[key], index=False)
    return destination


def main() -> None:
    path = write_fixture()
    print(f"Fixture synthétique écrite dans {path}")


if __name__ == "__main__":
    main()
