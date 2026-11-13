"""Pipeline de features et petit ajustement sur la fixture."""

import numpy as np
import pandas as pd
import pytest

from afrorisk.config import DATA_FIXTURES, SERVED_CATEGORICAL, SERVED_NUMERIC
from afrorisk.features import (
    add_application_features,
    aggregate_bureau,
    build_modeling_table,
    research_feature_frame,
    served_feature_frame,
)
from afrorisk.io import load_tables
from afrorisk.modeling import fit_model


def test_employment_sentinel_and_ratios():
    frame = pd.DataFrame(
        {
            "DAYS_BIRTH": [-36525.0],
            "DAYS_EMPLOYED": [365243],
            "AMT_INCOME_TOTAL": [100000.0],
            "AMT_CREDIT": [200000.0],
            "AMT_ANNUITY": [10000.0],
            "AMT_GOODS_PRICE": [180000.0],
            "EXT_SOURCE_1": [0.2],
            "EXT_SOURCE_2": [0.4],
            "EXT_SOURCE_3": [0.6],
            "CODE_GENDER": ["XNA"],
        }
    )
    out = add_application_features(frame)
    assert out.loc[0, "AGE_YEARS"] == pytest.approx(100.0)
    assert pd.isna(out.loc[0, "EMPLOYMENT_YEARS"])
    assert out.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(2.0)
    assert out.loc[0, "PAYMENT_RATE"] == pytest.approx(0.05)
    assert out.loc[0, "EXT_SOURCE_MEAN"] == pytest.approx(0.4)
    assert out.loc[0, "EXT_SOURCE_MIN"] == pytest.approx(0.2)
    assert out.loc[0, "EXT_SOURCE_MAX"] == pytest.approx(0.6)
    assert out.loc[0, "EXT_SOURCE_MISSING"] == pytest.approx(0.0)
    assert out.loc[0, "CODE_GENDER"] == "Non renseigné"


def test_bureau_aggregation_values():
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [10, 11, 12],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "DAYS_CREDIT": [-10, -100, -5],
            "CREDIT_DAY_OVERDUE": [0, 3, 0],
            "AMT_CREDIT_SUM": [100.0, 300.0, 50.0],
            "AMT_CREDIT_SUM_DEBT": [40.0, 0.0, 10.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 5.0, 0.0],
        }
    )
    agg = aggregate_bureau(bureau)
    assert agg.loc[1, "BUREAU_COUNT"] == 2
    assert agg.loc[1, "BUREAU_ACTIVE_COUNT"] == 1
    assert agg.loc[1, "BUREAU_AMT_CREDIT_SUM_SUM"] == pytest.approx(400.0)
    assert agg.loc[1, "BUREAU_DAYS_CREDIT_MIN"] == -100
    assert agg.loc[1, "BUREAU_AMT_OVERDUE_MEAN"] == pytest.approx(2.5)


def test_target_does_not_change_features():
    tables, source = load_tables(DATA_FIXTURES)
    assert source == "synthetic_fixture"
    original = build_modeling_table(tables)
    flipped = {key: value.copy() for key, value in tables.items()}
    flipped["application"]["TARGET"] = 1 - flipped["application"]["TARGET"]
    changed = build_modeling_table(flipped)
    features = [column for column in original.columns if column != "TARGET"]
    pd.testing.assert_frame_equal(original[features], changed[features])
    served = served_feature_frame(original)
    assert "TARGET" not in served.columns
    assert "CODE_GENDER" not in served.columns
    assert list(served.columns) == SERVED_NUMERIC + SERVED_CATEGORICAL
    research = research_feature_frame(original)
    assert "TARGET" not in research.columns
    assert len(original) == len(tables["application"])


def test_fixture_model_fits():
    tables, _ = load_tables(DATA_FIXTURES)
    frame = build_modeling_table(tables)
    y = frame["TARGET"].astype(int)
    X = served_feature_frame(frame)
    pipeline = fit_model(
        "decision_tree",
        X,
        y,
        SERVED_NUMERIC,
        SERVED_CATEGORICAL,
        fast=True,
    )
    proba = pipeline.predict_proba(X.iloc[:5])[:, 1]
    assert proba.shape == (5,)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()
