"""PSI et endpoint de dérive."""

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from afrorisk.api.main import app
from afrorisk.monitoring.drift import (
    compare_to_reference,
    psi_from_counts,
    reference_from_frame,
    stability_label,
)


def test_psi_is_zero_on_the_same_distribution():
    counts = [10, 20, 30, 40]
    assert psi_from_counts(counts, counts) == pytest.approx(0.0, abs=1e-9)


def test_psi_rises_when_the_mass_moves():
    stable = psi_from_counts([50, 50], [48, 52])
    shifted = psi_from_counts([90, 10], [10, 90])
    assert shifted > stable
    assert stability_label(shifted) == "dérive"
    assert stability_label(0.05) == "stable"
    assert stability_label(0.15) == "à surveiller"


def test_compare_flags_a_shifted_numeric_variable():
    reference_frame = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": list(range(20_000, 120_000, 1_000)),
            "NAME_CONTRACT_TYPE": ["Cash loans"] * 80 + ["Revolving loans"] * 20,
        }
    )
    # Les autres colonnes servies ne sont pas dans ce cadre réduit : seules celles
    # présentes dans la référence sont notées. On construit la référence à la main.
    reference = {
        "data_source": "test",
        "n_rows": 100,
        "numeric": {
            "AMT_INCOME_TOTAL": {
                "kind": "deciles",
                "n": 100,
                "edges": [20000, 60000, 120000],
                "counts": [40, 60],
            }
        },
        "categorical": {
            "NAME_CONTRACT_TYPE": {
                "n": 100,
                "levels": [
                    {"level": "Cash loans", "count": 90},
                    {"level": "Revolving loans", "count": 10},
                ],
            }
        },
    }
    recent = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100000] * 80 + [25000] * 20,
            "NAME_CONTRACT_TYPE": ["Revolving loans"] * 70 + ["Cash loans"] * 30,
        }
    )
    report = compare_to_reference(recent, reference)
    by_name = {item["feature"]: item for item in report["variables"]}
    assert by_name["NAME_CONTRACT_TYPE"]["status"] == "dérive"
    assert by_name["AMT_INCOME_TOTAL"]["psi"] > 0
    built = reference_from_frame(reference_frame, "test", "jeu de test")
    assert built["numeric"]["AMT_INCOME_TOTAL"]["kind"] == "deciles"


def test_cli_report(tmp_path: Path):
    from afrorisk.monitoring.drift import main

    reference = {
        "data_source": "test",
        "n_rows": 100,
        "numeric": {
            "AMT_INCOME_TOTAL": {
                "kind": "deciles",
                "n": 100,
                "edges": [0, 50, 100],
                "counts": [50, 50],
            }
        },
        "categorical": {},
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    recent = tmp_path / "recent.csv"
    from afrorisk.config import SERVED_CATEGORICAL, SERVED_NUMERIC

    row = {column: 40 for column in SERVED_NUMERIC}
    row.update({column: "Cash loans" for column in SERVED_CATEGORICAL})
    pd.DataFrame([row] * 8).to_csv(recent, index=False)
    output = tmp_path / "report.json"
    code = main(["report", "--recent", str(recent), "--reference", str(reference_path), "--output", str(output)])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["variables"][0]["feature"] == "AMT_INCOME_TOTAL"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_drift_endpoint_rejects_a_short_batch(client):
    record = {
        "amt_income_total": 120000,
        "amt_credit": 300000,
        "amt_annuity": 18000,
        "amt_goods_price": 280000,
        "age_years": 40,
        "employment_years": 5,
        "ext_source_2": 0.5,
        "ext_source_3": 0.4,
        "name_contract_type": "Cash loans",
        "name_education_type": "Secondary / secondary special",
        "name_family_status": "Married",
        "name_income_type": "Working",
        "cnt_children": 0,
        "flag_own_realty": "Y",
        "flag_own_car": "N",
        "region_rating_client": 2,
    }
    response = client.post("/drift", json={"records": [record] * 5})
    assert response.status_code == 422


def test_drift_endpoint_returns_a_report(client):
    record = {
        "amt_income_total": 120000,
        "amt_credit": 300000,
        "amt_annuity": 18000,
        "amt_goods_price": 280000,
        "age_years": 40,
        "employment_years": 5,
        "ext_source_2": 0.5,
        "ext_source_3": 0.4,
        "name_contract_type": "Cash loans",
        "name_education_type": "Secondary / secondary special",
        "name_family_status": "Married",
        "name_income_type": "Working",
        "cnt_children": 0,
        "flag_own_realty": "Y",
        "flag_own_car": "N",
        "region_rating_client": 2,
    }
    response = client.post("/drift", json={"records": [record] * 20})
    assert response.status_code == 200
    body = response.json()
    assert body["n_recent"] == 20
    assert "variables" in body
    names = {item["feature"] for item in body["variables"]}
    assert "AGE_YEARS" in names
    assert "CODE_GENDER" not in names
