"""Indice de stabilité de population (PSI) entre une référence et un lot récent.

Les seuils usuels sont 0,1 (écart à surveiller) et 0,25 (dérive marquée).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from afrorisk.config import MODELS_SERVED, SERVED_CATEGORICAL, SERVED_NUMERIC

PSI_STABLE = 0.1
PSI_SHIFT = 0.25
EPSILON = 1e-6
REFERENCE_PATH = MODELS_SERVED / "reference_stats.json"


def psi_from_counts(reference_counts: list[float], recent_counts: list[float]) -> float:
    reference = np.asarray(reference_counts, dtype=float)
    recent = np.asarray(recent_counts, dtype=float)
    if reference.shape != recent.shape or reference.size == 0:
        raise ValueError("Les deux répartitions doivent avoir le même nombre de classes.")
    reference = np.clip(reference, 0, None)
    recent = np.clip(recent, 0, None)
    if reference.sum() <= 0 or recent.sum() <= 0:
        return float("nan")
    reference = reference / reference.sum()
    recent = recent / recent.sum()
    reference = np.clip(reference, EPSILON, None)
    recent = np.clip(recent, EPSILON, None)
    reference = reference / reference.sum()
    recent = recent / recent.sum()
    return float(np.sum((recent - reference) * np.log(recent / reference)))


def stability_label(psi: float) -> str:
    if not np.isfinite(psi):
        return "non comparable"
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_SHIFT:
        return "à surveiller"
    return "dérive"


def _numeric_profile(values: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 20:
        return {"kind": "insuffisant", "n": int(clean.size)}
    edges = np.unique(np.quantile(clean, np.linspace(0, 1, 11)))
    if edges.size < 3:
        return {"kind": "constante", "value": float(np.median(clean)), "n": int(clean.size)}
    edges = edges.astype(float)
    edges[0] = min(float(edges[0]), float(clean.min()))
    edges[-1] = max(float(edges[-1]), float(clean.max()))
    if not np.all(np.diff(edges) > 0):
        return {"kind": "constante", "value": float(np.median(clean)), "n": int(clean.size)}
    counts, _ = np.histogram(clean, bins=edges)
    return {
        "kind": "deciles",
        "n": int(clean.size),
        "edges": [round(float(edge), 6) for edge in edges],
        "counts": [int(count) for count in counts],
    }


def _categorical_profile(values: pd.Series) -> dict:
    series = values.astype("object").where(values.notna(), "manquant").astype(str)
    counts = series.value_counts()
    return {
        "n": int(series.size),
        "levels": [
            {"level": str(level), "count": int(count)}
            for level, count in counts.items()
        ],
    }


def reference_from_frame(frame: pd.DataFrame, data_source: str, origin: str) -> dict:
    numeric = {}
    for column in SERVED_NUMERIC:
        if column in frame.columns:
            numeric[column] = _numeric_profile(frame[column])
    categorical = {}
    for column in SERVED_CATEGORICAL:
        if column in frame.columns:
            categorical[column] = _categorical_profile(frame[column])
    return {
        "data_source": data_source,
        "n_rows": int(len(frame)),
        "origin": origin,
        "psi_stable_below": PSI_STABLE,
        "psi_shift_below": PSI_SHIFT,
        "numeric": numeric,
        "categorical": categorical,
    }


def reference_from_served_background() -> dict:
    """Reconstitue un échantillon du jeu d'entraînement à partir du fond SHAP."""
    import joblib

    pipeline = joblib.load(MODELS_SERVED / "model.joblib")
    background = np.load(MODELS_SERVED / "background.npy")
    metadata = json.loads((MODELS_SERVED / "metadata.json").read_text(encoding="utf-8"))
    numeric = list(metadata["numeric_features"])
    categorical = list(metadata["categorical_features"])
    preprocess = pipeline.named_steps["preprocess"]
    scaler = preprocess.named_transformers_["num"].named_steps["scaler"]
    encoder = preprocess.named_transformers_["cat"].named_steps["onehot"]
    restored = scaler.inverse_transform(background[:, : len(numeric)])
    columns = {name: restored[:, index] for index, name in enumerate(numeric)}
    offset = len(numeric)
    for name, levels in zip(categorical, encoder.categories_):
        width = len(levels)
        block = background[:, offset : offset + width]
        chosen = block.argmax(axis=1)
        labels = []
        for row, index in enumerate(chosen):
            labels.append(str(levels[index]) if float(block[row].sum()) > 0 else "manquant")
        columns[name] = labels
        offset += width
    frame = pd.DataFrame(columns)
    return reference_from_frame(
        frame,
        data_source=str(metadata.get("data_source", "inconnu")),
        origin=(
            "Déciles estimés sur l'échantillon de 200 lignes conservé avec le modèle servi "
            "(fond utilisé pour SHAP), après inversion de la normalisation. "
            "Ce n'est pas le détail des 307 511 lignes."
        ),
    )


def _recent_numeric_counts(values: pd.Series, edges: list[float]) -> list[int]:
    clean = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return [0] * (len(edges) - 1)
    clipped = np.clip(clean, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=np.asarray(edges, dtype=float))
    return [int(count) for count in counts]


def _recent_categorical_counts(values: pd.Series, levels: list[str]) -> list[int]:
    series = values.astype("object").where(values.notna(), "manquant").astype(str)
    known = set(levels)
    counts = [int((series == level).sum()) for level in levels]
    other = int((~series.isin(known)).sum())
    return counts + [other]


def compare_to_reference(recent: pd.DataFrame, reference: dict) -> dict:
    variables = []
    for name, profile in reference.get("numeric", {}).items():
        if profile.get("kind") != "deciles" or name not in recent.columns:
            variables.append(
                {
                    "feature": name,
                    "kind": "numerique",
                    "psi": None,
                    "status": "non comparable",
                }
            )
            continue
        recent_counts = _recent_numeric_counts(recent[name], profile["edges"])
        psi = psi_from_counts(profile["counts"], recent_counts)
        variables.append(
            {
                "feature": name,
                "kind": "numerique",
                "psi": None if not np.isfinite(psi) else round(psi, 6),
                "status": stability_label(psi),
            }
        )
    for name, profile in reference.get("categorical", {}).items():
        if name not in recent.columns:
            variables.append(
                {
                    "feature": name,
                    "kind": "categorielle",
                    "psi": None,
                    "status": "non comparable",
                }
            )
            continue
        levels = [item["level"] for item in profile["levels"]]
        reference_counts = [item["count"] for item in profile["levels"]] + [0]
        recent_counts = _recent_categorical_counts(recent[name], levels)
        psi = psi_from_counts(reference_counts, recent_counts)
        variables.append(
            {
                "feature": name,
                "kind": "categorielle",
                "psi": None if not np.isfinite(psi) else round(psi, 6),
                "status": stability_label(psi),
            }
        )
    comparable = [item["psi"] for item in variables if isinstance(item["psi"], float)]
    return {
        "n_recent": int(len(recent)),
        "n_reference": int(reference.get("n_rows", 0)),
        "data_source": reference.get("data_source"),
        "psi_stable_below": PSI_STABLE,
        "psi_shift_below": PSI_SHIFT,
        "variables": variables,
        "max_psi": None if not comparable else round(max(comparable), 6),
        "n_shift": sum(item["status"] == "à surveiller" for item in variables),
        "n_drift": sum(item["status"] == "dérive" for item in variables),
    }


def load_reference(path: Path | None = None) -> dict:
    destination = path or REFERENCE_PATH
    if not destination.exists():
        raise FileNotFoundError(f"Référence absente : {destination}")
    return json.loads(destination.read_text(encoding="utf-8"))


def render_markdown(report: dict) -> str:
    lines = [
        "# Stabilité des variables servies",
        "",
        f"Lot récent : {report['n_recent']} lignes. Référence : {report['n_reference']} lignes "
        f"({report.get('data_source', 'inconnue')}).",
        "",
        "Un PSI inférieur à 0,1 est stable. Entre 0,1 et 0,25, l'écart est à surveiller. "
        "Au-delà de 0,25, la dérive est marquée.",
        "",
        "| Variable | Type | PSI | Lecture |",
        "| --- | --- | ---: | --- |",
    ]
    for item in report["variables"]:
        psi = "—" if item["psi"] is None else f"{item['psi']:.3f}"
        lines.append(f"| {item['feature']} | {item['kind']} | {psi} | {item['status']} |")
    lines.append("")
    lines.append(
        f"Variables à surveiller : {report['n_shift']}. Variables en dérive : {report['n_drift']}."
    )
    return "\n".join(lines) + "\n"


def _read_recent(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in SERVED_NUMERIC + SERVED_CATEGORICAL if column not in frame.columns]
    if missing:
        raise SystemExit(
            "Le fichier récent doit contenir les colonnes du modèle servi. Absentes : "
            + ", ".join(missing)
        )
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PSI des variables du modèle servi.")
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="Écrit la référence à partir du fond du modèle servi.")
    fit.add_argument("--output", type=Path, default=REFERENCE_PATH)

    report = sub.add_parser("report", help="Compare un lot CSV à la référence.")
    report.add_argument("--recent", type=Path, required=True)
    report.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    report.add_argument("--format", choices=("json", "markdown"), default="json")
    report.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "fit":
        payload = reference_from_served_background()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Référence écrite dans {args.output} ({payload['n_rows']} lignes).")
        return 0

    reference = load_reference(args.reference)
    result = compare_to_reference(_read_recent(args.recent), reference)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n" if args.format == "json" else render_markdown(result)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
