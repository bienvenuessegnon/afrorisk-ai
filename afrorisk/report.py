"""Écrit reports/metrics.json dans la section Résultats du README."""

import json
from pathlib import Path

from afrorisk.config import METRICS_PATH, MODEL_LABELS, PREVIOUS_HOME_CREDIT, README_PATH

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _table(rows: list[dict]) -> str:
    lines = [
        "| Modèle | ROC-AUC | PR-AUC | Précision | Rappel | F1 | Seuil |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {roc} | {pr} | {precision} | {recall} | {f1} | {threshold} |".format(
                model=MODEL_LABELS.get(row["model"], row["model"]),
                roc=_fmt(row["roc_auc"]),
                pr=_fmt(row["pr_auc"]),
                precision=_fmt(row["precision"]),
                recall=_fmt(row["recall"]),
                f1=_fmt(row["f1"]),
                threshold=_fmt(row["threshold"]),
            )
        )
    return "\n".join(lines)



def _delta(after: float, before: float) -> str:
    gap = after - before
    sign = "+" if gap >= 0 else ""
    return f"{_fmt(after)} ({sign}{_fmt(gap)})"


def _before_after(metrics: dict) -> str:
    if metrics.get("placeholder") or metrics.get("data_source") != "home_credit":
        return ""
    previous = PREVIOUS_HOME_CREDIT
    served = metrics["served"]["metrics"]
    research = metrics["research"]["metrics"]
    lines = [
        "### Comparaison avec la première version",
        "",
        "La première version était évaluée en 3 plis, avec moins de variables dérivées, "
        "et le sexe figurait encore parmi les features du modèle servi.",
        "",
        "| Portée | ROC-AUC | PR-AUC | Précision | Rappel | F1 | Seuil |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope, before, after in (
        ("Modèle servi", previous["served"], served),
        ("Modèle de recherche", previous["research"], research),
    ):
        lines.append(
            "| {scope} | {roc} | {pr} | {precision} | {recall} | {f1} | {threshold} |".format(
                scope=scope,
                roc=_delta(after["roc_auc"], before["roc_auc"]),
                pr=_delta(after["pr_auc"], before["pr_auc"]),
                precision=_delta(after["precision"], before["precision"]),
                recall=_delta(after["recall"], before["recall"]),
                f1=_delta(after["f1"], before["f1"]),
                threshold=f"{before['threshold']:.2f} → {after['threshold']:.2f}",
            )
        )
    return "\n".join(lines)


def render_results(metrics: dict) -> str:
    source = metrics["data_source"]
    if metrics.get("placeholder"):
        banner = (
            "> **Placeholder.** Ces chiffres viennent de la fixture synthétique "
            f"(`{source}`, {metrics['n_rows']} demandes, taux de défaut "
            f"{metrics['positive_rate']:.1%}). Ils ne mesurent pas Home Credit "
            "et ne valent pas une performance réelle."
        )
    else:
        banner = (
            "> Entraînement sur **Home Credit Default Risk** "
            f"({metrics['n_rows']} demandes, taux de défaut {metrics['positive_rate']:.1%}). "
            "Usage pédagogique : ce n'est pas une décision d'octroi."
        )

    served = metrics["served"]
    research = metrics["research"]
    matrix = served["metrics"]["confusion_matrix"]
    gender_note = (
        "Le modèle servi n'utilise pas le sexe (`CODE_GENDER`) : la variable a été retirée "
        "du formulaire, de l'API et des features exposées."
        if metrics.get("served_excludes_code_gender", True) and not metrics.get("placeholder")
        else ""
    )
    comparison = _before_after(metrics)
    parts = [
        START,
        banner,
        "",
        f"Validation croisée stratifiée : {metrics['cv_folds']} plis. "
        "Le modèle retenu dans chaque portée est celui qui a le ROC-AUC hors échantillon "
        "le plus élevé. Précision, rappel et F1 sont calculés au seuil qui minimise "
        f"le coût {metrics['cost_fn']:.0f} × faux négatifs + {metrics['cost_fp']:.0f} × faux positifs.",
        "",
        gender_note,
        "",
        comparison,
        "",
        "### Modèle servi (variables du formulaire)",
        "",
        f"Modèle retenu : **{MODEL_LABELS.get(served['selected_model'], served['selected_model'])}**. "
        f"Seuil de décision : **{served['threshold']:.2f}**.",
        "",
        _table(served["comparison"]),
        "",
        "Matrice de confusion du modèle servi, au seuil retenu, sur les prédictions hors pli "
        f"(lignes = réel, colonnes = prédit, ordre non-défaut / défaut) : `{matrix}`.",
        "",
        "### Modèle de recherche (application + agrégations)",
        "",
        f"Modèle retenu : **{MODEL_LABELS.get(research['selected_model'], research['selected_model'])}**.",
        "",
        _table(research["comparison"]),
        "",
        "Les facteurs SHAP globaux du modèle servi sont dans `reports/metrics.json` "
        "(`served.global_factors`) et sur le tableau de bord.",
        END,
    ]
    rendered = "\n".join(part for part in parts if part is not None)
    while "\n\n\n" in rendered:
        rendered = rendered.replace("\n\n\n", "\n\n")
    return rendered + "\n"



def update_readme(metrics: dict, readme_path: Path | None = None) -> None:
    path = readme_path or README_PATH
    block = render_results(metrics)
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise RuntimeError("Balises RESULTS absentes du README.")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    path.write_text(before + block + "\n" + after.lstrip("\n"), encoding="utf-8")


def write_metrics(metrics: dict, path: Path | None = None) -> Path:
    destination = path or METRICS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination
