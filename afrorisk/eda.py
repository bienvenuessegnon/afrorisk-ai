"""Analyse exploratoire : résumé chiffré et figures."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from afrorisk.config import EMPLOYMENT_SENTINEL, FIGURES, REPORTS
from afrorisk.features import add_application_features
from afrorisk.io import load_tables


def build_summary(application: pd.DataFrame) -> dict:
    featured = add_application_features(application)
    missing = application.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(15)
    return {
        "rows": int(len(application)),
        "columns": int(application.shape[1]),
        "duplicate_ids": int(application["SK_ID_CURR"].duplicated().sum()),
        "target_rate": float(application["TARGET"].mean()),
        "target_counts": {
            str(key): int(value) for key, value in application["TARGET"].value_counts().to_dict().items()
        },
        "employment_sentinel": int((application["DAYS_EMPLOYED"] == EMPLOYMENT_SENTINEL).sum()),
        "age_mean": float(featured["AGE_YEARS"].mean()),
        "missing_top": {key: round(float(value), 4) for key, value in missing.items()},
    }


def save_figures(application: pd.DataFrame, source: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    featured = add_application_features(application)
    featured["TARGET"] = application["TARGET"].to_numpy()
    sns.set_theme(style="whitegrid", font_scale=0.9)
    caption = "fixture synthétique" if source != "home_credit" else "Home Credit"

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = application["TARGET"].value_counts().sort_index()
    ax.bar(["Non-défaut", "Défaut"], [counts.get(0, 0), counts.get(1, 0)], color=["#1e3a56", "#8c6a3d"])
    ax.set_title(f"Répartition de la cible ({caption})")
    ax.set_ylabel("Demandes")
    fig.tight_layout()
    fig.savefig(FIGURES / "target_distribution.png", dpi=120)
    plt.close(fig)

    missing = application.isna().mean().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    missing.sort_values().plot.barh(ax=ax, color="#1e3a56")
    ax.set_title(f"Part de valeurs manquantes ({caption})")
    ax.set_xlabel("Proportion")
    fig.tight_layout()
    fig.savefig(FIGURES / "missing_values.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.boxplot(data=featured, x="TARGET", y="AGE_YEARS", ax=ax, color="#d9e2ec")
    ax.set_title(f"Âge selon la cible ({caption})")
    ax.set_xlabel("TARGET")
    ax.set_ylabel("Âge (années)")
    fig.tight_layout()
    fig.savefig(FIGURES / "age_vs_target.png", dpi=120)
    plt.close(fig)

    melted = featured[["TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].melt(
        id_vars="TARGET", var_name="score", value_name="valeur"
    )
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.boxplot(data=melted, x="score", y="valeur", hue="TARGET", ax=ax)
    ax.set_title(f"Scores externes selon la cible ({caption})")
    fig.tight_layout()
    fig.savefig(FIGURES / "ext_source_vs_target.png", dpi=120)
    plt.close(fig)

    columns = [
        "AGE_YEARS",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "CREDIT_INCOME_RATIO",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "TARGET",
    ]
    corr = featured[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, annot_kws={"size": 8})
    ax.set_title(f"Corrélations ({caption})")
    fig.tight_layout()
    fig.savefig(FIGURES / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    summary = build_summary(application)
    summary["source"] = source
    (REPORTS / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    tables, source = load_tables()
    summary = build_summary(tables["application"])
    save_figures(tables["application"], source)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Figures écrites dans {FIGURES}")


if __name__ == "__main__":
    main()
