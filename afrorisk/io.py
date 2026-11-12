"""Chargement des tables Home Credit, ou de la fixture synthétique."""

from pathlib import Path

import pandas as pd

from afrorisk.config import DATA_FIXTURES, DATA_RAW
from afrorisk.schema import TABLE_COLUMNS, TABLE_FILES


def resolve_data_dir() -> tuple[Path, str]:
    """Retourne le dossier et la source : home_credit si les CSV bruts sont là."""
    marker = DATA_RAW / TABLE_FILES["application"]
    if marker.exists():
        return DATA_RAW, "home_credit"
    return DATA_FIXTURES, "synthetic_fixture"


def load_tables(directory: Path | None = None) -> tuple[dict[str, pd.DataFrame], str]:
    if directory is None:
        directory, source = resolve_data_dir()
    else:
        source = "home_credit" if directory.resolve() == DATA_RAW.resolve() else "custom"
        if directory.resolve() == DATA_FIXTURES.resolve():
            source = "synthetic_fixture"

    tables: dict[str, pd.DataFrame] = {}
    for key, filename in TABLE_FILES.items():
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier manquant : {path}. "
                "Téléchargez les données avec scripts/download_data.sh "
                "ou créez la fixture avec python -m afrorisk.synthetic."
            )
        tables[key] = pd.read_csv(path, usecols=TABLE_COLUMNS[key])
    return tables, source
