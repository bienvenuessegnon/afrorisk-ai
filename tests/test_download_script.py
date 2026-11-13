"""Le script de téléchargement refuse de partir sans jeton."""

import subprocess
from pathlib import Path


def test_download_requires_token(monkeypatch):
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    script = Path("scripts/download_data.sh")
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "KAGGLE_API_TOKEN" in result.stdout + result.stderr
