#!/usr/bin/env bash
# Télécharge Home Credit Default Risk dans data/raw.
# Jeton récent : https://www.kaggle.com/settings → Create API Token.
#   export KAGGLE_API_TOKEN=KGAT_...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/data/raw"
mkdir -p "${DEST}"

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN est absent."
  echo "Créez un jeton API sur https://www.kaggle.com/settings puis :"
  echo "  export KAGGLE_API_TOKEN=KGAT_..."
  exit 1
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Le client Kaggle n'est pas installé. Exemple : pip install 'kaggle>=1.7.0'"
  exit 1
fi

# Le client lit KAGGLE_API_TOKEN. On laisse aussi une copie locale au format attendu
# par certaines versions, sans l'afficher.
TOKEN_DIR="${HOME}/.kaggle"
mkdir -p "${TOKEN_DIR}"
umask 077
printf '%s' "${KAGGLE_API_TOKEN}" > "${TOKEN_DIR}/access_token"
chmod 600 "${TOKEN_DIR}/access_token"

echo "Téléchargement de la compétition home-credit-default-risk vers ${DEST}"
kaggle competitions download -c home-credit-default-risk -p "${DEST}"
unzip -o "${DEST}/home-credit-default-risk.zip" -d "${DEST}"
echo "Fichiers extraits dans ${DEST}"
echo "Acceptez au préalable les règles de la compétition sur Kaggle si le client les réclame."
