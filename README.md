# AfroRisk AI

Estimation du risque de défaut de crédit, avec les facteurs qui poussent le score vers le haut ou vers le bas.

AfroRisk AI — Turning financial data into explainable risk intelligence.

Le service sert à lire un dossier. Il ne décide pas d'accorder ou de refuser un crédit.

## Architecture

```text
data/raw ou data/fixtures
        │
        ▼
agrégations (bureau, bureau_balance, previous_application,
             POS_CASH_balance, installments_payments, credit_card_balance)
        │
        ├── modèle de recherche (toutes les variables construites)
        └── modèle servi (formulaire, ~18 saisies + ratios)
                │
                ▼
             SHAP  →  FastAPI  →  interface React
```

| Couche | Rôle |
| --- | --- |
| Données | CSV Home Credit, ou fixture synthétique si `data/raw` est vide |
| Variables | Nettoyage, ratios, agrégations historiques connues à la date de la demande |
| Modèles | Régression logistique, arbre, forêt aléatoire, XGBoost, LightGBM |
| Explications | SHAP global (échantillon) et SHAP local (un dossier) |
| API | FastAPI : `/predict`, `/health`, `/metrics-summary`, `/docs` |
| Interface | React (Vite) : indicateurs, formulaire, jauge, facteurs, historique local |

MLflow enregistre les métriques de chaque comparaison dans `mlruns/` (non versionné). L'historique des simulations reste dans le navigateur (`localStorage`) : pas de base PostgreSQL.

## Résultats

<!-- RESULTS:START -->
> Entraînement sur **Home Credit Default Risk** (307511 demandes, taux de défaut 8.1%). Usage pédagogique : ce n'est pas une décision d'octroi.

Validation croisée stratifiée : 3 plis. Le modèle retenu dans chaque portée est celui qui a le ROC-AUC hors échantillon le plus élevé. Précision, rappel et F1 sont calculés au seuil qui minimise le coût 5 × faux négatifs + 1 × faux positifs.

### Modèle servi (variables du formulaire)

Modèle retenu : **LightGBM**. Seuil de décision : **0.70**.

| Modèle | ROC-AUC | PR-AUC | Précision | Rappel | F1 | Seuil |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.755 | 0.238 | 0.261 | 0.351 | 0.299 | 0.700 |
| XGBoost | 0.754 | 0.238 | 0.261 | 0.349 | 0.299 | 0.700 |
| Forêt aléatoire | 0.747 | 0.228 | 0.230 | 0.409 | 0.294 | 0.600 |
| Régression logistique | 0.742 | 0.221 | 0.249 | 0.333 | 0.284 | 0.700 |
| Arbre de décision | 0.722 | 0.204 | 0.226 | 0.341 | 0.272 | 0.700 |

Matrice de confusion du modèle servi, au seuil retenu, sur les prédictions hors pli (lignes = réel, colonnes = prédit, ordre non-défaut / défaut) : `[[257955, 24731], [16111, 8714]]`.

### Modèle de recherche (application + agrégations)

Modèle retenu : **LightGBM**.

| Modèle | ROC-AUC | PR-AUC | Précision | Rappel | F1 | Seuil |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.776 | 0.264 | 0.249 | 0.464 | 0.324 | 0.650 |
| XGBoost | 0.773 | 0.262 | 0.249 | 0.457 | 0.322 | 0.650 |
| Régression logistique | 0.761 | 0.241 | 0.259 | 0.370 | 0.305 | 0.700 |
| Forêt aléatoire | 0.760 | 0.239 | 0.258 | 0.366 | 0.303 | 0.600 |
| Arbre de décision | 0.721 | 0.202 | 0.254 | 0.245 | 0.249 | 0.750 |

Les facteurs SHAP globaux du modèle servi sont dans `reports/metrics.json` (`served.global_factors`) et sur le tableau de bord.
<!-- RESULTS:END -->

Les figures d'exploration sont produites dans `reports/figures/` par la même commande. Le détail machine est dans `reports/metrics.json` et `reports/eda_summary.json`.

## Variables du formulaire

Le modèle servi ne reprend pas les centaines de colonnes brutes. L'analyste saisit :

| Saisie | Colonne d'origine |
| --- | --- |
| Revenu annuel | `AMT_INCOME_TOTAL` |
| Montant du crédit | `AMT_CREDIT` |
| Annuité | `AMT_ANNUITY` |
| Prix du bien | `AMT_GOODS_PRICE` |
| Âge | `DAYS_BIRTH` |
| Ancienneté professionnelle | `DAYS_EMPLOYED` (la valeur sentinelle 365243 devient manquante) |
| Scores externes 1, 2 et 3 | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` |
| Situation familiale | `NAME_FAMILY_STATUS` |
| Niveau d'études | `NAME_EDUCATION_TYPE` |
| Type de contrat | `NAME_CONTRACT_TYPE` |
| Type de revenu | `NAME_INCOME_TYPE` |
| Nombre d'enfants | `CNT_CHILDREN` |
| Propriétaire immobilier | `FLAG_OWN_REALTY` |
| Propriétaire d'un véhicule | `FLAG_OWN_CAR` |
| Note de la région (1 à 3) | `REGION_RATING_CLIENT` |
| Sexe, facultatif | `CODE_GENDER` |

Le pipeline ajoute, sans les redemander : âge en années, ancienneté en années, moyenne des scores externes, crédit/revenu, annuité/revenu, prix du bien/crédit.

Le modèle de recherche ajoute les agrégations des six tables secondaires (volumes, retards, taux de refus, utilisation de carte, etc.) ainsi que le logement et la profession. Ces variables ne sont pas au formulaire : elles ne sont pas connues d'un utilisateur qui ne dispose que du dossier courant.

Le sexe est facultatif. S'il est omis, la modalité `Non renseigné` est utilisée. Voir la section éthique.

## Ce qui est livré dans le dépôt

| Élément | Statut |
| --- | --- |
| `data/fixtures/` | Jeu synthétique, même colonnes que celles lues sur Home Credit. Ce n'est pas le concours Kaggle. |
| `data/raw/` | Vide et ignoré par git. Rempli par `scripts/download_data.sh`. |
| `models/served/` | Modèle de démonstration entraîné sur la fixture, tant que `metadata.json` indique `"placeholder": true`. |
| `models/research/` | Recréé par l'entraînement, non versionné. |
| `reports/metrics.json` | Écrit par l'entraînement. Tant que `placeholder` est vrai, les chiffres ne sont pas des performances Home Credit. |

Après un entraînement sur les CSV Kaggle, `placeholder` passe à `false` et cette section Résultats est réécrite par le script. Il ne faut pas y coller de chiffres à la main.

## Installation

Python 3.12 et Node.js 22.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install
```

## Données Kaggle

Le jeton attendu est le jeton API récent (`KAGGLE_API_TOKEN`, souvent préfixé par `KGAT`), pas l'ancien couple utilisateur / clé.

1. Créer un compte Kaggle et accepter les règles de la compétition [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).
2. Créer un jeton dans les paramètres du compte.
3. Lancer :

```bash
export KAGGLE_API_TOKEN=KGAT_...
bash scripts/download_data.sh
```

Le script exige `kaggle` ≥ 1.7 (`pip install 'kaggle>=1.7.0'`). Il télécharge l'archive dans `data/raw` et la décompresse. Les CSV ne sont pas commités.

Sans ces fichiers, `make train` utilise `data/fixtures`. Pour régénérer la fixture :

```bash
python -m afrorisk.synthetic
```

## Entraînement

Une commande suffit. Elle détecte `data/raw/application_train.csv`, compare les cinq modèles en validation croisée stratifiée, choisit le meilleur ROC-AUC pour la recherche et pour le service, recalcule un seuil de coût, écrit les explications SHAP, met à jour `reports/metrics.json` et la section Résultats de ce fichier, puis trace les runs MLflow.

```bash
make train
# équivalent
python -m afrorisk.train
```

Le seuil minimise `5 × faux négatifs + 1 × faux positifs` sur les probabilités hors pli. La précision, le rappel, le F1 et la matrice de confusion du rapport sont calculés à ce seuil. Le ROC-AUC et le PR-AUC n'en dépendent pas.

Réglages utiles sur le jeu complet (plusieurs Go de RAM, tables `bureau_balance` et `installments_payments` volumineuses) :

```bash
AFRORISK_FAST=1 AFRORISK_MAX_ROWS=80000 python -m afrorisk.train
AFRORISK_CV_FOLDS=5 python -m afrorisk.train
```

`AFRORISK_FAST=1` réduit le nombre d'arbres. `AFRORISK_MAX_ROWS` tire un échantillon stratifié. Sans ces variables, un petit jeu (la fixture, ou moins de 8 000 lignes) utilise déjà le mode court ; le jeu complet utilise des modèles plus longs et 3 plis au-delà de 20 000 lignes.

Consulter MLflow :

```bash
mlflow ui --backend-store-uri mlruns
```

L'exploration seule :

```bash
python -m afrorisk.eda
# ou le notebook notebooks/01_eda.ipynb
```

## Tests

```bash
pytest
cd frontend && npm run build
```

Les tests chargent la fixture, vérifient que la cible ne modifie pas les variables, et interrogent l'API (validation Pydantic, prédiction, `/health`, `/docs`, `/metrics-summary`).

## Lancer l'application

```bash
uvicorn afrorisk.api.main:app --host 0.0.0.0 --port 8000
cd frontend && npm run dev
```

L'interface lit `VITE_API_URL` (défaut `http://localhost:8000`). La documentation interactive est sur [http://localhost:8000/docs](http://localhost:8000/docs).

Endpoints :

| Méthode | Chemin | Rôle |
| --- | --- | --- |
| GET | `/health` | Disponibilité du modèle |
| GET | `/metrics-summary` | Indicateurs pour le tableau de bord |
| POST | `/predict` | Probabilité, classe, niveau de risque, principaux facteurs SHAP |
| GET | `/docs` | Schéma OpenAPI |

`CORS_ORIGINS` est une liste séparée par des virgules. Défaut : `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:8080`, `http://127.0.0.1:8080`.

Docker local (API sur le port 8000, interface sur le port 8080) :

```bash
docker compose up --build
```

## Déploiement

L'API embarque le modèle versionné dans `models/served` et `reports/metrics.json`. Un entraînement sur Home Credit doit donc être fait en local (ou sur une machine qui a les CSV), puis ces artefacts doivent être commités ou fournis à l'image avant le déploiement. Le tier gratuit de Render n'est pas un endroit pour réentraîner sur 300 000 lignes.

### API sur Render

1. Pousser le dépôt. Le fichier `render.yaml` décrit un service web Docker.
2. Créer un Blueprint Render pointant sur ce dépôt, ou un Web Service dont le runtime est Docker, avec le `Dockerfile` à la racine.
3. Définir `CORS_ORIGINS` avec l'URL exacte du front (par exemple `https://afrorisk.vercel.app`), sans barre oblique finale.
4. Render injecte `PORT`. L'image le respecte. Le contrôle de santé est `/health`.
5. Vérifier `https://<service>.onrender.com/health` puis `/docs`.

### Alternative : Hugging Face Spaces (Docker)

1. Créer un Space SDK Docker.
2. Utiliser le `Dockerfile` de la racine. Le port par défaut de l'image est `7860`, celui attendu par Spaces.
3. Dans les secrets du Space, poser `CORS_ORIGINS` avec l'origine du front.
4. Le Space sert l'API, pas l'interface React.

### Interface sur Vercel

1. Importer le dépôt. Répertoire racine du projet : `frontend`.
2. Framework Vite. Commande de build : `npm run build`. Dossier de sortie : `dist`.
3. Variable d'environnement `VITE_API_URL` = URL publique de l'API (`https://<service>.onrender.com`). Elle est lue au build.
4. `frontend/vercel.json` renvoie les routes de l'application sur `index.html`.
5. Redéployer le front après chaque changement de `VITE_API_URL`.

## Éthique et limites

- Démonstration et formation. Pas un outil d'octroi.
- Home Credit décrit une clientèle et une période qui ne sont pas celles du Bénin ni de l'Afrique de l'Ouest. Un score transporté tel quel est une extrapolation.
- Tant que le modèle est marqué placeholder, il n'a vu que des lignes synthétiques. Ses probabilités n'ont aucune portée opérationnelle.
- Le sexe est une variable sensible. Le formulaire permet de ne pas le renseigner. Même renseigné, il ne doit pas servir de motif de refus : les écarts qu'il capte reprennent souvent des biais historiques du jeu d'apprentissage.
- SHAP décrit le modèle, pas une cause. Un facteur « augmente le risque » au sens du score, pas au sens d'un mécanisme économique prouvé.
- Le seuil 5 / 1 est un choix pédagogique (rater un défaut coûte plus cher que refuser un bon dossier). Un établissement doit le fixer avec ses propres coûts, et revoir le calibrage des probabilités.
- Les tables secondaires sont agrégées comme de l'information déjà connue à la date de la demande (convention Home Credit : `DAYS_*` et `MONTHS_BALANCE` sont relatifs à cette date). La cible n'entre dans aucune variable. Cela n'empêche pas d'autres biais : population sélectionnée, scores externes opaques, valeurs manquantes informatives.
- Les montants extrêmes sont bornés aux percentiles 0,5 et 99,5 appris sur l'échantillon d'entraînement.

## Disposition du dépôt

```text
afrorisk/           pipeline, entraînement, API
data/fixtures/      jeu synthétique versionné
frontend/           interface React
models/served/      modèle exposé par l'API
notebooks/          exploration
reports/            métriques et figures
scripts/download_data.sh
tests/
Dockerfile          API (Render et Hugging Face)
render.yaml
docker-compose.yml
```
