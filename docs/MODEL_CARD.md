# Carte du modèle

AfroRisk estime une probabilité de défaut et en détaille les facteurs. Les chiffres ci-dessous sont ceux de `reports/metrics.json` (entraînement du 12 novembre 2026). Aucune performance par sous-groupe n'est publiée : les prédictions hors pli n'ont pas été archivées, et les recalculer demanderait le jeu Home Credit complet.

## Usage prévu

Lecture d'un dossier de crédit, en formation ou en démonstration. Le résultat est une probabilité, une classe au seuil retenu, un niveau de risque et les facteurs qui poussent le score.

## Usages exclus

- Décider d'accorder ou de refuser un crédit.
- Produire une note réglementaire ou un motif opposable.
- Appliquer le score tel quel à une clientèle qui n'est pas celle de Home Credit.
- Utiliser le sexe comme critère. Il est absent du modèle servi, du formulaire et de l'API.

## Données

Home Credit Default Risk, table `application_train` : 307 511 demandes, taux de défaut 8,07 % (0,08072881945686496). Le modèle de recherche ajoute les agrégations de `bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `installments_payments` et `credit_card_balance`, ainsi que des variables de dossier (logement, profession, organisme). Ces agrégations n'utilisent pas la cible. Le modèle servi se limite aux saisies du formulaire et aux ratios calculés côté serveur (35 variables). Il n'utilise pas `CODE_GENDER`.

## Évaluation

Validation croisée stratifiée à 5 plis. Le modèle retenu dans chaque portée est celui dont le ROC-AUC hors pli est le plus élevé. La précision, le rappel, le F1 et la matrice de confusion sont calculés au seuil qui minimise le coût `5 × faux négatifs + 1 × faux positifs` sur ces probabilités hors pli.

### Modèle servi

LightGBM, seuil 0,70.

| Indicateur | Valeur |
| --- | ---: |
| ROC-AUC | 0,7622 |
| PR-AUC | 0,2459 |
| Précision | 0,2674 |
| Rappel | 0,3594 |
| F1 | 0,3067 |

Matrice de confusion hors pli, ordre non-défaut puis défaut : `[[258244, 24442], [15902, 8923]]`.

| Modèle | ROC-AUC | PR-AUC | Précision | Rappel | F1 | Seuil |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0,7622 | 0,2459 | 0,2674 | 0,3594 | 0,3067 | 0,70 |
| XGBoost | 0,7618 | 0,2458 | 0,2673 | 0,3590 | 0,3064 | 0,70 |
| Forêt aléatoire | 0,7455 | 0,2274 | 0,2472 | 0,3524 | 0,2906 | 0,65 |
| Régression logistique | 0,7445 | 0,2245 | 0,2504 | 0,3369 | 0,2873 | 0,70 |
| Arbre de décision | 0,7225 | 0,2057 | 0,2250 | 0,3628 | 0,2778 | 0,70 |

### Modèle de recherche

LightGBM, moyenne de germes, seuil 0,65. Ce modèle n'est pas exposé par l'API.

| Indicateur | Valeur |
| --- | ---: |
| ROC-AUC | 0,7928 |
| PR-AUC | 0,2886 |
| Précision | 0,2738 |
| Rappel | 0,4567 |
| F1 | 0,3424 |

Matrice de confusion hors pli : `[[252617, 30069], [13487, 11338]]`.

| Modèle | ROC-AUC | PR-AUC | Précision | Rappel | F1 | Seuil |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM (moyenne de germes) | 0,7928 | 0,2886 | 0,2738 | 0,4567 | 0,3424 | 0,65 |
| LightGBM | 0,7908 | 0,2854 | 0,2699 | 0,4597 | 0,3402 | 0,65 |
| XGBoost | 0,7904 | 0,2835 | 0,2603 | 0,4848 | 0,3388 | 0,65 |
| Régression logistique | 0,7754 | 0,2576 | 0,2694 | 0,4024 | 0,3227 | 0,70 |
| Forêt aléatoire | 0,7592 | 0,2420 | 0,2391 | 0,4226 | 0,3054 | 0,60 |
| Arbre de décision | 0,7242 | 0,2034 | 0,2235 | 0,3668 | 0,2777 | 0,70 |

## Seuil

Le coût traité un faux négatif (défaut non signalé) comme cinq fois plus lourd qu'un faux positif (bon dossier signalé à tort). Ce rapport est un choix pédagogique, pas un barème d'établissement. Le seuil du modèle servi est 0,70. En dessous de la moitié de ce seuil le niveau affiché est faible, entre la moitié et le seuil il est modéré, au-delà il est élevé.

## Équité

Le sexe a été retiré du modèle servi. Il reste disponible dans le modèle de recherche, uniquement pour l'analyse, et n'entre dans aucune prédiction exposée. L'absence d'une variable sensible ne garantit pas l'absence de biais : le revenu, la profession, la région ou les scores externes peuvent la reprendre indirectement. Aucun écart par tranche d'âge ou par type de contrat n'est chiffré ici, faute de prédictions hors pli conservées.

## Limites et risques

- Home Credit décrit une autre clientèle et une autre période. Le score n'est pas transposable sans nouvel apprentissage et sans nouveau seuil.
- SHAP décrit le modèle. Un facteur qui augmente le score n'est pas une cause établie.
- Les scores externes sont opaques. Là où il n'existe pas de bureau de crédit, il faut les remplacer par des signaux locaux, pas les imputer au hasard.
- Le rappel du modèle servi, au seuil de coût, reste d'environ 0,36 : une majorité des défauts observés dans la validation n'est pas classée en défaut.
- La référence de dérive (`models/served/reference_stats.json`) repose sur 200 lignes de l'échantillon d'entraînement conservé avec le modèle, pas sur les 307 511 lignes. Elle sert à un contrôle de stabilité, pas à une mesure définitive.

## Suivi

Le PSI compare un lot récent à cette référence. En dessous de 0,1 la variable est stable, entre 0,1 et 0,25 l'écart est à surveiller, au-delà de 0,25 la dérive est marquée. Voir `afrorisk/monitoring/drift.py`.
