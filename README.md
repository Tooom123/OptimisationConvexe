# Challenge Optimisation de Portefeuille

## Objectif

Construire un modèle qui prédit les retours de 12 actifs financiers et optimise les poids du portefeuille pour maximiser le Sharpe ratio.

## Structure du projet

```
.
├── data/
│   ├── X_train.parquet        # Features d'entraînement
│   ├── R_train.parquet        # Retours réels (vérité terrain)
│   └── X_test.parquet         # Features de test
│
├── models/
│   ├── model_Tom.py           # Modèle de Tom (référence)
│   ├── model_Aymeric.py       # ← TON FICHIER (Aymeric)
│   ├── model_Artus.py         # ← TON FICHIER (Artus)
│   └── model_Estebane.py      # ← TON FICHIER (Estebane)
│
├── portfolio_model.py         # Exemple complet avec Adam + Ledoit-Wolf (à lire !)
├── main.py                    # Lance tous les modèles et affiche le classement
├── evaluate.py                # Calcule Sharpe, CumRet, MSE sur les soumissions
└── prk/                       # Parquets générés automatiquement
```

## Ce que tu dois faire

### 1. Remplir ton fichier `model_Prénom.py`

Ton fichier doit définir une classe `PortfolioChallengeModel` avec **obligatoirement** deux méthodes :

```python
import numpy as np
import pandas as pd
from pathlib import Path

class PortfolioChallengeModel:

    def fit(self, x_train: pd.DataFrame, r_train: pd.DataFrame):
        """Entraîne ton modèle sur les données historiques."""
        ...

    def predict_returns(self, x_df: pd.DataFrame) -> pd.DataFrame:
        """
        Retourne un DataFrame avec les colonnes pred_asset_1 … pred_asset_12,
        même index que x_df.
        """
        ...
```

Les méthodes `load_data`, `build_weights`, `build_submission`, `save_submission` et `fit_predict_save` sont **héritées de la classe de base** — tu n'as pas à les réécrire. Si tu veux les surcharger, tu peux.

**Interface complète attendue** (cf. `portfolio_model.py` pour un exemple concret) :

| Méthode | Rôle |
|---|---|
| `fit(x_train, r_train)` | Apprentissage |
| `predict_returns(x_df)` | Prédiction des retours (colonnes `pred_asset_i`) |
| `build_weights(pred_df)` | *(optionnel)* Construction des poids (`weight_asset_i`), contrainte `sum(|w|) ≤ 1` |

### 2. Lancer le classement

```bash
python main.py
```

Cela va :
1. Détecter automatiquement les fichiers non-vides
2. Entraîner chaque modèle et générer `prk/model_Prénom.parquet`
3. Afficher le classement du groupe avec Sharpe, CumRet, MSE

### 3. Métriques évaluées

| Métrique | Description | Objectif |
|---|---|---|
| **Sharpe annualisé** | rendement moyen / volatilité × √8760 | **Maximiser** |
| CumRet% | Retour cumulatif sur la période train | Maximiser |
| MSE | Erreur quadratique prédictions vs retours réels | Minimiser |
| Gross exposure | Somme des valeurs absolues des poids (≤ 1 requis) | Contrainte |

### 4. Contraintes sur les poids

```
∀t :  Σ |weight_asset_i(t)| ≤ 1   et   |weight_asset_i(t)| ≤ 1
```

Les soumissions qui violent ces contraintes sont signalées par un `!` dans le tableau.

## Pour démarrer

Regarde `portfolio_model.py` : il implémente un pipeline complet avec régression linéaire (Adam), régularisation L2 et optimisation de portefeuille via covariance de Ledoit-Wolf. Tu peux t'en inspirer ou partir d'une approche totalement différente.

Idées d'exploration :
- Modèles de régression différents (Ridge, Lasso, arbres…)
- Features engineering sur `X_train`
- Stratégies de pondération alternatives (equal weight, mean-variance…)
- Fenêtres temporelles glissantes

## Dépendances

```bash
pip install -r requirements.txt
```
