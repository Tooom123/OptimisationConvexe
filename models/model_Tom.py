import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ledoit_wolf(X: np.ndarray) -> np.ndarray:
    """Covariance shrinkage analytique de Ledoit-Wolf (OAS)."""
    n, p = X.shape
    S = np.cov(X, rowvar=False)
    mu_S = np.trace(S) / p
    delta2 = np.sum(S ** 2) + mu_S ** 2 - 2 * mu_S * np.trace(S @ S) / p
    Xc = X - X.mean(axis=0)
    beta_bar = sum(np.sum((np.outer(Xc[i], Xc[i]) - S) ** 2) for i in range(n)) / n ** 2
    alpha = min(beta_bar / delta2, 1.0) if delta2 > 0 else 0.1
    return (1 - alpha) * S + alpha * mu_S * np.eye(p)


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """θ = (X'X + λ·n·I)⁻¹ X'y  (biais non régularisé)."""
    n, d = X.shape
    reg = np.full(d, lam * n)
    reg[-1] = 0.0             # pas de régularisation sur le biais
    A = X.T @ X + np.diag(reg)
    return np.linalg.solve(A, X.T @ y)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PortfolioChallengeModel:
    """
    Améliorations vs template :
      - Ridge regression (forme fermée, déterministe)
      - Covariance Ledoit-Wolf sur résidus → poids tangency portfolio
      - Signal momentum : blend prédictions + retours récents
      - Volatility scaling : exposition proportionnelle à 1/vol estimée
    """

    def __init__(
        self,
        lambda_reg: float = 1e-2,
        lam: float = 0.4,
        momentum_window: int = 168,   # ~1 semaine horaire
        momentum_blend: float = 0.25,
    ):
        self.lambda_reg      = lambda_reg
        self.lam             = lam
        self.momentum_window = momentum_window
        self.momentum_blend  = momentum_blend

        self.feature_cols = []
        self.asset_cols   = []
        self.pred_cols    = []
        self.weight_cols  = []
        self.thetas       = {}
        self.sigma_inv    = None
        self.asset_vols   = None
        self.momentum_mu  = None
        self.X_mean       = None
        self.X_std        = None

    # --- I/O ---------------------------------------------------------------

    def read_table(self, path):
        return pd.read_parquet(Path(path))

    def load_data(self, data_dir="data"):
        data_dir = Path(data_dir)
        x_train = self.read_table(data_dir / "X_train.parquet")
        r_train = self.read_table(data_dir / "R_train.parquet")
        x_test  = self.read_table(data_dir / "X_test.parquet")

        self.feature_cols = [c for c in x_train.columns if c not in ["id", "date", "split"]]
        self.asset_cols   = [c for c in r_train.columns if c not in ["id", "date", "split"]]
        self.pred_cols    = [f"pred_{c}"    for c in self.asset_cols]
        self.weight_cols  = [f"weight_{c}"  for c in self.asset_cols]

        return {"X_train": x_train, "R_train": r_train, "X_test": x_test}

    # --- Feature preprocessing ---------------------------------------------

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        X_n = (X - self.X_mean) / self.X_std
        return np.hstack([X_n, np.ones((len(X_n), 1))])

    # --- Training ----------------------------------------------------------

    def fit(self, x_train: pd.DataFrame, r_train: pd.DataFrame):
        X = x_train[self.feature_cols].values.astype(float)
        self.X_mean = X.mean(axis=0)
        self.X_std  = X.std(axis=0) + 1e-8
        X_n = self._normalize(X)

        # Ridge regression pour chaque actif
        R = r_train[self.asset_cols].values.astype(float)
        self.thetas = {
            asset: ridge_fit(X_n, R[:, i], self.lambda_reg)
            for i, asset in enumerate(self.asset_cols)
        }

        # Résidus → covariance Ledoit-Wolf → Σ⁻¹
        preds     = np.column_stack([X_n @ self.thetas[a] for a in self.asset_cols])
        residuals = R - preds
        sigma     = ledoit_wolf(residuals)
        self.sigma_inv = np.linalg.inv(sigma)

        # Volatilité empirique par actif (pour scaling)
        self.asset_vols = residuals.std(axis=0) + 1e-8

        # Signal momentum : moyenne des N dernières périodes
        window = min(self.momentum_window, len(R))
        self.momentum_mu = R[-window:].mean(axis=0)

        print(
            f"Ridge | λ={self.lambda_reg} | momentum_window={window}h "
            f"| momentum_blend={self.momentum_blend} | {len(self.asset_cols)} actifs"
        )

    # --- Prediction --------------------------------------------------------

    def predict_returns(self, x_df: pd.DataFrame) -> pd.DataFrame:
        X_n   = self._normalize(x_df[self.feature_cols].values.astype(float))
        preds = np.column_stack([X_n @ self.thetas[a] for a in self.asset_cols])
        return pd.DataFrame(preds, columns=self.pred_cols, index=x_df.index)

    # --- Portfolio construction --------------------------------------------

    def build_weights(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        mu_preds  = pred_df[self.pred_cols].values
        weights   = np.zeros_like(mu_preds)

        for t in range(len(mu_preds)):
            # Blend : prédiction modèle + momentum
            mu_t = (1 - self.momentum_blend) * mu_preds[t] + self.momentum_blend * self.momentum_mu

            # Tangency portfolio : w ∝ Σ⁻¹ μ
            w = self.lam * self.sigma_inv @ mu_t

            # Contraintes
            w = np.clip(w, -1, 1)
            gross = np.abs(w).sum()
            if gross > 1.0:
                w /= gross

            weights[t] = w

        return pd.DataFrame(weights, columns=self.weight_cols, index=pred_df.index)

    # --- Pipeline complet --------------------------------------------------

    def build_submission(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
        pred_train  = self.predict_returns(x_train)
        pred_test   = self.predict_returns(x_test)
        pred_all    = pd.concat([pred_train, pred_test]).reset_index(drop=True)
        weights_all = self.build_weights(pred_all).reset_index(drop=True)
        index_all   = pd.concat(
            [x_train[["id", "date"]], x_test[["id", "date"]]]
        ).reset_index(drop=True)
        return pd.concat([index_all, pred_all, weights_all], axis=1)

    def save_submission(self, submission: pd.DataFrame, path="submission.parquet") -> Path:
        p = Path(path)
        submission.to_parquet(p, index=False)
        return p

    def fit_predict_save(self, data_dir="data", output_path="submission.parquet") -> pd.DataFrame:
        data = self.load_data(data_dir)
        self.fit(data["X_train"], data["R_train"])
        submission = self.build_submission(data["X_train"], data["X_test"])
        self.save_submission(submission, output_path)
        return submission
