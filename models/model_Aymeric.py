"""
Adapted HuberLasso Stock Prediction Pipeline
=============================================
Inherits from PortfolioChallengeModel (model_template.py).

Input  : data/X_train.parquet, data/R_train.parquet, data/X_test.parquet
Output : submission.parquet  (id, date, pred_<asset>, weight_<asset>)
         — rows for BOTH train and test splits —

Weight constraints
------------------
  ∑ |W_{t,j}| ≤ 1   and   W_{t,j} ∈ [−1, 1]
"""

import numpy as np
import pandas as pd
from pathlib import Path

from model_template import PortfolioChallengeModel


# ──────────────────────────────────────────────
#  HUBER LOSS
# ──────────────────────────────────────────────

def huber_loss(r, delta=1.0):
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, 0.5 * r**2, delta * (abs_r - 0.5 * delta))

def huber_grad(r, delta=1.0):
    return np.where(np.abs(r) <= delta, r, delta * np.sign(r))


# ──────────────────────────────────────────────
#  SOFT-THRESHOLD / LASSO
# ──────────────────────────────────────────────

def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


class HuberLasso:
    """
    Proximal Gradient Descent for Huber + L1 regularisation.
    Works directly on the pre-built feature matrix X (already standardised).
    """

    def __init__(self, lam=1e-4, delta=1.0, lr=None, max_iter=3000, tol=1e-8):
        self.lam = lam
        self.delta = delta
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.theta = None
        self.losses = []

    def _f(self, X, y, theta):
        r = X @ theta - y
        return np.mean(huber_loss(r, self.delta))

    def _grad_f(self, X, y, theta):
        n = X.shape[0]
        r = X @ theta - y
        psi = huber_grad(r, self.delta)
        return (1.0 / n) * X.T @ psi

    def _objective(self, X, y, theta):
        return self._f(X, y, theta) + self.lam * np.sum(np.abs(theta))

    def fit(self, X, y):
        n, d = X.shape
        self.theta = np.zeros(d)

        if self.lr is None:
            eigmax = np.linalg.eigvalsh(X.T @ X / n)[-1]
            lr = 1.0 / (eigmax + 1e-10)
        else:
            lr = self.lr

        self.losses = []
        prev_loss = np.inf
        for _ in range(self.max_iter):
            grad = self._grad_f(X, y, self.theta)
            z = self.theta - lr * grad
            self.theta = soft_threshold(z, lr * self.lam)
            loss = self._objective(X, y, self.theta)
            self.losses.append(loss)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        return self

    def predict(self, X):
        return X @ self.theta

    def sparsity(self):
        return np.mean(np.abs(self.theta) < 1e-10) * 100


# ──────────────────────────────────────────────
#  DATA HELPERS
# ──────────────────────────────────────────────

def standardize(X, mean=None, std=None):
    if mean is None:
        mean = np.nanmean(X, axis=0)
    if std is None:
        std = np.nanstd(X, axis=0)
        std[std < 1e-12] = 1.0
    return (X - mean) / std, mean, std


# ──────────────────────────────────────────────
#  MAIN MODEL CLASS
# ──────────────────────────────────────────────

class HuberLassoModel(PortfolioChallengeModel):
    """
    HuberLasso model implementing the PortfolioChallengeModel interface.
    One HuberLasso regressor is trained per asset column.
    """

    def __init__(self, delta: float = 0.01, lam: float = 1e-6, verbose: bool = True):
        super().__init__()
        self.delta = delta
        self.lam = lam
        self.verbose = verbose

        # Internals set during fit()
        self._fitted_models: list[HuberLasso] = []
        self.mu_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None

    # ── helpers ──────────────────────────────────────────────────────────

    def read_table(self, path) -> pd.DataFrame:
        """Override to ensure date columns are parsed as datetime (fixes merge key mismatch)."""
        df = super().read_table(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def _standardize_X(self, X_raw: np.ndarray) -> np.ndarray:
        X_s, _, _ = standardize(X_raw, self.mu_, self.sigma_)
        return np.nan_to_num(X_s)

    # ── PortfolioChallengeModel interface ────────────────────────────────

    def fit(self, x_train: pd.DataFrame, r_train: pd.DataFrame) -> None:
        """
        Fit one HuberLasso per asset.

        Parameters
        ----------
        x_train : feature DataFrame (must contain self.feature_cols)
        r_train : returns DataFrame  (must contain self.asset_cols)
        """
        # Align on id/date keys
        merge_keys = [k for k in ["id", "date"] if k in x_train.columns and k in r_train.columns]
        merged = x_train.merge(
            r_train[merge_keys + self.asset_cols], on=merge_keys, how="inner"
        ).sort_values(merge_keys).reset_index(drop=True)

        if merged.empty:
            raise ValueError(
                f"Inner join (x_train ∩ r_train) produced 0 rows on keys {merge_keys}. "
                "Vérifiez que les colonnes id/date ont le même type dans les deux tables "
                "(ex. date en datetime64 dans les deux)."
            )
        if self.verbose:
            print(f"  Merged train rows: {len(merged)}")

        X_raw = merged[self.feature_cols].values.astype(np.float64)
        X_s, self.mu_, self.sigma_ = standardize(X_raw)
        X_s = np.nan_to_num(X_s)

        self._fitted_models = []
        M = len(self.asset_cols)

        for j, asset in enumerate(self.asset_cols):
            if self.verbose:
                print(f"  Fitting asset [{j+1}/{M}]: {asset} …", end=" ", flush=True)

            y_tr = merged[asset].values.astype(np.float64)
            valid = ~np.isnan(y_tr)
            X_tr_j = X_s[valid]
            y_tr_j = y_tr[valid]

            model = HuberLasso(lam=self.lam, delta=self.delta, max_iter=3000, tol=1e-8)
            if len(y_tr_j) >= 30:
                model.fit(X_tr_j, y_tr_j)
                if self.verbose:
                    print(f"done  (sparsity={model.sparsity():.1f}%)")
            else:
                # Not enough samples: keep zero-weights model
                model.theta = np.zeros(X_s.shape[1])
                if self.verbose:
                    print("skipped (too few samples)")

            self._fitted_models.append(model)

    def predict_returns(self, x_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame of shape (len(x_df), n_assets) with pred_<asset> columns.
        """
        if not self._fitted_models:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        X_raw = x_df[self.feature_cols].values.astype(np.float64)
        X_s = self._standardize_X(X_raw)

        preds = [model.predict(X_s) for model in self._fitted_models]
        # column_stack needs at least 2 arrays; vstack+T handles M=1 safely
        pred_matrix = np.stack(preds, axis=1)  # shape (T, M)

        return pd.DataFrame(pred_matrix, columns=self.pred_cols, index=x_df.index)

    def build_weights(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert return predictions into L1-normalised portfolio weights.

        Constraints:  ∑|w_{t,j}| ≤ 1  and  w_{t,j} ∈ [−1, 1]
        Uses the row-wise L1-normalisation from the original caca.py pipeline
        (stricter than the template's gross-exposure cap).
        """
        raw = pred_df.to_numpy(dtype=float)
        T, M = raw.shape
        weights = np.zeros_like(raw)

        for t in range(T):
            row = raw[t].copy()
            l1 = np.sum(np.abs(row))
            if l1 > 1e-12:
                row = row / l1          # ∑|w| == 1
            row = np.clip(row, -1.0, 1.0)
            weights[t] = row

        return pd.DataFrame(weights, columns=self.weight_cols, index=pred_df.index)


# ──────────────────────────────────────────────
#  CROSS-VALIDATION HELPER (optional evaluation)
# ──────────────────────────────────────────────

def evaluate_cv(
    data_dir: str = "data",
    delta: float = 0.01,
    lam: float = 1e-6,
    n_splits: int = 5,
    train_pct: float = 0.7,
    verbose: bool = True,
) -> dict:
    """
    Time-series cross-validation on the training set only.
    Reports Huber Loss per asset and portfolio Sharpe (sign-based).
    """
    # Reuse load_data from the model to respect the template's I/O contract
    tmp_model = HuberLassoModel(delta=delta, lam=lam, verbose=False)
    data = tmp_model.load_data(data_dir)
    X_train_df = data["X_train"]
    R_train_df = data["R_train"]

    asset_cols   = tmp_model.asset_cols
    feature_cols = tmp_model.feature_cols

    merge_keys   = [k for k in ["id", "date"] if k in X_train_df.columns]
    train_merged = X_train_df.merge(
        R_train_df[[*merge_keys, *asset_cols]], on=merge_keys, how="inner"
    ).sort_values(merge_keys).reset_index(drop=True)

    X_raw = train_merged[feature_cols].values.astype(np.float64)
    R_raw = train_merged[asset_cols].values.astype(np.float64)

    n = len(X_raw)
    window = n // n_splits
    all_metrics = []

    for fold in range(n_splits):
        start = fold * window
        end   = min(start + window, n)
        if end - start < 100:
            continue

        split = start + int((end - start) * train_pct)
        X_tr_raw, X_te_raw = X_raw[start:split], X_raw[split:end]
        R_tr_raw, R_te_raw = R_raw[start:split], R_raw[split:end]

        X_tr_s, mu, sigma = standardize(X_tr_raw)
        X_te_s, _,  _     = standardize(X_te_raw, mu, sigma)
        X_tr_s = np.nan_to_num(X_tr_s)
        X_te_s = np.nan_to_num(X_te_s)

        T_te = len(X_te_s)
        M    = len(asset_cols)
        pred_matrix = np.zeros((T_te, M))

        for j, asset in enumerate(asset_cols):
            y_tr = R_tr_raw[:, j]
            valid = ~np.isnan(y_tr)
            if valid.sum() < 30:
                continue
            model = HuberLasso(lam=lam, delta=delta, max_iter=3000, tol=1e-8)
            model.fit(X_tr_s[valid], y_tr[valid])
            pred_matrix[:, j] = model.predict(X_te_s)

        # Build weights via a throw-away model instance
        _m = HuberLassoModel(verbose=False)
        _m.asset_cols  = asset_cols
        _m.weight_cols = [f"weight_{a}" for a in asset_cols]
        _m.pred_cols   = [f"pred_{a}"   for a in asset_cols]
        pred_df  = pd.DataFrame(pred_matrix, columns=_m.pred_cols)
        w_df     = _m.build_weights(pred_df)
        weights  = w_df.to_numpy()

        pnl = (weights * np.nan_to_num(R_te_raw)).sum(axis=1)
        fold_loss = np.mean([
            np.mean(huber_loss(pred_matrix[:, j] - np.nan_to_num(R_te_raw[:, j]), delta))
            for j in range(M)
        ])
        sharpe = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252 * 24)

        all_metrics.append({"fold": fold + 1, "huber_loss": fold_loss, "sharpe": sharpe})

        if verbose:
            print(f"  Fold {fold+1}: Huber Loss={fold_loss:.6f}  Sharpe={sharpe:.4f}")

    if verbose and all_metrics:
        losses  = [m["huber_loss"] for m in all_metrics]
        sharpes = [m["sharpe"]     for m in all_metrics]
        print(f"\n  Mean Huber Loss : {np.mean(losses):.6f} ± {np.std(losses):.6f}")
        print(f"  Mean Sharpe     : {np.mean(sharpes):.4f} ± {np.std(sharpes):.4f}")

    return {"folds": all_metrics}


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────

def main():
    model = HuberLassoModel(delta=0.01, lam=1e-6, verbose=True)
    model.fit_predict_save(data_dir="data", output_path="submission.parquet")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HuberLasso multi-asset pipeline")
    parser.add_argument("--data_dir", default="data",
                        help="Directory containing X_train/R_train/X_test parquet files")
    parser.add_argument("--output",   default="submission.parquet")
    parser.add_argument("--delta",    type=float, default=0.01)
    parser.add_argument("--lam",      type=float, default=1e-6)
    parser.add_argument("--cv",       action="store_true",
                        help="Run cross-validation before predicting")
    args = parser.parse_args()

    if args.cv:
        print("=== Cross-Validation ===")
        evaluate_cv(data_dir=args.data_dir, delta=args.delta, lam=args.lam)
        print()

    print("=== Generating Predictions ===")
    model = HuberLassoModel(delta=args.delta, lam=args.lam, verbose=True)
    model.fit_predict_save(data_dir=args.data_dir, output_path=args.output)

