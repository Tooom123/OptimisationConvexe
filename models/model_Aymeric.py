"""
================================================================================
  ADVANCED QUANTITATIVE PORTFOLIO MODEL
  Hedge-fund grade implementation extending PortfolioChallengeModel template
================================================================================
  Architecture:
    Feature Engineering → Per-Asset Prediction → Covariance Estimation
    → Portfolio Optimization (Sharpe) → Ensemble → Submission

  Special asset handling:
    Assets  7, 8  : nonlinear function of cross-sectional mean  → polynomial + MLP
    Asset   9     : threshold / jump process                    → regime classifier + regression
    Asset  10     : cross-asset dependency                      → lagged correlation features
    Assets 11, 12 : high noise                                  → strong shrinkage + low allocation
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy


# ─────────────────────────────────────────────────────────────────────────────
# 0.  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO") -> None:
    prefix = {"INFO": "ℹ️ ", "OK": "✅", "WARN": "⚠️ ", "SECTION": "\n" + "═" * 70 + "\n  "}
    print(f"{prefix.get(level, '')} {msg}")


def sharpe(returns: np.ndarray, eps: float = 1e-8) -> float:
    """Annualised Sharpe (252 trading periods)."""
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    return float(mu / (sigma + eps) * np.sqrt(252))


def rolling_sharpe_std(returns: np.ndarray, window: int = 63) -> float:
    """Std of rolling Sharpe — penalises instability."""
    if len(returns) < window + 1:
        return 0.0
    vals = []
    for i in range(window, len(returns)):
        vals.append(sharpe(returns[i - window : i]))
    return float(np.std(vals)) if vals else 0.0


def selection_score(returns: np.ndarray, penalty: float = 0.5) -> float:
    """Score = Sharpe - penalty * rolling_sharpe_std."""
    s = sharpe(returns)
    v = rolling_sharpe_std(returns)
    return s - penalty * v


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Builds a rich, time-safe feature matrix from raw X.
    All lag / rolling features are purely backward-looking.
    """

    LAGS       = [1, 2, 3, 5, 10, 21]
    ROLL_WIN   = [5, 10, 21, 63]
    POLY_DEG   = 2           # for cross-sectional mean features (assets 7/8)
    N_PCA      = 5

    def __init__(self, feature_cols: list):
        self.feature_cols = feature_cols
        self.pca_components_: np.ndarray | None = None
        self.pca_mean_: np.ndarray | None = None
        self.clip_quantiles_: dict = {}

    # ------------------------------------------------------------------
    def fit(self, x_df: pd.DataFrame) -> None:
        X = x_df[self.feature_cols].values.astype(float)
        # PCA on raw features
        X_centered = X - np.nanmean(X, axis=0)
        X_centered = np.nan_to_num(X_centered)
        cov = np.cov(X_centered.T) + 1e-6 * np.eye(X_centered.shape[1])
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        self.pca_components_ = eigvecs[:, idx[: self.N_PCA]]
        self.pca_mean_       = np.nanmean(X, axis=0)
        # Clip quantiles (per feature) to handle outliers
        for col in self.feature_cols:
            vals = x_df[col].dropna().values
            q1, q99 = np.percentile(vals, 1), np.percentile(vals, 99)
            self.clip_quantiles_[col] = (q1, q99)

    # ------------------------------------------------------------------
    def transform(self, x_df: pd.DataFrame) -> pd.DataFrame:
        df = x_df.copy().sort_values("date").reset_index(drop=True)
        raw = df[self.feature_cols].copy()

        # ── clip outliers
        for col in self.feature_cols:
            if col in self.clip_quantiles_:
                lo, hi = self.clip_quantiles_[col]
                raw[col] = raw[col].clip(lo, hi)

        feats = pd.DataFrame(index=df.index)

        # ── raw features (winsorised)
        for col in self.feature_cols:
            feats[col] = raw[col].values

        # ── lag features
        for lag in self.LAGS:
            for col in self.feature_cols:
                feats[f"{col}_lag{lag}"] = raw[col].shift(lag).values

        # ── rolling statistics
        for w in self.ROLL_WIN:
            for col in self.feature_cols:
                feats[f"{col}_rmean{w}"] = raw[col].rolling(w, min_periods=1).mean().values
                feats[f"{col}_rstd{w}"]  = raw[col].rolling(w, min_periods=2).std().fillna(0).values

        # ── z-score (rolling)
        for w in [21, 63]:
            for col in self.feature_cols:
                mu   = raw[col].rolling(w, min_periods=1).mean()
                sigma = raw[col].rolling(w, min_periods=2).std().fillna(1e-8).replace(0, 1e-8)
                feats[f"{col}_zscore{w}"] = ((raw[col] - mu) / sigma).values

        # ── momentum
        for lag in [5, 10, 21]:
            cross = raw[self.feature_cols].mean(axis=1)
            feats[f"xmean_mom{lag}"] = cross - cross.shift(lag).fillna(cross)

        # ── cross-sectional mean (critical for assets 7/8)
        xbar = raw[self.feature_cols].mean(axis=1)
        feats["xbar"] = xbar.values
        feats["xbar_abs"] = xbar.abs().values
        feats["xbar_sign"] = np.sign(xbar.values)
        # polynomial of xbar (for nonlinear signal detection)
        for d in range(2, self.POLY_DEG + 2):
            feats[f"xbar_pow{d}"] = (xbar ** d).values

        # ── threshold features for asset 9 (jump process)
        for thr in [0.3, 0.5, 0.7, 1.0]:
            feats[f"xbar_above_{thr}"]  = (xbar >  thr).astype(float).values
            feats[f"xbar_below_{thr}"] = (xbar < -thr).astype(float).values

        # ── volatility regime
        roll_vol = raw[self.feature_cols].std(axis=1).rolling(21, min_periods=5).mean().fillna(0)
        feats["vol_regime"] = (roll_vol > roll_vol.median()).astype(float).values
        feats["roll_vol21"] = roll_vol.values

        # ── pairwise feature interactions (top-5 features only, to keep dim manageable)
        top5 = self.feature_cols[:5]
        for i in range(len(top5)):
            for j in range(i + 1, len(top5)):
                feats[f"inter_{top5[i]}_{top5[j]}"] = (raw[top5[i]] * raw[top5[j]]).values

        # ── PCA projections
        if self.pca_components_ is not None:
            X_raw = raw[self.feature_cols].fillna(0).values
            X_centered = X_raw - self.pca_mean_
            pca_proj = X_centered @ self.pca_components_
            for k in range(self.N_PCA):
                feats[f"pca_{k}"] = pca_proj[:, k]

        # ── lagged cross-asset correlations (for asset 10)
        for lag in [1, 3, 5]:
            for col in self.feature_cols[:6]:  # use first 6 as proxy for other assets
                feats[f"xbar_x_{col}_lag{lag}"] = (xbar.shift(lag) * raw[col]).fillna(0).values

        feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
        return feats

    # ------------------------------------------------------------------
    def fit_transform(self, x_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(x_df)
        return self.transform(x_df)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BASE MODELS (from scratch + sklearn-compatible interface)
# ─────────────────────────────────────────────────────────────────────────────

class RidgeModel:
    """Closed-form Ridge regression."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        X = np.c_[np.ones(len(X)), X]
        n, p = X.shape
        A = X.T @ X + self.alpha * np.eye(p)
        A[0, 0] -= self.alpha  # don't regularise bias
        self.coef_ = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.c_[np.ones(len(X)), X]
        return X @ self.coef_


class LassoSGD:
    """Lasso via coordinate-descent-style SGD (from scratch)."""
    def __init__(self, alpha: float = 0.01, lr: float = 1e-3,
                 n_iter: int = 500, batch: int = 64, tol: float = 1e-5):
        self.alpha = alpha
        self.lr    = lr
        self.n_iter = n_iter
        self.batch  = batch
        self.tol    = tol
        self.coef_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoSGD":
        n, p = X.shape
        w = np.zeros(p)
        b = 0.0
        prev_loss = np.inf
        rng = np.random.default_rng(42)
        for t in range(self.n_iter):
            idx = rng.choice(n, min(self.batch, n), replace=False)
            Xb, yb = X[idx], y[idx]
            residuals = Xb @ w + b - yb
            grad_w = (Xb.T @ residuals) / len(idx) + self.alpha * np.sign(w)
            grad_b = residuals.mean()
            # cosine-annealing lr schedule
            lr_t = self.lr * (1 + np.cos(np.pi * t / self.n_iter)) / 2 + 1e-6
            w -= lr_t * grad_w
            b -= lr_t * grad_b
            # early stopping
            if t % 50 == 0:
                loss = np.mean((X @ w + b - y) ** 2) + self.alpha * np.sum(np.abs(w))
                if abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss
        self.coef_ = w
        self.bias_  = b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.bias_


class ElasticNetSGD:
    """ElasticNet via mini-batch SGD (from scratch)."""
    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.5,
                 lr: float = 1e-3, n_iter: int = 500, batch: int = 64):
        self.alpha    = alpha
        self.l1_ratio = l1_ratio
        self.lr       = lr
        self.n_iter   = n_iter
        self.batch    = batch
        self.coef_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetSGD":
        n, p = X.shape
        w = np.zeros(p)
        b = 0.0
        rng = np.random.default_rng(42)
        for t in range(self.n_iter):
            idx = rng.choice(n, min(self.batch, n), replace=False)
            Xb, yb = X[idx], y[idx]
            residuals = Xb @ w + b - yb
            l1 = self.alpha * self.l1_ratio * np.sign(w)
            l2 = self.alpha * (1 - self.l1_ratio) * w
            grad_w = (Xb.T @ residuals) / len(idx) + l1 + l2
            grad_b = residuals.mean()
            lr_t = self.lr / (1 + 0.001 * t)
            w -= lr_t * grad_w
            b -= lr_t * grad_b
        self.coef_ = w
        self.bias_  = b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.bias_


class MLPFromScratch:
    """
    2-hidden-layer MLP trained with mini-batch Adam.
    Architecture: input → H1 → ReLU → H2 → ReLU → 1
    """
    def __init__(self, hidden1: int = 64, hidden2: int = 32,
                 lr: float = 1e-3, n_iter: int = 300, batch: int = 128,
                 dropout: float = 0.1, weight_decay: float = 1e-4):
        self.h1           = hidden1
        self.h2           = hidden2
        self.lr           = lr
        self.n_iter       = n_iter
        self.batch        = batch
        self.dropout      = dropout
        self.weight_decay = weight_decay
        self.params: dict = {}

    # ── activation
    @staticmethod
    def relu(x):  return np.maximum(0, x)
    @staticmethod
    def relu_grad(x): return (x > 0).astype(float)

    # ── He initialisation
    def _init_params(self, n_in: int) -> None:
        rng = np.random.default_rng(7)
        self.params = {
            "W1": rng.standard_normal((n_in, self.h1)) * np.sqrt(2 / n_in),
            "b1": np.zeros(self.h1),
            "W2": rng.standard_normal((self.h1, self.h2)) * np.sqrt(2 / self.h1),
            "b2": np.zeros(self.h2),
            "W3": rng.standard_normal((self.h2, 1)) * np.sqrt(2 / self.h2),
            "b3": np.zeros(1),
        }
        # Adam moments
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    # ── forward pass
    def _forward(self, X: np.ndarray, training: bool = False):
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        z1 = X @ W1 + b1
        a1 = self.relu(z1)
        if training and self.dropout > 0:
            mask1 = (np.random.rand(*a1.shape) > self.dropout) / (1 - self.dropout)
            a1 = a1 * mask1
        else:
            mask1 = None
        z2 = a1 @ W2 + b2
        a2 = self.relu(z2)
        if training and self.dropout > 0:
            mask2 = (np.random.rand(*a2.shape) > self.dropout) / (1 - self.dropout)
            a2 = a2 * mask2
        else:
            mask2 = None
        z3 = a2 @ W3 + b3
        cache = (X, z1, a1, mask1, z2, a2, mask2, z3)
        return z3.squeeze(), cache

    # ── backward pass
    def _backward(self, cache, y: np.ndarray):
        X, z1, a1, mask1, z2, a2, mask2, z3 = cache
        n = len(y)
        dz3 = (z3.squeeze() - y) / n
        dW3 = a2.T @ dz3[:, None]
        db3 = dz3.sum(keepdims=True)
        da2 = dz3[:, None] @ self.params["W3"].T
        if mask2 is not None:
            da2 = da2 * mask2
        dz2 = da2 * self.relu_grad(z2)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.params["W2"].T
        if mask1 is not None:
            da1 = da1 * mask1
        dz1 = da1 * self.relu_grad(z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        # L2 regularisation
        for k in ["W1", "W2", "W3"]:
            grads[k] += self.weight_decay * self.params[k]
        return grads

    # ── Adam update
    def _adam_step(self, grads: dict) -> None:
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for k in self.params:
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * grads[k]
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPFromScratch":
        self._init_params(X.shape[1])
        rng = np.random.default_rng(42)
        best_loss, best_params = np.inf, None
        patience, no_improve = 20, 0
        for epoch in range(self.n_iter):
            idx = rng.permutation(len(X))
            for start in range(0, len(X), self.batch):
                b_idx = idx[start : start + self.batch]
                Xb, yb = X[b_idx], y[b_idx]
                _, cache = self._forward(Xb, training=True)
                grads = self._backward(cache, yb)
                self._adam_step(grads)
            # validation loss (full data, no dropout)
            if epoch % 10 == 0:
                preds, _ = self._forward(X, training=False)
                loss = np.mean((preds - y) ** 2)
                if loss < best_loss - 1e-6:
                    best_loss = loss
                    best_params = deepcopy(self.params)
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break
        if best_params is not None:
            self.params = best_params
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds, _ = self._forward(X, training=False)
        return preds


class SimpleGBM:
    """
    Shallow Gradient Boosting via repeated residual fitting (from scratch).
    Each weak learner = shallow Ridge on residuals.
    """
    def __init__(self, n_estimators: int = 50, lr: float = 0.1,
                 subsample: float = 0.8, alpha: float = 1.0):
        self.n_estimators = n_estimators
        self.lr           = lr
        self.subsample    = subsample
        self.alpha        = alpha
        self.learners: list  = []
        self.baseline_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleGBM":
        self.baseline_ = np.mean(y)
        residuals = y - self.baseline_
        rng = np.random.default_rng(42)
        self.learners = []
        for _ in range(self.n_estimators):
            # subsample rows
            n = len(X)
            idx = rng.choice(n, int(n * self.subsample), replace=False)
            learner = RidgeModel(alpha=self.alpha)
            learner.fit(X[idx], residuals[idx])
            upd = learner.predict(X)
            residuals -= self.lr * upd
            self.learners.append(learner)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.full(len(X), self.baseline_)
        for learner in self.learners:
            out += self.lr * learner.predict(X)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PER-ASSET PREDICTOR FACTORY  (specialised per asset type)
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdJumpPredictor:
    """
    Specialised predictor for asset 9 (jump process).
    Two-stage: regime classifier → conditional regressor.
    """
    def __init__(self):
        self.clf_above: RidgeModel | None = None
        self.clf_below: RidgeModel | None = None
        self.clf_neutral: RidgeModel | None = None
        self.threshold: float = 0.5

    def _regime(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 (above), -1 (below), 0 (neutral) based on xbar."""
        xbar_col = -1  # last engineered column (xbar_abs) – adjusted in fit
        xbar = X[:, self._xbar_idx]
        above  = xbar >  self.threshold
        below  = xbar < -self.threshold
        regime = np.zeros(len(X))
        regime[above] = 1
        regime[below] = -1
        return regime

    def fit(self, X: np.ndarray, y: np.ndarray,
            xbar_idx: int = 0, threshold: float = 0.5) -> "ThresholdJumpPredictor":
        self._xbar_idx = xbar_idx
        self.threshold = threshold
        regime = self._regime(X)
        for r, attr in [(1, "clf_above"), (-1, "clf_below"), (0, "clf_neutral")]:
            mask = regime == r
            if mask.sum() > 10:
                m = RidgeModel(alpha=1.0)
                m.fit(X[mask], y[mask])
                setattr(self, attr, m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        regime = self._regime(X)
        out = np.zeros(len(X))
        for r, attr in [(1, "clf_above"), (-1, "clf_below"), (0, "clf_neutral")]:
            mask = regime == r
            m = getattr(self, attr)
            if m is not None and mask.sum() > 0:
                out[mask] = m.predict(X[mask])
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ENSEMBLE PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Weighted ensemble of base models.
    Weights are proportional to max(0, Sharpe_val) on a held-out window.
    Falls back to uniform if all Sharpes are ≤ 0.
    """
    def __init__(self, models: list[tuple[str, object]]):
        self.models  = models          # list of (name, model_instance)
        self.weights_: np.ndarray | None = None

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> "EnsemblePredictor":
        sharpes = []
        for name, m in self.models:
            m.fit(X_tr, y_tr)
            val_pred = m.predict(X_val)
            pnl = val_pred * y_val          # PnL proxy
            sh  = sharpe(pnl)
            sharpes.append(max(0.0, sh))
            log(f"  [{name}] val Sharpe={sh:.3f}")
        total = sum(sharpes)
        if total < 1e-9:
            self.weights_ = np.ones(len(sharpes)) / len(sharpes)
        else:
            self.weights_ = np.array(sharpes) / total
        log(f"  Ensemble weights: {dict(zip([n for n,_ in self.models], np.round(self.weights_,3)))}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([m.predict(X) for _, m in self.models], axis=1)
        return preds @ self.weights_


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COVARIANCE ESTIMATOR (Ledoit-Wolf shrinkage, from scratch)
# ─────────────────────────────────────────────────────────────────────────────

class RobustCovarianceEstimator:
    """
    Oracle Approximating Shrinkage (simplified analytical Ledoit-Wolf).
    Sigma_shrunk = (1-delta)*S + delta * mu_hat * I
    """
    def __init__(self, window: int = 63, shrinkage: float | None = None):
        self.window    = window
        self.shrinkage = shrinkage   # None → analytical estimation

    def estimate(self, returns: np.ndarray) -> np.ndarray:
        """
        returns: shape (T, n_assets)
        Returns shrunk covariance matrix (n_assets, n_assets).
        """
        T, n = returns.shape
        if T < 3:
            return np.eye(n)
        S = np.cov(returns.T, ddof=1)
        if n == 1:
            return S.reshape(1, 1)
        # analytical shrinkage intensity (Oracle Approximating)
        if self.shrinkage is None:
            mu = np.trace(S) / n
            # Frobenius norm²
            delta = 0.0
            for i in range(n):
                for j in range(n):
                    delta += S[i, j] ** 2
            # shrinkage intensity
            rho = min(1.0, ((n + 2) / (n * T)) * delta / (np.trace(S @ S) / n + (np.trace(S) ** 2 / n - 2 * np.trace(S @ S) / n + delta) / T))
            rho = max(0.0, min(1.0, rho))
        else:
            rho = self.shrinkage
            mu  = np.trace(S) / n
        Sigma = (1 - rho) * S + rho * mu * np.eye(n)
        return Sigma

    def rolling(self, returns: np.ndarray, t: int) -> np.ndarray:
        start = max(0, t - self.window)
        window_data = returns[start : t]
        return self.estimate(window_data)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PORTFOLIO OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioOptimizer:
    """
    Given mu (predicted returns) and Sigma (covariance), returns weights.
    Maximises Sharpe ratio subject to sum|w| ≤ 1.
    """
    def __init__(self, method: str = "sharpe",
                 max_gross: float = 1.0,
                 risk_aversion: float = 1.0,
                 noisy_asset_ids: list[int] | None = None,
                 noisy_penalty: float = 5.0):
        """
        method: 'sharpe' | 'mv' | 'risk_parity'
        noisy_asset_ids: indices of high-noise assets (11, 12) → penalised
        """
        self.method          = method
        self.max_gross       = max_gross
        self.risk_aversion   = risk_aversion
        self.noisy_asset_ids = noisy_asset_ids or []
        self.noisy_penalty   = noisy_penalty

    def optimise(self, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        n = len(mu)
        if self.method == "sharpe":
            return self._sharpe_opt(mu, Sigma, n)
        elif self.method == "mv":
            return self._mv_opt(mu, Sigma, n)
        elif self.method == "risk_parity":
            return self._risk_parity(Sigma, n)
        return self._sharpe_opt(mu, Sigma, n)

    def _regularise_sigma(self, Sigma: np.ndarray) -> np.ndarray:
        n = Sigma.shape[0]
        # Ensure PD
        min_eig = np.linalg.eigvalsh(Sigma).min()
        if min_eig < 1e-6:
            Sigma = Sigma + (1e-6 - min_eig + 1e-8) * np.eye(n)
        # Penalise noisy assets by inflating their variance
        for idx in self.noisy_asset_ids:
            if idx < n:
                Sigma[idx, idx] *= self.noisy_penalty
        return Sigma

    def _sharpe_opt(self, mu: np.ndarray, Sigma: np.ndarray, n: int) -> np.ndarray:
        Sigma = self._regularise_sigma(Sigma.copy())
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)
        w_raw = Sigma_inv @ mu
        # Gradient ascent on Sharpe with L2 regularisation
        w = w_raw.copy()
        lr = 0.1
        for _ in range(200):
            port_var   = w @ Sigma @ w
            port_ret   = w @ mu
            denom      = np.sqrt(max(port_var, 1e-10))
            grad       = (mu * denom - port_ret * (Sigma @ w) / denom) / max(port_var, 1e-10)
            w          = w + lr * grad
            w          = self._project(w)
        return w

    def _mv_opt(self, mu: np.ndarray, Sigma: np.ndarray, n: int) -> np.ndarray:
        Sigma = self._regularise_sigma(Sigma.copy())
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)
        w_raw = Sigma_inv @ mu / self.risk_aversion
        return self._project(w_raw)

    def _risk_parity(self, Sigma: np.ndarray, n: int) -> np.ndarray:
        Sigma = self._regularise_sigma(Sigma.copy())
        vols  = np.sqrt(np.diag(Sigma))
        w = 1.0 / np.maximum(vols, 1e-8)
        w = w / w.sum()
        return self._project(w)

    def _project(self, w: np.ndarray) -> np.ndarray:
        """L1-ball projection (sum|w| ≤ max_gross), clip individual weights to [-1, 1]."""
        w = np.clip(w, -1.0, 1.0)
        gross = np.sum(np.abs(w))
        if gross > self.max_gross:
            w = w / gross * self.max_gross
        return w


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN MODEL  (extends PortfolioChallengeModel)
# ─────────────────────────────────────────────────────────────────────────────

class SubmissionScorerTemplate:
    def __init__(self):
        self.asset_cols  = []
        self.pred_cols   = []
        self.weight_cols = []

    def set_assets(self, asset_cols):
        self.asset_cols  = list(asset_cols)
        self.pred_cols   = [f"pred_{col}"   for col in self.asset_cols]
        self.weight_cols = [f"weight_{col}" for col in self.asset_cols]

    def validate_submission_format(self, submission):
        required = ["id", "date"] + self.pred_cols + self.weight_cols
        missing  = [col for col in required if col not in submission.columns]
        if missing:
            raise ValueError(f"Missing columns in submission: {missing}")
        if submission[required].isnull().any().any():
            raise ValueError("Submission contains missing values.")
        if submission["id"].duplicated().any():
            raise ValueError("Submission contains duplicate ids.")

    def score_predictions(self, predictions, realized_returns):
        common_assets = [col for col in self.asset_cols if col in realized_returns.columns]
        if not common_assets:
            raise ValueError("No common asset columns found for scoring.")
        pred_cols = [f"pred_{col}" for col in common_assets]
        merged = realized_returns[["id", "date"] + common_assets].merge(
            predictions[["id", "date"] + pred_cols], on=["id", "date"], how="inner"
        )
        if merged.empty:
            raise ValueError("No overlapping rows found between predictions and realized returns.")
        y_true = merged[common_assets].to_numpy(dtype=float)
        y_pred = merged[pred_cols].to_numpy(dtype=float)
        mse    = float(np.mean((y_true - y_pred) ** 2))
        return {"mse": mse, "n_rows": int(len(merged))}


class PortfolioChallengeModel:
    """
    Full production-grade implementation.
    Override of the template with:
      - rich feature engineering
      - per-asset specialised models
      - ensemble with Sharpe weighting
      - Ledoit-Wolf covariance
      - Sharpe-maximising portfolio optimisation
    """

    VAL_FRAC     = 0.2          # fraction of train used for validation
    PENALTY_VAR  = 0.5          # Sharpe variance penalty in selection score
    SPECIAL_78   = ["asset_7",  "asset_8"]
    SPECIAL_9    = ["asset_9"]
    SPECIAL_10   = ["asset_10"]
    NOISY        = ["asset_11", "asset_12"]

    def __init__(self):
        self.feature_cols  : list = []
        self.asset_cols    : list = []
        self.pred_cols     : list = []
        self.weight_cols   : list = []

        self.feat_eng       : FeatureEngineer | None = None
        self.per_asset_models: dict = {}      # asset → fitted EnsemblePredictor
        self.cov_estimator  : RobustCovarianceEstimator = RobustCovarianceEstimator()
        self.optimizer      : PortfolioOptimizer | None = None
        self._xbar_idx      : int = 0         # index of "xbar" column in engineered features
        self._feat_cols_eng : list = []        # column names after engineering

    # ──────────────────────────────────────────────────────────────────────
    def read_table(self, path) -> pd.DataFrame:
        path = Path(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file format: {path}. Expected .parquet")

    # ──────────────────────────────────────────────────────────────────────
    def load_data(self, data_dir: str = "data") -> dict:
        data_dir = Path(data_dir)
        x_train  = self.read_table(data_dir / "X_train.parquet")
        r_train  = self.read_table(data_dir / "R_train.parquet")
        x_test   = self.read_table(data_dir / "X_test.parquet")

        self.feature_cols  = [c for c in x_train.columns if c not in ("id", "date", "split")]
        self.asset_cols    = [c for c in r_train.columns if c not in ("id", "date", "split")]
        self.pred_cols     = [f"pred_{c}"   for c in self.asset_cols]
        self.weight_cols   = [f"weight_{c}" for c in self.asset_cols]

        log(f"Loaded: {len(x_train)} train rows, {len(x_test)} test rows, "
            f"{len(self.feature_cols)} features, {len(self.asset_cols)} assets", "OK")
        return {"X_train": x_train, "R_train": r_train, "X_test": x_test}

    # ──────────────────────────────────────────────────────────────────────
    def _build_model_suite(self, asset: str, X_shape: int) -> list[tuple[str, object]]:
        """Return list of (name, model) specialised per asset type."""
        base_models = [
            ("ridge_1",   RidgeModel(alpha=0.1)),
            ("ridge_10",  RidgeModel(alpha=10.0)),
            ("lasso",     LassoSGD(alpha=0.01,  n_iter=300)),
            ("elasticnet",ElasticNetSGD(alpha=0.01, l1_ratio=0.5, n_iter=300)),
            ("gbm",       SimpleGBM(n_estimators=40, lr=0.05, alpha=1.0)),
        ]

        # Assets 7 & 8 → add MLP (nonlinear mean dependence)
        if asset in self.SPECIAL_78:
            base_models += [
                ("mlp_small", MLPFromScratch(hidden1=32, hidden2=16, n_iter=200)),
                ("mlp_large", MLPFromScratch(hidden1=64, hidden2=32, n_iter=300)),
            ]

        # Asset 9 → handled by ThresholdJumpPredictor later (see fit)
        # We still add base models for ensemble backup
        if asset in self.SPECIAL_9:
            base_models += [
                ("mlp_jump", MLPFromScratch(hidden1=64, hidden2=32, n_iter=250)),
            ]

        # Assets 11/12 → only regularised linear (shrink hard)
        if asset in self.NOISY:
            base_models = [
                ("ridge_strong", RidgeModel(alpha=100.0)),
                ("ridge_med",    RidgeModel(alpha=10.0)),
            ]

        return base_models

    # ──────────────────────────────────────────────────────────────────────
    def fit(self, x_train: pd.DataFrame, r_train: pd.DataFrame) -> None:
        log("FEATURE ENGINEERING", "SECTION")

        # Sort by date
        x_train = x_train.sort_values("date").reset_index(drop=True)
        r_train = r_train.sort_values("date").reset_index(drop=True)

        # Align on common ids
        common_ids = set(x_train["id"]).intersection(r_train["id"])
        x_tr = x_train[x_train["id"].isin(common_ids)].reset_index(drop=True)
        r_tr = r_train[r_train["id"].isin(common_ids)].reset_index(drop=True)

        # Feature engineering
        self.feat_eng = FeatureEngineer(self.feature_cols)
        X_eng = self.feat_eng.fit_transform(x_tr)
        self._feat_cols_eng = list(X_eng.columns)
        self._xbar_idx      = self._feat_cols_eng.index("xbar") if "xbar" in self._feat_cols_eng else 0

        log(f"Engineered feature matrix: {X_eng.shape}", "OK")

        # Train / val split (temporal)
        n       = len(X_eng)
        n_val   = max(int(n * self.VAL_FRAC), 30)
        n_tr    = n - n_val
        X_tr_arr   = X_eng.iloc[:n_tr].values.astype(float)
        X_val_arr  = X_eng.iloc[n_tr:].values.astype(float)

        # Standardise features
        self._feat_mean = X_tr_arr.mean(axis=0)
        self._feat_std  = X_tr_arr.std(axis=0) + 1e-8
        X_tr_arr  = (X_tr_arr  - self._feat_mean) / self._feat_std
        X_val_arr = (X_val_arr - self._feat_mean) / self._feat_std

        # Portfolio-level Sharpe tracking for noisy-asset penalty setup
        noisy_indices = [
            i for i, a in enumerate(self.asset_cols) if a in self.NOISY
        ]
        self.optimizer = PortfolioOptimizer(
            method="sharpe",
            noisy_asset_ids=noisy_indices,
            noisy_penalty=6.0,
        )

        log("PER-ASSET MODEL TRAINING", "SECTION")
        all_val_pnl = []

        for asset in self.asset_cols:
            if asset not in r_tr.columns:
                log(f"  {asset}: not found in returns, skipping", "WARN")
                continue
            log(f"\n  ── {asset.upper()} ──")
            y_all = r_tr[asset].values.astype(float)
            y_tr  = y_all[:n_tr]
            y_val = y_all[n_tr:]

            # Specialised: Asset 9 threshold predictor (added to ensemble)
            models_suite = self._build_model_suite(asset, X_tr_arr.shape[1])

            if asset in self.SPECIAL_9:
                jump_model = ThresholdJumpPredictor()
                jump_model.fit(X_tr_arr, y_tr, xbar_idx=self._xbar_idx, threshold=0.4)
                models_suite = [("jump", jump_model)] + models_suite

            ensemble = EnsemblePredictor(models_suite)
            ensemble.fit(X_tr_arr, y_tr, X_val_arr, y_val)

            val_pred = ensemble.predict(X_val_arr)
            val_pnl  = val_pred * y_val
            all_val_pnl.append(val_pnl)

            sh  = sharpe(val_pnl)
            sc  = selection_score(val_pnl, penalty=self.PENALTY_VAR)
            log(f"  → Val Sharpe={sh:.3f}  SelectScore={sc:.3f}", "OK")

            self.per_asset_models[asset] = ensemble

        # Overall portfolio Sharpe on val
        if all_val_pnl:
            min_len   = min(len(p) for p in all_val_pnl)
            pnl_mat   = np.stack([p[:min_len] for p in all_val_pnl], axis=1)
            port_ret  = pnl_mat.mean(axis=1)
            log(f"\nPortfolio Val Sharpe (equal-weight): {sharpe(port_ret):.3f}", "OK")

    # ──────────────────────────────────────────────────────────────────────
    def predict_returns(self, x_df: pd.DataFrame) -> pd.DataFrame:
        if not self.pred_cols:
            raise ValueError("Model assets are not initialized. Call load_data() first.")
        x_df = x_df.sort_values("date").reset_index(drop=True)
        X_eng = self.feat_eng.transform(x_df)
        X_arr = X_eng.values.astype(float)
        X_arr = (X_arr - self._feat_mean) / self._feat_std

        pred_matrix = np.zeros((len(x_df), len(self.asset_cols)))
        for i, asset in enumerate(self.asset_cols):
            if asset in self.per_asset_models:
                pred_matrix[:, i] = self.per_asset_models[asset].predict(X_arr)

        return pd.DataFrame(pred_matrix, columns=self.pred_cols, index=x_df.index)

    # ──────────────────────────────────────────────────────────────────────
    def build_weights(self, pred_df: pd.DataFrame,
                      x_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Portfolio-optimised weights using rolling covariance + Sharpe maximisation.
        Falls back to L1-normalised predictions if optimisation is unstable.
        """
        raw_preds = pred_df.to_numpy(dtype=float)
        n, k = raw_preds.shape
        weights = np.zeros_like(raw_preds)

        # We need recent return history to estimate covariance.
        # Proxy: use pred_df itself as a noisy return proxy for covariance.
        for t in range(n):
            mu = raw_preds[t]
            # Rolling covariance from past predictions as return proxy
            start = max(0, t - self.cov_estimator.window)
            window_data = raw_preds[start:t] if t > 0 else raw_preds[[t]]
            Sigma = self.cov_estimator.estimate(window_data) if len(window_data) >= 3 else np.eye(k)
            try:
                w = self.optimizer.optimise(mu, Sigma)
            except Exception:
                w = mu / (np.abs(mu).sum() + 1e-8)
            weights[t] = w

        # Final L1 enforcement
        gross = np.abs(weights).sum(axis=1, keepdims=True)
        gross = np.where(gross > 1.0, gross, 1.0)
        weights = weights / gross
        return pd.DataFrame(weights, columns=self.weight_cols, index=pred_df.index)

    # ──────────────────────────────────────────────────────────────────────
    def build_submission(self, x_train: pd.DataFrame,
                         x_test: pd.DataFrame) -> pd.DataFrame:
        log("BUILDING SUBMISSION", "SECTION")
        pred_train = self.predict_returns(x_train)
        pred_test  = self.predict_returns(x_test)
        pred_all   = pd.concat([pred_train, pred_test], axis=0).reset_index(drop=True)

        x_all = pd.concat([x_train, x_test], axis=0).reset_index(drop=True)
        weights_all = self.build_weights(pred_all, x_all).reset_index(drop=True)

        index_all = pd.concat(
            [x_train[["id", "date"]].copy(), x_test[["id", "date"]].copy()], axis=0
        ).reset_index(drop=True)

        submission = pd.concat([index_all, pred_all, weights_all], axis=1)

        # Fill any remaining NaN with 0
        submission = submission.fillna(0.0)
        log(f"Submission shape: {submission.shape}", "OK")
        return submission

    # ──────────────────────────────────────────────────────────────────────
    def save_submission(self, submission: pd.DataFrame,
                        path: str = "submission.parquet") -> Path:
        path = Path(path)
        submission.to_parquet(path, index=False)
        log(f"Saved → {path}", "OK")
        return path

    # ──────────────────────────────────────────────────────────────────────
    def fit_predict_save(self, data_dir: str = "data",
                         output_path: str = "submission.parquet") -> pd.DataFrame:
        log("LOADING DATA", "SECTION")
        data = self.load_data(data_dir)
        log("FITTING MODELS", "SECTION")
        self.fit(data["X_train"], data["R_train"])
        submission = self.build_submission(data["X_train"], data["X_test"])
        # Validate
        scorer = self.create_scorer()
        scorer.validate_submission_format(submission)
        log("Submission validated ✓", "OK")
        self.save_submission(submission, output_path)
        return submission

    # ──────────────────────────────────────────────────────────────────────
    def create_scorer(self) -> SubmissionScorerTemplate:
        scorer = SubmissionScorerTemplate()
        scorer.set_assets(self.asset_cols)
        return scorer

    # ──────────────────────────────────────────────────────────────────────
    def evaluate_on_train(self, submission: pd.DataFrame,
                          r_train: pd.DataFrame) -> dict:
        """
        Full evaluation report:
          - MSE per asset
          - Portfolio Sharpe, Sharpe std, selection score
        """
        log("EVALUATION REPORT", "SECTION")
        scorer = self.create_scorer()
        mse_report = scorer.score_predictions(submission, r_train)
        log(f"Overall MSE={mse_report['mse']:.6f}  n_rows={mse_report['n_rows']}", "OK")

        # Portfolio PnL: weight × realised return
        w_cols = self.weight_cols
        r_cols = self.asset_cols
        merged = r_train[["id", "date"] + r_cols].merge(
            submission[["id", "date"] + w_cols], on=["id", "date"], how="inner"
        )
        if merged.empty:
            log("No overlap for portfolio Sharpe calculation.", "WARN")
            return mse_report

        W = merged[w_cols].values
        R = merged[r_cols].values
        port_ret = (W * R).sum(axis=1)

        sh = sharpe(port_ret)
        sh_std = rolling_sharpe_std(port_ret)
        sc = selection_score(port_ret, penalty=self.PENALTY_VAR)

        log(f"Portfolio Sharpe     = {sh:.4f}")
        log(f"Rolling Sharpe Std   = {sh_std:.4f}  (lower is better)")
        log(f"Selection Score      = {sc:.4f}  (Sharpe - {self.PENALTY_VAR}×Std)")

        # Per-asset breakdown
        log("\nPer-asset Sharpe (weight × return):")
        for asset, wcol in zip(r_cols, w_cols):
            if asset in merged.columns and wcol in merged.columns:
                pnl_a = merged[wcol].values * merged[asset].values
                log(f"  {asset:12s}: Sharpe={sharpe(pnl_a):.3f}")

        return {
            **mse_report,
            "sharpe"          : sh,
            "sharpe_std"      : sh_std,
            "selection_score" : sc,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("QUANTITATIVE PORTFOLIO CHALLENGE — ADVANCED MODEL", "SECTION")
    model = PortfolioChallengeModel()

    # ── Full pipeline
    data = model.load_data(data_dir="data")

    log("FITTING", "SECTION")
    model.fit(data["X_train"], data["R_train"])

    submission = model.build_submission(data["X_train"], data["X_test"])

    # ── Evaluate on training set (informative; not used for selection)
    model.evaluate_on_train(submission, data["R_train"])

    # ── Validate & save
    scorer = model.create_scorer()
    scorer.validate_submission_format(submission)
    log("Format validation passed", "OK")

    out = model.save_submission(submission, "submission.parquet")
    log(f"\n🏆  Final submission saved: {out}", "OK")
    log(f"    Columns : {list(submission.columns[:6])} ... ({len(submission.columns)} total)")
    log(f"    Rows    : {len(submission)}")
    log(f"    Weight L1 max: {submission[model.weight_cols].abs().sum(axis=1).max():.4f} (≤1 required)")


if __name__ == "__main__":
    main()
