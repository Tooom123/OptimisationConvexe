import numpy as np
import pandas as pd
from pathlib import Path


def ledoit_wolf_shrinkage(X):
    """
    Shrinkage analytique de Ledoit-Wolf (Oracle Approximating Shrinkage).
    Retourne la covariance régularisée sans dépendance externe.
    """
    n, p = X.shape
    S = np.cov(X, rowvar=False)
    mu = np.trace(S) / p

    delta2 = np.sum(S ** 2) + mu ** 2 - 2 * mu * np.trace(S @ S) / p
    beta_bar = 0.0
    for i in range(n):
        xi = X[i] - X.mean(axis=0)
        outer = np.outer(xi, xi)
        beta_bar += np.sum((outer - S) ** 2)
    beta_bar /= (n ** 2)

    alpha = min(beta_bar / delta2, 1.0) if delta2 > 0 else 0.1
    return (1 - alpha) * S + alpha * mu * np.eye(p)


def mse_loss(residuals):
    return 0.5 * np.mean(residuals ** 2)

def mse_gradient(residuals, X):
    return X.T @ residuals / len(residuals)

def huber_loss(residuals, delta=1.0):
    abs_r = np.abs(residuals)
    return np.mean(np.where(abs_r <= delta, 0.5 * residuals ** 2, delta * (abs_r - 0.5 * delta)))

def huber_gradient(residuals, X, delta=1.0):
    pseudo_resid = np.where(np.abs(residuals) <= delta, residuals, delta * np.sign(residuals))
    return X.T @ pseudo_resid / len(residuals)


class GradientDescent:
    def __init__(self, lr=0.01, n_iter=500, loss='mse', delta=1.0, lambda_reg=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.loss = loss
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.theta = None
        self.loss_history = []

    def fit(self, X, y):
        _, d = X.shape
        self.theta = np.zeros(d)
        self.loss_history = []

        for _ in range(self.n_iter):
            residuals = X @ self.theta - y

            if self.loss == 'mse':
                grad = mse_gradient(residuals, X)
                l = mse_loss(residuals)
            else:
                grad = huber_gradient(residuals, X, self.delta)
                l = huber_loss(residuals, self.delta)

            reg = self.lambda_reg * self.theta
            reg[-1] = 0.0
            grad = grad + reg

            self.theta -= self.lr * grad
            self.loss_history.append(l)

        return self

    def predict(self, X):
        return X @ self.theta


class SGD:
    def __init__(self, lr=0.01, n_iter=500, loss='mse', delta=1.0, lambda_reg=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.loss = loss
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.theta = None
        self.loss_history = []

    def fit(self, X, y):
        n, d = X.shape
        self.theta = np.zeros(d)
        self.loss_history = []

        for _ in range(self.n_iter):
            idx = np.random.randint(0, n)
            xi = X[idx:idx+1]
            yi = y[idx:idx+1]

            residual = xi @ self.theta - yi

            if self.loss == 'mse':
                grad = mse_gradient(residual, xi)
            else:
                grad = huber_gradient(residual, xi, self.delta)

            reg = self.lambda_reg * self.theta
            reg[-1] = 0.0
            grad = grad + reg

            self.theta -= self.lr * grad

            full_resid = X @ self.theta - y
            l = mse_loss(full_resid) if self.loss == 'mse' else huber_loss(full_resid, self.delta)
            self.loss_history.append(l)

        return self

    def predict(self, X):
        return X @ self.theta


class MiniBatchSGD:
    def __init__(self, lr=0.01, n_iter=500, batch_size=64, loss='mse', delta=1.0, lambda_reg=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.loss = loss
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.theta = None
        self.loss_history = []

    def fit(self, X, y):
        n, d = X.shape
        self.theta = np.zeros(d)
        self.loss_history = []

        for _ in range(self.n_iter):
            indices = np.random.choice(n, size=min(self.batch_size, n), replace=False)
            X_batch, y_batch = X[indices], y[indices]

            residuals = X_batch @ self.theta - y_batch

            if self.loss == 'mse':
                grad = mse_gradient(residuals, X_batch)
            else:
                grad = huber_gradient(residuals, X_batch, self.delta)

            reg = self.lambda_reg * self.theta
            reg[-1] = 0.0
            grad = grad + reg

            self.theta -= self.lr * grad

            full_resid = X @ self.theta - y
            l = mse_loss(full_resid) if self.loss == 'mse' else huber_loss(full_resid, self.delta)
            self.loss_history.append(l)

        return self

    def predict(self, X):
        return X @ self.theta


class Adam:
    """
    Adam optimizer avec régularisation L2.
    Bien plus stable et rapide à converger que les variantes SGD basiques.
    """
    def __init__(self, lr=1e-3, n_iter=500, loss='mse', delta=1.0, lambda_reg=1e-3,
                 beta1=0.9, beta2=0.999, eps=1e-8, batch_size=64):
        self.lr = lr
        self.n_iter = n_iter
        self.loss = loss
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.batch_size = batch_size
        self.theta = None
        self.loss_history = []

    def fit(self, X, y):
        n, d = X.shape
        self.theta = np.zeros(d)
        m = np.zeros(d)
        v = np.zeros(d)
        self.loss_history = []

        for t in range(1, self.n_iter + 1):
            indices = np.random.choice(n, size=min(self.batch_size, n), replace=False)
            X_batch, y_batch = X[indices], y[indices]

            residuals = X_batch @ self.theta - y_batch

            if self.loss == 'mse':
                grad = mse_gradient(residuals, X_batch)
            else:
                grad = huber_gradient(residuals, X_batch, self.delta)

            reg = self.lambda_reg * self.theta
            reg[-1] = 0.0
            grad = grad + reg

            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad ** 2
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            self.theta -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            full_resid = X @ self.theta - y
            l = mse_loss(full_resid) if self.loss == 'mse' else huber_loss(full_resid, self.delta)
            self.loss_history.append(l)

        return self

    def predict(self, X):
        return X @ self.theta


class PortfolioChallengeModel:
    def __init__(self, optimizer='adam', loss='huber', lr=1e-3,
                 n_iter=500, batch_size=64, lam=0.5, lambda_reg=1e-2):
        self.optimizer_name = optimizer
        self.loss = loss
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.lam = lam
        self.lambda_reg = lambda_reg

        self.feature_cols = []
        self.asset_cols = []
        self.pred_cols = []
        self.weight_cols = []
        self.thetas = {}
        self.sigma_inv = None

    def read_table(self, path):
        return pd.read_parquet(Path(path))

    def load_data(self, data_dir="data"):
        data_dir = Path(data_dir)
        x_train = self.read_table(data_dir / "X_train.parquet")
        r_train = self.read_table(data_dir / "R_train.parquet")
        x_test  = self.read_table(data_dir / "X_test.parquet")

        self.feature_cols = [c for c in x_train.columns if c not in ["id", "date", "split"]]
        self.asset_cols   = [c for c in r_train.columns if c not in ["id", "date", "split"]]
        self.pred_cols    = [f"pred_{c}" for c in self.asset_cols]
        self.weight_cols  = [f"weight_{c}" for c in self.asset_cols]

        return {"X_train": x_train, "R_train": r_train, "X_test": x_test}

    def _make_optimizer(self):
        kwargs = dict(lr=self.lr, n_iter=self.n_iter, loss=self.loss, lambda_reg=self.lambda_reg)
        if self.optimizer_name == 'gd':
            return GradientDescent(**kwargs)
        elif self.optimizer_name == 'sgd':
            return SGD(**kwargs)
        elif self.optimizer_name == 'adam':
            return Adam(**kwargs, batch_size=self.batch_size)
        else:
            return MiniBatchSGD(**kwargs, batch_size=self.batch_size)

    def fit(self, x_train, r_train):
        X = x_train[self.feature_cols].values.astype(float)
        self.X_mean = X.mean(axis=0)
        self.X_std  = X.std(axis=0) + 1e-8
        X_norm = (X - self.X_mean) / self.X_std
        X_norm = np.hstack([X_norm, np.ones((len(X_norm), 1))])

        self.thetas = {}
        for asset in self.asset_cols:
            y = r_train[asset].values.astype(float)
            opt = self._make_optimizer()
            opt.fit(X_norm, y)
            self.thetas[asset] = opt.theta

        preds = self._predict_raw(X_norm)
        residuals = r_train[self.asset_cols].values - preds

        self.sigma = ledoit_wolf_shrinkage(residuals)
        self.sigma_inv = np.linalg.inv(self.sigma)

        print(f"Modèle entraîné : {self.optimizer_name} | loss={self.loss} "
              f"| lambda_reg={self.lambda_reg} | {len(self.asset_cols)} actifs")

    def _normalize(self, X):
        X_norm = (X - self.X_mean) / self.X_std
        return np.hstack([X_norm, np.ones((len(X_norm), 1))])

    def _predict_raw(self, X_norm):
        return np.column_stack([X_norm @ self.thetas[a] for a in self.asset_cols])

    def predict_returns(self, x_df):
        X_norm = self._normalize(x_df[self.feature_cols].values.astype(float))
        preds = self._predict_raw(X_norm)
        return pd.DataFrame(preds, columns=self.pred_cols, index=x_df.index)

    def build_weights(self, pred_df):
        mu = pred_df[self.pred_cols].values
        weights = np.zeros_like(mu)

        for t in range(len(mu)):
            w = self.lam * self.sigma_inv @ mu[t]
            w = np.clip(w, -1, 1)
            gross = np.abs(w).sum()
            if gross > 1.0:
                w = w / gross
            weights[t] = w

        return pd.DataFrame(weights, columns=self.weight_cols, index=pred_df.index)

    def build_submission(self, x_train, x_test):
        pred_train = self.predict_returns(x_train)
        pred_test  = self.predict_returns(x_test)
        pred_all   = pd.concat([pred_train, pred_test]).reset_index(drop=True)
        weights_all = self.build_weights(pred_all).reset_index(drop=True)
        index_all = pd.concat(
            [x_train[["id", "date"]], x_test[["id", "date"]]]
        ).reset_index(drop=True)
        return pd.concat([index_all, pred_all, weights_all], axis=1)

    def save_submission(self, submission, path="submission.parquet"):
        submission.to_parquet(Path(path), index=False)
        return path

    def fit_predict_save(self, data_dir="data", output_path="submission.parquet"):
        data = self.load_data(data_dir)
        self.fit(data["X_train"], data["R_train"])
        submission = self.build_submission(data["X_train"], data["X_test"])
        self.save_submission(submission, output_path)
        return submission


def compare_optimizers(X, y, n_iter=300, lr=0.005):
    results = {}
    for name, opt in [
        ('GD',          GradientDescent(lr=lr, n_iter=n_iter, loss='mse')),
        ('SGD',         SGD(lr=lr, n_iter=n_iter, loss='mse')),
        ('MiniBatch64', MiniBatchSGD(lr=lr, n_iter=n_iter, batch_size=64, loss='mse')),
        ('Adam',        Adam(lr=1e-3, n_iter=n_iter, loss='mse')),
        ('GD-Huber',    GradientDescent(lr=lr, n_iter=n_iter, loss='huber')),
    ]:
        opt.fit(X, y)
        results[name] = opt.loss_history
        print(f"{name:15s} | loss finale = {opt.loss_history[-1]:.6f}")
    return results


if __name__ == "__main__":
    model = PortfolioChallengeModel(
        optimizer='adam',
        loss='huber',
        lr=1e-3,
        n_iter=500,
        batch_size=64,
        lam=0.5,
        lambda_reg=1e-2,
    )
    submission = model.fit_predict_save(
        data_dir="data",
        output_path="submission.parquet"
    )
    print("Submission générée :", submission.shape)
    print(submission.head())
