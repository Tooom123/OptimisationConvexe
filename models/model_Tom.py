import numpy as np
import pandas as pd
from pathlib import Path

class SubmissionScorerTemplate:
    def __init__(self):
        self.asset_cols = []
        self.pred_cols = []
        self.weight_cols = []

    def set_assets(self, asset_cols):
        self.asset_cols = list(asset_cols)
        self.pred_cols = [f"pred_{col}" for col in self.asset_cols]
        self.weight_cols = [f"weight_{col}" for col in self.asset_cols]

    def validate_submission_format(self, submission):
        required = ["id", "date"] + self.pred_cols + self.weight_cols
        missing = [col for col in required if col not in submission.columns]
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
            predictions[["id", "date"] + pred_cols],
            on=["id", "date"],
            how="inner",
        )
        if merged.empty:
            raise ValueError("No overlapping rows found between predictions and realized returns.")

        y_true = merged[common_assets].to_numpy(dtype=float)
        y_pred = merged[pred_cols].to_numpy(dtype=float)
        mse = float(np.mean((y_true - y_pred) ** 2))
        return {"mse": mse, "n_rows": int(len(merged))}

class PortfolioChallengeModel:
    def __init__(self):
        self.feature_cols = []
        self.asset_cols = []
        self.pred_cols = []
        self.weight_cols = []

    def read_table(self, path):
        path = Path(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file format: {path}. Expected .parquet")

    def load_data(self, data_dir="data"):
        data_dir = Path(data_dir)
        x_train = self.read_table(data_dir / "X_train.parquet")
        r_train = self.read_table(data_dir / "R_train.parquet")
        x_test = self.read_table(data_dir / "X_test.parquet")

        self.feature_cols = [col for col in x_train.columns if col not in ["id", "date", "split"]]
        self.asset_cols = [col for col in r_train.columns if col not in ["id", "date", "split"]]
        self.pred_cols = [f"pred_{col}" for col in self.asset_cols]
        self.weight_cols = [f"weight_{col}" for col in self.asset_cols]

        return {
            "X_train": x_train,
            "R_train": r_train,
            "X_test": x_test,
        }

    def fit(self, x_train, r_train):
        pass

    def predict_returns(self, x_df):
        if not self.pred_cols:
            raise ValueError("Model assets are not initialized. Call load_data() first.")
        predictions = np.zeros((len(x_df), len(self.pred_cols)), dtype=float)
        return pd.DataFrame(predictions, columns=self.pred_cols, index=x_df.index)

    def build_weights(self, pred_df):
        raw = pred_df.to_numpy(dtype=float)
        gross = np.abs(raw).sum(axis=1, keepdims=True)
        gross = np.where(gross > 1.0, gross, 1.0)
        weights = raw / gross
        return pd.DataFrame(weights, columns=self.weight_cols, index=pred_df.index)

    def build_submission(self, x_train, x_test):
        pred_train = self.predict_returns(x_train)
        pred_test = self.predict_returns(x_test)
        pred_all = pd.concat([pred_train, pred_test], axis=0).reset_index(drop=True)

        weights_all = self.build_weights(pred_all).reset_index(drop=True)
        index_all = pd.concat(
            [x_train[["id", "date"]].copy(), x_test[["id", "date"]].copy()],
            axis=0,
        ).reset_index(drop=True)

        return pd.concat([index_all, pred_all, weights_all], axis=1)

    def save_submission(self, submission, path="submission.parquet"):
        path = Path(path)
        submission.to_parquet(path, index=False)
        return path

    def fit_predict_save(self, data_dir="data", output_path="submission.parquet"):
        data = self.load_data(data_dir)
        self.fit(data["X_train"], data["R_train"])
        submission = self.build_submission(data["X_train"], data["X_test"])
        self.save_submission(submission, output_path)
        return submission

    def create_scorer(self):
        scorer = SubmissionScorerTemplate()
        scorer.set_assets(self.asset_cols)
        return scorer

def main():
    model = PortfolioChallengeModel()
    model.fit_predict_save(data_dir="data", output_path="submission.parquet")

if __name__ == "__main__":
    main()
