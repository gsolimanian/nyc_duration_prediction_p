import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import lightgbm as lgb
import mlflow
import optuna
import pandas as pd
from category_encoders import TargetEncoder
from mlflow.models import infer_signature
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


RAW_FEATURE_COLUMNS = [
    "hour_category",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "pickup_dayofweek",
    "pickup_hour",
    "is_rush_hour",
    "is_weekend",
]

TARGET_COLUMN = "duration"


def extract_month_from_path(file_path: str) -> int:
    """
    Extract month from a filename like:
    yellow_tripdata_2026-01_processed.parquet -> 1
    """
    stem = Path(file_path).stem
    match = re.search(r"(\d{4})-(\d{2})", stem)
    if match:
        return int(match.group(2))

    match = re.search(r"_(\d{2})(?:_|$)", stem)
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not extract month from file name: {file_path}")


def load_and_combine_months(file_paths: List[str]) -> pd.DataFrame:
    """
    Load multiple monthly parquet files and combine them into one DataFrame.
    Adds a 'month' column for time-based splitting.
    """
    if not file_paths:
        raise ValueError("No input files were provided.")

    dfs = []
    for path in file_paths:
        print(f"Loading: {path}")
        df = pd.read_parquet(path)
        df["month"] = extract_month_from_path(path)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total combined data shape: {combined.shape}")
    return combined


def split_train_validation_by_month(
    df: pd.DataFrame,
    train_months: List[int],
    val_month: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and validation based on month.
    """
    train_df = df[df["month"].isin(train_months)].copy()
    val_df = df[df["month"] == val_month].copy()

    if train_df.empty:
        raise ValueError(f"Train split is empty for months: {train_months}")
    if val_df.empty:
        raise ValueError(f"Validation split is empty for month: {val_month}")

    print(f"Train shape: {train_df.shape} | Months: {train_months}")
    print(f"Validation shape: {val_df.shape} | Month: {val_month}")

    return train_df, val_df


def validate_required_columns(df: pd.DataFrame, required_columns: List[str], stage: str = "") -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        prefix = f"[{stage}] " if stage else ""
        raise ValueError(f"{prefix}Missing required columns: {missing}")


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Keep only the columns needed for training.
    """
    required_columns = RAW_FEATURE_COLUMNS + [TARGET_COLUMN]
    validate_required_columns(df, required_columns, stage="select_features")
    return df[required_columns].copy(), TARGET_COLUMN


class CreatePUDo(BaseEstimator, TransformerMixin):
    """
    Create origin-destination pair feature.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        validate_required_columns(X, ["PULocationID", "DOLocationID"], stage="CreatePUDo")
        X["PU_DO"] = X["PULocationID"].astype(str) + "_" + X["DOLocationID"].astype(str)
        return X


class TargetEncodePUDo(BaseEstimator, TransformerMixin):
    """
    Target encode PU_DO using the training target only.
    """

    def __init__(self, smoothing: int = 15, min_samples_leaf: int = 30):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encoder = None

    def fit(self, X, y):
        validate_required_columns(X, ["PU_DO"], stage="TargetEncodePUDo.fit")
        self.encoder = TargetEncoder(
            cols=["PU_DO"],
            smoothing=self.smoothing,
            min_samples_leaf=self.min_samples_leaf,
        )
        self.encoder.fit(X[["PU_DO"]], y)
        return self

    def transform(self, X):
        X = X.copy()
        validate_required_columns(X, ["PU_DO"], stage="TargetEncodePUDo.transform")
        encoded = self.encoder.transform(X[["PU_DO"]]).iloc[:, 0].astype(float).values
        X["PU_DO_encoded"] = encoded
        return X


class FeatureCombiner(BaseEstimator, TransformerMixin):
    """
    One-hot encode hour_category and combine with numeric features.
    """

    def __init__(self):
        self.dv = DictVectorizer(sparse=True)
        self.categorical_features = ["hour_category"]
        self.numeric_features = [
            "trip_distance",
            "pickup_dayofweek",
            "pickup_hour",
            "is_rush_hour",
            "is_weekend",
            "PU_DO_encoded",
        ]

    def fit(self, X, y=None):
        validate_required_columns(
            X,
            self.categorical_features + self.numeric_features,
            stage="FeatureCombiner.fit",
        )
        cat_dicts = X[self.categorical_features].to_dict(orient="records")
        self.dv.fit(cat_dicts)
        return self

    def transform(self, X):
        validate_required_columns(
            X,
            self.categorical_features + self.numeric_features,
            stage="FeatureCombiner.transform",
        )

        cat_dicts = X[self.categorical_features].to_dict(orient="records")
        X_cat = self.dv.transform(cat_dicts)

        X_num = csr_matrix(X[self.numeric_features].astype(float).to_numpy())
        return hstack([X_cat, X_num], format="csr")


def create_taxi_pipeline(params: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("create_pu_do", CreatePUDo()),
            ("target_encode_pu_do", TargetEncodePUDo()),
            ("feature_combiner", FeatureCombiner()),
            ("model", lgb.LGBMRegressor(**params)),
        ]
    )


def build_lgbm_params(best_params: dict) -> dict:
    params = best_params.copy()
    params.update(
        {
            "objective": "regression",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
    )
    return params


def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def objective(trial, X_train, X_val, y_train, y_val):
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

        pipeline = create_taxi_pipeline(params)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        score = rmse(y_val, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("rmse", score)

        return score


def run_lightgbm_optuna_tuning(
    X_train,
    X_val,
    y_train,
    y_val,
    n_trials: int = 50
):
    with mlflow.start_run(run_name="lightgbm_optuna_tuning"):

        study = optuna.create_study(
            direction="minimize",
            study_name="LightGBM_Hyperparameter_Tuning",
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            lambda trial: objective(trial, X_train, X_val, y_train, y_val),
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        print("\n" + "=" * 60)
        print(f"Best RMSE: {study.best_value:.4f}")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        print("=" * 60)

        mlflow.log_metric("best_rmse", study.best_value)
        mlflow.log_params(study.best_params)

        return study.best_params, study.best_value


def train_final_model(
    X_train,
    y_train,
    X_val,
    y_val,
    best_params: dict,
    local_model_path: Optional[str] = None,
):
    params = build_lgbm_params(best_params)
    pipeline = create_taxi_pipeline(params)

    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)
    train_rmse = rmse(y_train, train_pred)
    val_rmse = rmse(y_val, val_pred)

    signature = infer_signature(X_train, train_pred)

    with mlflow.start_run(run_name="final_lightgbm_model"):
        mlflow.log_params(params)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            serialization_format="cloudpickle",
        )

        if local_model_path:
            output_path = Path(local_model_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, output_path)
            mlflow.log_artifact(str(output_path), artifact_path="local_model")

    print(f"Final Train RMSE: {train_rmse:.4f}")
    print(f"Final Validation RMSE: {val_rmse:.4f}")

    return pipeline


def maybe_sample(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    if n is None or n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def main(
    file_paths: List[str],
    train_months: List[int],
    val_month: int,
    tracking_uri: str,
    experiment_name: str,
    n_trials: int,
    sample_train: int,
    sample_val: int,
    local_model_path: Optional[str] = None,
):
    mlflow.set_tracking_uri(tracking_uri)
    print("Tracking URI:", mlflow.get_tracking_uri())

    mlflow.set_experiment(experiment_name)

    df = load_and_combine_months(file_paths)
    train_df, val_df = split_train_validation_by_month(df, train_months, val_month)

    train_df = maybe_sample(train_df, sample_train)
    val_df = maybe_sample(val_df, sample_val)

    train_df, target_column = select_features(train_df)
    val_df, _ = select_features(val_df)

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    best_params, best_rmse = run_lightgbm_optuna_tuning(
        X_train, X_val, y_train, y_val, n_trials=n_trials
    )

    print(f"\nBest RMSE from Optuna: {best_rmse:.4f}")

    _ = train_final_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        best_params=best_params,
        local_model_path=local_model_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of processed parquet files",
    )

    parser.add_argument(
        "--train-months",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Training months, e.g. --train-months 1 2",
    )

    parser.add_argument(
        "--val-month",
        type=int,
        required=True,
        help="Validation month, e.g. 3",
    )

    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="MLflow tracking URI",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="nyc_duration_prediction_boosting_model_pipeline_v2",
        help="MLflow experiment name",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials",
    )

    parser.add_argument(
        "--sample-train",
        type=int,
        default=5000,
        help="Train sample size. Use 0 for full data.",
    )

    parser.add_argument(
        "--sample-val",
        type=int,
        default=500,
        help="Validation sample size. Use 0 for full data.",
    )

    parser.add_argument(
        "--local-model-path",
        type=str,
        default="models/nyc_taxi_lgbm_pipeline.joblib",
        help="Optional local path to save the trained pipeline",
    )

    args = parser.parse_args()

    main(
        file_paths=args.inputs,
        train_months=args.train_months,
        val_month=args.val_month,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        n_trials=args.n_trials,
        sample_train=args.sample_train,
        sample_val=args.sample_val,
        local_model_path=args.local_model_path,
    )