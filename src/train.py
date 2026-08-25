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


# =========================================================
# Configuration
# =========================================================

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


# =========================================================
# Utility functions
# =========================================================

def extract_month_from_path(file_path: str) -> int:
    """
    Extract month from file names such as:

    yellow_tripdata_2026-01_processed.parquet
    yellow_tripdata_2026-04_processed.parquet
    """

    stem = Path(file_path).stem

    # Example: 2026-01
    match = re.search(r"(\d{4})-(\d{2})", stem)

    if match:
        month = int(match.group(2))

        if not 1 <= month <= 12:
            raise ValueError(
                f"Invalid month in file name: {file_path}"
            )

        return month

    # Fallback for names such as:
    # yellow_tripdata_01_processed.parquet
    match = re.search(r"_(\d{2})(?:_|$)", stem)

    if match:
        month = int(match.group(1))

        if not 1 <= month <= 12:
            raise ValueError(
                f"Invalid month in file name: {file_path}"
            )

        return month

    raise ValueError(
        f"Could not extract month from file name: {file_path}"
    )


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    stage: str = "",
) -> None:
    """
    Check whether required columns exist in the DataFrame.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        prefix = f"[{stage}] " if stage else ""

        raise ValueError(
            f"{prefix}Missing required columns: "
            f"{missing_columns}"
        )


def rmse(y_true, y_pred) -> float:
    """
    Calculate Root Mean Squared Error.
    """

    return mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5


def maybe_sample(
    df: pd.DataFrame,
    n: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample the DataFrame for Optuna tuning.

    If n <= 0, the full DataFrame is returned.
    """

    if n is None or n <= 0 or n >= len(df):
        return df.reset_index(drop=True)

    return df.sample(
        n=n,
        random_state=random_state,
    ).reset_index(drop=True)


# =========================================================
# Loading monthly data
# =========================================================

def load_and_combine_months(
    file_paths: List[str],
) -> pd.DataFrame:
    """
    Load monthly processed Parquet files and combine them.

    A month column is added based on the file name.
    """

    if not file_paths:
        raise ValueError(
            "No input files were provided."
        )

    dataframes = []
    loaded_months = set()

    for file_path in file_paths:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Input file does not exist: {path}"
            )

        month = extract_month_from_path(
            file_path
        )

        if month in loaded_months:
            raise ValueError(
                f"Duplicate file for month {month}: "
                f"{file_path}"
            )

        print(f"Loading: {path}")

        df = pd.read_parquet(path)

        if df.empty:
            raise ValueError(
                f"Input file is empty: {path}"
            )

        validate_required_columns(
            df,
            RAW_FEATURE_COLUMNS + [TARGET_COLUMN],
            stage=f"month_{month}",
        )

        df["month"] = month

        dataframes.append(df)
        loaded_months.add(month)

        print(
            f"Month {month} loaded successfully. "
            f"Shape: {df.shape}"
        )

    combined_df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    print("\n" + "=" * 60)
    print(
        f"Total combined data shape: "
        f"{combined_df.shape}"
    )
    print(
        f"Loaded months: "
        f"{sorted(loaded_months)}"
    )
    print("=" * 60)

    return combined_df


# =========================================================
# Train / Validation / Test split
# =========================================================

def split_train_validation_test_by_month(
    df: pd.DataFrame,
    train_months: List[int],
    val_month: int,
    test_month: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split the data by month.

    Example:

        train_months = [1, 2]
        val_month = 3
        test_month = 4

    Result:

        Train      -> months 1 and 2
        Validation -> month 3
        Test       -> month 4
    """

    if "month" not in df.columns:
        raise ValueError(
            "The DataFrame must contain a 'month' column."
        )

    if not train_months:
        raise ValueError(
            "train_months cannot be empty."
        )

    if len(set(train_months)) != len(train_months):
        raise ValueError(
            "Duplicate months found in train_months."
        )

    if val_month in train_months:
        raise ValueError(
            "Validation month cannot be included "
            "in train_months."
        )

    if test_month in train_months:
        raise ValueError(
            "Test month cannot be included "
            "in train_months."
        )

    if val_month == test_month:
        raise ValueError(
            "Validation month and test month "
            "must be different."
        )

    train_df = df[
        df["month"].isin(train_months)
    ].copy()

    val_df = df[
        df["month"] == val_month
    ].copy()

    test_df = df[
        df["month"] == test_month
    ].copy()

    if train_df.empty:
        raise ValueError(
            f"Train split is empty for months: "
            f"{train_months}"
        )

    if val_df.empty:
        raise ValueError(
            f"Validation split is empty for month: "
            f"{val_month}"
        )

    if test_df.empty:
        raise ValueError(
            f"Test split is empty for month: "
            f"{test_month}"
        )

    print("\n" + "=" * 60)
    print("Time-based data split completed")
    print("=" * 60)

    print(
        f"Train      | months={train_months} "
        f"| shape={train_df.shape}"
    )

    print(
        f"Validation | month={val_month} "
        f"| shape={val_df.shape}"
    )

    print(
        f"Test       | month={test_month} "
        f"| shape={test_df.shape}"
    )

    print("=" * 60)

    return train_df, val_df, test_df


# =========================================================
# Feature selection
# =========================================================

def select_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, str]:
    """
    Select features and target column.
    """

    required_columns = RAW_FEATURE_COLUMNS + [
        TARGET_COLUMN
    ]

    validate_required_columns(
        df,
        required_columns,
        stage="select_features",
    )

    selected_df = df[
        required_columns
    ].copy()

    return selected_df, TARGET_COLUMN


# =========================================================
# Custom transformers
# =========================================================

class CreatePUDo(
    BaseEstimator,
    TransformerMixin,
):
    """
    Create pickup-dropoff pair feature.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        validate_required_columns(
            X,
            [
                "PULocationID",
                "DOLocationID",
            ],
            stage="CreatePUDo",
        )

        X["PU_DO"] = (
            X["PULocationID"].astype(str)
            + "_"
            + X["DOLocationID"].astype(str)
        )

        return X


class TargetEncodePUDo(
    BaseEstimator,
    TransformerMixin,
):
    """
    Target encode PU_DO.

    The encoder is fitted only on the data passed to
    pipeline.fit(), so validation and test targets are
    not used during fitting.
    """

    def __init__(
        self,
        smoothing: int = 15,
        min_samples_leaf: int = 30,
    ):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encoder = None

    def fit(self, X, y):
        validate_required_columns(
            X,
            ["PU_DO"],
            stage="TargetEncodePUDo.fit",
        )

        self.encoder = TargetEncoder(
            cols=["PU_DO"],
            smoothing=self.smoothing,
            min_samples_leaf=self.min_samples_leaf,
        )

        self.encoder.fit(
            X[["PU_DO"]],
            y,
        )

        return self

    def transform(self, X):
        if self.encoder is None:
            raise RuntimeError(
                "TargetEncodePUDo must be fitted "
                "before transform."
            )

        X = X.copy()

        validate_required_columns(
            X,
            ["PU_DO"],
            stage="TargetEncodePUDo.transform",
        )

        encoded_values = (
            self.encoder
            .transform(X[["PU_DO"]])
            .iloc[:, 0]
            .astype(float)
            .to_numpy()
        )

        X["PU_DO_encoded"] = encoded_values

        return X


class FeatureCombiner(
    BaseEstimator,
    TransformerMixin,
):
    """
    One-hot encode hour_category and combine it with
    numerical features.
    """

    def __init__(self):
        self.dv = DictVectorizer(
            sparse=True
        )

        self.categorical_features = [
            "hour_category"
        ]

        self.numeric_features = [
            "trip_distance",
            "pickup_dayofweek",
            "pickup_hour",
            "is_rush_hour",
            "is_weekend",
            "PU_DO_encoded",
        ]

    def fit(self, X, y=None):
        required_columns = (
            self.categorical_features
            + self.numeric_features
        )

        validate_required_columns(
            X,
            required_columns,
            stage="FeatureCombiner.fit",
        )

        categorical_records = (
            X[self.categorical_features]
            .astype(str)
            .to_dict(orient="records")
        )

        self.dv.fit(
            categorical_records
        )

        return self

    def transform(self, X):
        required_columns = (
            self.categorical_features
            + self.numeric_features
        )

        validate_required_columns(
            X,
            required_columns,
            stage="FeatureCombiner.transform",
        )

        categorical_records = (
            X[self.categorical_features]
            .astype(str)
            .to_dict(orient="records")
        )

        X_categorical = self.dv.transform(
            categorical_records
        )

        X_numeric = csr_matrix(
            X[self.numeric_features]
            .astype(float)
            .to_numpy()
        )

        return hstack(
            [
                X_categorical,
                X_numeric,
            ],
            format="csr",
        )


# =========================================================
# Model pipeline
# =========================================================

def create_taxi_pipeline(
    params: dict,
) -> Pipeline:
    """
    Create the complete preprocessing and model pipeline.
    """

    return Pipeline(
        steps=[
            (
                "create_pu_do",
                CreatePUDo(),
            ),
            (
                "target_encode_pu_do",
                TargetEncodePUDo(),
            ),
            (
                "feature_combiner",
                FeatureCombiner(),
            ),
            (
                "model",
                lgb.LGBMRegressor(
                    **params
                ),
            ),
        ]
    )


def build_lgbm_params(
    best_params: dict,
) -> dict:
    """
    Add fixed parameters to Optuna parameters.
    """

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


# =========================================================
# Optuna objective
# =========================================================

def objective(
    trial,
    X_train,
    X_val,
    y_train,
    y_val,
) -> float:
    """
    Optuna objective function.

    For every trial:

        1. Fit the pipeline on train data.
        2. Predict on validation data.
        3. Calculate validation RMSE.
        4. Return RMSE to Optuna.

    Test data is not used here.
    """

    with mlflow.start_run(
        nested=True,
        run_name=f"trial_{trial.number}",
    ):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                300,
                2000,
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves",
                20,
                150,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                12,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction",
                0.6,
                1.0,
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction",
                0.6,
                1.0,
            ),
            "bagging_freq": trial.suggest_int(
                "bagging_freq",
                1,
                10,
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                10,
                100,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                10.0,
                log=True,
            ),
            "objective": "regression",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

        print(
            f"\nStarting Optuna trial "
            f"{trial.number}"
        )

        pipeline = create_taxi_pipeline(
            params
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        validation_predictions = pipeline.predict(
            X_val
        )

        validation_rmse = rmse(
            y_val,
            validation_predictions,
        )

        mlflow.log_params(
            params
        )

        mlflow.log_metric(
            "validation_rmse",
            validation_rmse,
        )

        print(
            f"Trial {trial.number} "
            f"Validation RMSE: "
            f"{validation_rmse:.4f}"
        )

        return validation_rmse


# =========================================================
# Optuna tuning
# =========================================================

def run_lightgbm_optuna_tuning(
    X_train,
    X_val,
    y_train,
    y_val,
    n_trials: int = 50,
) -> Tuple[dict, float]:
    """
    Tune LightGBM using train and validation data.

    The test data is not passed to this function.
    """

    with mlflow.start_run(
        run_name="lightgbm_optuna_tuning",
    ):
        study = optuna.create_study(
            direction="minimize",
            study_name=(
                "LightGBM_Hyperparameter_Tuning"
            ),
            sampler=optuna.samplers.TPESampler(
                seed=42
            ),
        )

        study.optimize(
            lambda trial: objective(
                trial=trial,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
            ),
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        best_params = study.best_params
        best_validation_rmse = study.best_value

        print("\n" + "=" * 60)
        print(
            f"Best Validation RMSE: "
            f"{best_validation_rmse:.4f}"
        )

        print("Best parameters:")

        for key, value in best_params.items():
            print(f"  {key}: {value}")

        print("=" * 60)

        mlflow.log_params(
            best_params
        )

        mlflow.log_metric(
            "best_validation_rmse",
            best_validation_rmse,
        )

        return (
            best_params,
            best_validation_rmse,
        )


# =========================================================
# Final training and test evaluation
# =========================================================

def train_final_model(
    X_train_val,
    y_train_val,
    X_test,
    y_test,
    best_params: dict,
    best_validation_rmse: float,
    test_month: int,
    local_model_path: Optional[str] = None,
):
    """
    Train the final model on months 1, 2 and 3.

    Then evaluate it on month 4.

    The test set is not used during hyperparameter
    tuning or final model fitting.
    """

    final_params = build_lgbm_params(
        best_params
    )

    pipeline = create_taxi_pipeline(
        final_params
    )

    print(
        "\nTraining final model on "
        "months 1, 2 and 3..."
    )

    pipeline.fit(
        X_train_val,
        y_train_val,
    )

    print(
        f"Evaluating final model on "
        f"month {test_month}..."
    )

    train_val_predictions = pipeline.predict(
        X_train_val
    )

    test_predictions = pipeline.predict(
        X_test
    )

    train_val_rmse = rmse(
        y_train_val,
        train_val_predictions,
    )

    test_rmse = rmse(
        y_test,
        test_predictions,
    )

    # MLflow model signature
    signature_input = X_train_val.head(5)

    signature_predictions = pipeline.predict(
        signature_input
    )

    signature = infer_signature(
        signature_input,
        signature_predictions,
    )

    with mlflow.start_run(
        run_name="final_lightgbm_model",
    ):
        mlflow.log_params(
            final_params
        )

        mlflow.log_metric(
            "best_validation_rmse_during_tuning",
            best_validation_rmse,
        )

        mlflow.log_metric(
            "train_validation_rmse",
            train_val_rmse,
        )

        mlflow.log_metric(
            "final_test_rmse",
            test_rmse,
        )

        mlflow.log_param(
            "final_training_months",
            "1,2,3",
        )

        mlflow.log_param(
            "test_month",
            str(test_month),
        )

        mlflow.log_param(
            "final_training_rows",
            str(len(X_train_val)),
        )

        mlflow.log_param(
            "test_rows",
            str(len(X_test)),
        )

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            serialization_format="cloudpickle",
        )

        if local_model_path:
            output_path = Path(
                local_model_path
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            joblib.dump(
                pipeline,
                output_path,
            )

            mlflow.log_artifact(
                str(output_path),
                artifact_path="local_model",
            )

    print("\n" + "=" * 60)
    print(
        f"Best Validation RMSE: "
        f"{best_validation_rmse:.4f}"
    )
    print(
        f"Train + Validation RMSE: "
        f"{train_val_rmse:.4f}"
    )
    print(
        f"Final Test RMSE on month {test_month}: "
        f"{test_rmse:.4f}"
    )
    print("=" * 60)

    return pipeline, test_rmse


# =========================================================
# Main pipeline
# =========================================================

def main(
    file_paths: List[str],
    train_months: List[int],
    val_month: int,
    test_month: int,
    tracking_uri: str,
    experiment_name: str,
    n_trials: int,
    sample_train: int,
    sample_val: int,
    local_model_path: Optional[str] = None,
):
    """
    Complete workflow:

        Train:
            months 1 and 2

        Validation:
            month 3

        Test:
            month 4

        Final model:
            trained on months 1, 2 and 3
            evaluated on month 4
    """

    mlflow.set_tracking_uri(
        tracking_uri
    )

    print(
        "MLflow Tracking URI:",
        mlflow.get_tracking_uri(),
    )

    mlflow.set_experiment(
        experiment_name
    )

    # -----------------------------------------------------
    # Load all files
    # -----------------------------------------------------

    df = load_and_combine_months(
        file_paths
    )

    # -----------------------------------------------------
    # Split monthly data
    # -----------------------------------------------------

    train_df, val_df, test_df = (
        split_train_validation_test_by_month(
            df=df,
            train_months=train_months,
            val_month=val_month,
            test_month=test_month,
        )
    )

    # -----------------------------------------------------
    # Select features
    # -----------------------------------------------------

    train_df, target_column = select_features(
        train_df
    )

    val_df, _ = select_features(
        val_df
    )

    test_df, _ = select_features(
        test_df
    )

    # -----------------------------------------------------
    # Create tuning samples
    #
    # These samples are used only by Optuna.
    # -----------------------------------------------------

    tuning_train_df = maybe_sample(
        train_df,
        sample_train,
    )

    tuning_val_df = maybe_sample(
        val_df,
        sample_val,
    )

    X_tuning_train = tuning_train_df.drop(
        columns=[target_column]
    )

    y_tuning_train = tuning_train_df[
        target_column
    ]

    X_tuning_val = tuning_val_df.drop(
        columns=[target_column]
    )

    y_tuning_val = tuning_val_df[
        target_column
    ]

    print("\n" + "=" * 60)
    print("Data used for Optuna tuning")
    print("=" * 60)
    print(
        f"Tuning train shape: "
        f"{X_tuning_train.shape}"
    )
    print(
        f"Tuning validation shape: "
        f"{X_tuning_val.shape}"
    )
    print("=" * 60)

    # -----------------------------------------------------
    # Hyperparameter tuning
    #
    # Only train and validation are used here.
    # Test month is not used.
    # -----------------------------------------------------

    best_params, best_validation_rmse = (
        run_lightgbm_optuna_tuning(
            X_train=X_tuning_train,
            X_val=X_tuning_val,
            y_train=y_tuning_train,
            y_val=y_tuning_val,
            n_trials=n_trials,
        )
    )

    print(
        f"\nBest Validation RMSE from Optuna: "
        f"{best_validation_rmse:.4f}"
    )

    # -----------------------------------------------------
    # Combine full months 1, 2 and 3
    #
    # The final model uses all rows, not only the
    # sampled rows used during Optuna.
    # -----------------------------------------------------

    train_val_df = pd.concat(
        [
            train_df,
            val_df,
        ],
        ignore_index=True,
    )

    X_train_val = train_val_df.drop(
        columns=[target_column]
    )

    y_train_val = train_val_df[
        target_column
    ]

    # -----------------------------------------------------
    # Prepare untouched month 4 test data
    # -----------------------------------------------------

    X_test = test_df.drop(
        columns=[target_column]
    )

    y_test = test_df[
        target_column
    ]

    # -----------------------------------------------------
    # Final training and test evaluation
    # -----------------------------------------------------

    final_pipeline, test_rmse = (
        train_final_model(
            X_train_val=X_train_val,
            y_train_val=y_train_val,
            X_test=X_test,
            y_test=y_test,
            best_params=best_params,
            best_validation_rmse=best_validation_rmse,
            test_month=test_month,
            local_model_path=local_model_path,
        )
    )

    print(
        "\nTraining pipeline completed successfully."
    )

    print(
        f"Final Test RMSE on month {test_month}: "
        f"{test_rmse:.4f}"
    )

    return final_pipeline


# =========================================================
# Command-line interface
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Train LightGBM with a time-based "
            "train-validation-test split."
        )
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "Processed Parquet files for months "
            "1, 2, 3 and 4."
        ),
    )

    parser.add_argument(
        "--train-months",
        nargs="+",
        type=int,
        default=[1, 2],
        help=(
            "Training months. Default: 1 2"
        ),
    )

    parser.add_argument(
        "--val-month",
        type=int,
        default=3,
        help=(
            "Validation month. Default: 3"
        ),
    )

    parser.add_argument(
        "--test-month",
        type=int,
        default=4,
        help=(
            "Test month. Default: 4"
        ),
    )

    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://127.0.0.1:5000",
        help=(
            "MLflow tracking URI."
        ),
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=(
            "nyc_duration_prediction_train_val_test"
        ),
        help=(
            "MLflow experiment name."
        ),
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help=(
            "Number of Optuna trials. "
            "Default: 50."
        ),
    )

    parser.add_argument(
        "--sample-train",
        type=int,
        default=5000,
        help=(
            "Number of train rows for Optuna. "
            "Use 0 for all train rows."
        ),
    )

    parser.add_argument(
        "--sample-val",
        type=int,
        default=500,
        help=(
            "Number of validation rows for Optuna. "
            "Use 0 for all validation rows."
        ),
    )

    parser.add_argument(
        "--local-model-path",
        type=str,
        default=(
            "models/nyc_taxi_lgbm_pipeline_final.joblib"
        ),
        help=(
            "Path for saving the final pipeline."
        ),
    )

    args = parser.parse_args()

    main(
        file_paths=args.inputs,
        train_months=args.train_months,
        val_month=args.val_month,
        test_month=args.test_month,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        n_trials=args.n_trials,
        sample_train=args.sample_train,
        sample_val=args.sample_val,
        local_model_path=args.local_model_path,
    )