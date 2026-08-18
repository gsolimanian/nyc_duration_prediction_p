import mlflow
import optuna
import lightgbm as lgb
import joblib
import pandas as pd
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder
from scipy.sparse import hstack
from sklearn.metrics import root_mean_squared_error
from mlflow.models import infer_signature
import pandas as pd
from typing import List, Tuple
import pandas as pd
from typing import List, Tuple

def load_and_combine_months(file_paths: List[str]) -> pd.DataFrame:
    """
    Load multiple monthly parquet files and combine them into one DataFrame.
    Adds a 'month' column for time-based splitting.
    """
    dfs = []
    
    for path in file_paths:
        print(f"Loading: {path}")
        df = pd.read_parquet(path)
        
        # استخراج ماه از نام فایل (فرض: نام فایل شامل تاریخ است)
        # مثال: yellow_tripdata_2026-01.parquet → month = 1
        month = int(path.split('-')[-1].split('.')[0].split('_')[0])

        df['month'] = month
        
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
    Split data into Train and Validation based on month.
    """
    train_df = df[df['month'].isin(train_months)].copy()
    val_df = df[df['month'] == val_month].copy()
    
    print(f"Train shape: {train_df.shape} | Months: {train_months}")
    print(f"Validation shape: {val_df.shape} | Month: {val_month}")
    
    return train_df, val_df


def select_features(df: pd.DataFrame):
    """Select features and define categorical vs numerical."""
    categorical = ['hour_category']           # فقط این را وکتورایز می‌کنیم (تعداد دسته کم است)
    
    numerical = [
        'trip_distance',
        'PULocationID',
        'DOLocationID',
        'pickup_dayofweek',
        'pickup_hour',
        'is_rush_hour',
        'is_weekend',
        'PU_DO'
    ]
    target = 'duration'

    
    df = df[categorical + numerical + [target]].copy()
    return df, categorical, numerical, target

def load_and_combine_months(file_paths: List[str]) -> pd.DataFrame:
    """
    Load multiple monthly parquet files and combine them into one DataFrame.
    Adds a 'month' column for time-based splitting.
    """
    dfs = []
    
    for path in file_paths:
        print(f"Loading: {path}")
        df = pd.read_parquet(path)
        
        # استخراج ماه از نام فایل (فرض: نام فایل شامل تاریخ است)
        # مثال: yellow_tripdata_2026-01.parquet → month = 1
        month = int(path.split('-')[-1].split('.')[0].split('_')[0])

        df['month'] = month
        
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
    Split data into Train and Validation based on month.
    """
    train_df = df[df['month'].isin(train_months)].copy()
    val_df = df[df['month'] == val_month].copy()
    
    print(f"Train shape: {train_df.shape} | Months: {train_months}")
    print(f"Validation shape: {val_df.shape} | Month: {val_month}")
    
    return train_df, val_df


# ========================
# ۱. ترانسفورمرهای سفارشی
# ========================

class CreatePU_DO(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['PU_DO'] = X['PULocationID'].astype(str) + '_' + X['DOLocationID'].astype(str)
        return X


class TargetEncodePU_DO(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=15, min_samples_leaf=30):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encoder = None

    def fit(self, X, y):
        self.encoder = TargetEncoder(
            cols=['PU_DO'],
            smoothing=self.smoothing,
            min_samples_leaf=self.min_samples_leaf
        )
        self.encoder.fit(X[['PU_DO']], y)
        return self

    def transform(self, X):
        X = X.copy()
        X['PU_DO_encoded'] = self.encoder.transform(X[['PU_DO']])
        return X


class FeatureCombiner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dv = DictVectorizer()

    def fit(self, X, y=None):
        cat_dicts = X[['hour_category']].to_dict(orient='records')
        self.dv.fit(cat_dicts)
        return self

    def transform(self, X):
        cat_dicts = X[['hour_category']].to_dict(orient='records')
        X_cat = self.dv.transform(cat_dicts)

        numerical = ['trip_distance', 'pickup_dayofweek', 'PU_DO_encoded']
        X_num = X[numerical].values

        return hstack([X_cat, X_num]).tocsr()


# ========================
# ۲. تابع ساخت Pipeline
# ========================
def create_taxi_pipeline(params: dict):
    pipeline = Pipeline([
        ('create_pu_do',     CreatePU_DO()),
        ('target_encoding',  TargetEncodePU_DO()),
        ('feature_combiner', FeatureCombiner()),
        ('model',            lgb.LGBMRegressor(**params))
    ])
    return pipeline


# ========================
# ۳. تابع Objective برای Optuna
# ========================
def objective(trial, X_train, X_val, y_train, y_val):
    with mlflow.start_run(nested=True, run_name=f"lgb_trial_{trial.number}"):

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': 2,
            'verbose': -1
        }

        pipeline = create_taxi_pipeline(params)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)

        return rmse


# ========================
# ۴. تابع اصلی Tuning
# ========================
def run_lightgbm_optuna_tuning(X_train, X_val, y_train, y_val, n_trials=50):
    with mlflow.start_run(run_name="lightgbm_optuna_tuning"):
        
        study = optuna.create_study(
            direction="minimize",
            study_name="LightGBM_Hyperparameter_Tuning",
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(
            lambda trial: objective(trial, X_train, X_val, y_train, y_val),
            n_trials=n_trials,
            show_progress_bar=True
        )

        print("\n" + "="*60)
        print(f"بهترین RMSE: {study.best_value:.4f}")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        print("="*60)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_rmse", study.best_value)

        return study.best_params


# ========================
# ۵. تابع آموزش مدل نهایی + لاگ در MLflow
# ========================
def train_final_model(X_train, y_train, best_params, run_name="final_lightgbm_model"):
    
    with mlflow.start_run(run_name=run_name):
        
        params = best_params.copy()
        params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': 42,
            'n_jobs': 2,
            'verbose': -1
        })

        pipeline = create_taxi_pipeline(params)
        pipeline.fit(X_train, y_train)

        signature = infer_signature(X_train, y_train)

        # ========================
        # لاگ کردن Pipeline با cloudpickle (حل مشکل سریالایز)
        # ========================
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            serialization_format="cloudpickle",           # ← مهم
            registered_model_name="NYC_Taxi_Duration_LightGBM",
            metadata={
                "description": "Final LightGBM model after Optuna tuning",
                "training_period": "AUG 2026",
                "feature_engineering": "Target Encoding on PU_DO + One-Hot on hour_category",
                "number_of_trials": 50,
                "best_rmse": round(best_params.get("best_rmse", 0), 4)
            }
        )

        print("✅ Pipeline نهایی در MLflow لاگ شد.")
        return pipeline


# ========================
# ۶. اجرای اصلی
# ========================
if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print("Tracking URI:", mlflow.get_tracking_uri())
    mlflow.set_experiment('nyc_duration_prediction_boosting_model_pipeline_v2')
    # ========================
    # لود و تقسیم داده (طبق کد خودت)
    # ========================
    file_paths = [
        "G:/nyc_duration_prediction/nyc_duration_prediction/data/processed/yellow_tripdata_2026-01_processed.parquet",
        "G:/nyc_duration_prediction/nyc_duration_prediction/data/processed/yellow_tripdata_2026-02_processed.parquet",
        "G:/nyc_duration_prediction/nyc_duration_prediction/data/processed/yellow_tripdata_2026-03_processed.parquet",
    ]

    df = load_and_combine_months(file_paths)
    train, val = split_train_validation_by_month(df, train_months=[1, 2], val_month=3)

    train_df = train.sample(n=5000, random_state=42).reset_index(drop=True)
    val_df = val.sample(n=500, random_state=42).reset_index(drop=True)

    # ========================
    # انتخاب فیچرها
    # ========================
    train_df, categorical, numerical, target = select_features(train_df)
    val_df, _, _, _ = select_features(val_df)

    # ========================
    # آماده‌سازی داده برای Pipeline
    # ========================
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]

    # ========================
    # اجرای کامل
    # ========================
    best_params = run_lightgbm_optuna_tuning(X_train, X_val, y_train, y_val, n_trials=50)

    final_pipeline = train_final_model(X_train, y_train, best_params)