import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

def save_processed_data(
    df: pd.DataFrame, 
    output_path: str, 
    verbose: bool = True
) -> Optional[str]:
    """
    Save the processed DataFrame to a Parquet file.

    This function is designed to be used inside a data pipeline.

    Parameters:
    -----------
    df : pd.DataFrame
        The processed DataFrame to be saved.
    output_path : str
        Full path where the file should be saved (including filename).
    verbose : bool, default True
        Whether to print information about the saving process.

    Returns:
    --------
    str or None
        Returns the output path if saving is successful, otherwise None.
    """
    if df is None or df.empty:
        if verbose:
            print("Warning: DataFrame is empty or None. Nothing to save.")
        return None

    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save as Parquet
        df.to_parquet(output_path, index=False)

        if verbose:
            print(f"✓ Processed data saved successfully.")
            print(f"  Path: {output_path}")
            print(f"  Shape: {df.shape}")
            print(f"  Size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

        return output_path

    except Exception as e:
        print(f"Error saving processed data: {e}")
        return None


def vectorize_features(df: pd.DataFrame, categorical: list, numerical: list):
    """
    Convert categorical and numerical features into a format suitable for modeling.
    Uses DictVectorizer to handle categorical variables.
    """
    print("Vectorizing features...", flush=True)

    # انتخاب فقط فیچرهای مورد نیاز
    df = df[categorical + numerical].copy()

    # تبدیل دیتافریم به لیست دیکشنری (فرمت مورد نیاز DictVectorizer)
    records = df.to_dict(orient='records')

    # وکتورایز کردن
    dv = DictVectorizer()

    # تبدیل به ماتریس عددی
    X = dv.fit_transform(records)

    print(f"Vectorization completed. Shape: {X.shape}", flush=True)
    
    return X, dv



def select_features(df: pd.DataFrame):
    """
    Select features and separate categorical vs numerical.
    For tree-based models, we avoid heavy one-hot encoding on high cardinality features.
    """
    categorical = ['hour_category']           # فقط این را وکتورایز می‌کنیم (تعداد دسته کم است)
    
    numerical = [
        'trip_distance',
        'PULocationID',
        'DOLocationID',
        'pickup_dayofweek',
        'pickup_hour',
        'is_rush_hour',
        'is_weekend'
    ]

    target = 'duration'

    df_selected = df[categorical + numerical + [target]].copy()

    print(f"Categorical features: {categorical}")
    print(f"Numerical features:   {numerical}")
    
    return df_selected, categorical, numerical, target


def categorize_hour(hour):
    if 0 <= hour <= 5:
        return 'late_night'
    elif 6 <= hour <= 9:
        return 'morning_peak'
    elif 10 <= hour <= 15:
        return 'midday'
    elif 16 <= hour <= 19:
        return 'evening_peak'
    else:
        return 'night'


def load_data(file_path: str):
    try:
        print("Loading data...", flush=True)
        data = pd.read_parquet(file_path)
        print(f"Data loaded successfully. Shape: {data.shape}", flush=True)
        return data
    except Exception as e:
        print(f"Error loading data: {e}", flush=True)
        return None


def processing_data(data: pd.DataFrame) -> pd.DataFrame:
    print("Starting data processing...", flush=True)

    # Remove rows with missing values
    clean_data = data.dropna().copy()

    # Create duration column (in minutes)
    clean_data['duration'] = (
        clean_data['tpep_dropoff_datetime'] - clean_data['tpep_pickup_datetime']
    ).dt.total_seconds() / 60
    clean_data['duration'] = clean_data['duration'].round(2)

    # Convert pickup time to datetime
    clean_data['tpep_pickup_datetime'] = pd.to_datetime(clean_data['tpep_pickup_datetime'])

    # Create time-based features
    clean_data['pickup_hour'] = clean_data['tpep_pickup_datetime'].dt.hour
    clean_data['pickup_dayofweek'] = clean_data['tpep_pickup_datetime'].dt.dayofweek

    # Create additional features
    clean_data['is_weekend'] = clean_data['pickup_dayofweek'].isin([5, 6])   # 5=Friday, 6=Saturday
    clean_data['hour_category'] = clean_data['pickup_hour'].apply(categorize_hour)
    clean_data['is_rush_hour'] = clean_data['pickup_hour'].isin([5, 6, 7, 8, 9])
    clean_data['PU_DO'] = (clean_data['PULocationID'].astype(str) + '_' + clean_data['DOLocationID'].astype(str))


    # Data Cleaning
    clean_data = clean_data.dropna()    
    # Remove invalid trip durations (negative, zero, or longer than 3 hours)
    clean_data = clean_data[(clean_data['duration'] > 0) & (clean_data['duration'] <= 180)]

    # Remove trips with zero or negative distance
    clean_data = clean_data[clean_data['trip_distance'] > 0]

    # Remove invalid passenger counts (must be between 1 and 6)
    clean_data = clean_data[(clean_data['passenger_count'] > 0) & (clean_data['passenger_count'] <= 6)]

    print(f"Data after processing. Shape: {clean_data.shape}", flush=True)
    return clean_data

def main(data_path: str , output_path:str):
    print("=== Starting Data Pipeline ===")

    
    # 1. Load raw data
    data = load_data(data_path)

    if data is not None:
        # 2. Process data
        processed_data = processing_data(data)

        # 3. Save processed data
        saved_path = save_processed_data(processed_data, output_path)

        if saved_path:
            print("\n=== Pipeline completed successfully ===")
        else:
            print("Failed to save processed data.")
    else:
        print("Data loading failed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
         "--input", 
        type=str, 
        required=True, 
        help="Path to the raw parquet file (e.g., data/raw/yellow_tripdata_2026-01.parquet)"
    )


    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the raw parquet file (e.g., data/raw/yellow_tripdata_2026-01.parquet)"
    )
    args = parser.parse_args()


    main(data_path = args.input , output_path = args.output)