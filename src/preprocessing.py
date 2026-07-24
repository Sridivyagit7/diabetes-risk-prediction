"""
preprocessing.py
-----------------
Cleans the raw extracted cohort data:
  - reads the SQL-derived cohort table (data/diabetes.db -> cohort_features)
  - handles missing values
  - removes duplicates
  - fixes dtypes
Outputs data/diabetes_processed.csv, ready for feature_engineering.py
"""

import sqlite3

import pandas as pd

from config import SQLITE_DB_PATH, PROCESSED_DATA_PATH


NUMERIC_COLS = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income"]


def load_cohort_from_sql() -> pd.DataFrame:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql("SELECT * FROM cohort_features", conn)
    conn.close()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    # Impute numeric columns with median (robust to outliers/skew)
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed {n_missing} missing values in '{col}' with median={median_val}")

    # Ensure binary/categorical columns are integers
    binary_like = [
        "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
        "metabolic_risk_flag", "senior_flag",
    ]
    for col in binary_like:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Basic sanity-bound BMI to a plausible clinical range
    df = df[(df["BMI"] >= 10) & (df["BMI"] <= 70)]

    return df.reset_index(drop=True)


def run_preprocessing() -> pd.DataFrame:
    df = load_cohort_from_sql()
    print(f"Loaded {len(df)} rows from cohort_features table")
    df_clean = clean_data(df)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved cleaned data ({df_clean.shape}) to {PROCESSED_DATA_PATH}")
    return df_clean


if __name__ == "__main__":
    run_preprocessing()
