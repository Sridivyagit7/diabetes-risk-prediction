"""
feature_engineering.py
------------------------
Builds model-ready features from the cleaned data:
  - one-hot encodes categorical fields created in SQL (bmi_category)
  - creates interaction / composite risk features
  - scales numeric features
Outputs data/diabetes_features.csv
"""

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DATA_PATH, FEATURED_DATA_PATH, SCALER_PATH, TARGET_COL

NUMERIC_FEATURES = [
    "BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income",
    "healthy_lifestyle_score",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Composite comorbidity count: how many major risk conditions a patient has
    df["comorbidity_count"] = (
        df["HighBP"] + df["HighChol"] + df["Stroke"] + df["HeartDiseaseorAttack"]
    )

    # Poor overall health flag (self-reported health 4=fair, 5=poor)
    df["poor_gen_health"] = (df["GenHlth"] >= 4).astype(int)

    # Unhealthy days ratio (mental + physical unhealthy days out of last 30, scaled)
    df["unhealthy_days_total"] = df["MentHlth"] + df["PhysHlth"]

    # Access-to-care barrier flag
    df["care_barrier"] = ((df["NoDocbcCost"] == 1) | (df["AnyHealthcare"] == 0)).astype(int)

    # One-hot encode the SQL-derived bmi_category
    df = pd.get_dummies(df, columns=["bmi_category"], prefix="bmi")

    return df


def scale_numeric(df: pd.DataFrame, fit: bool = True, scaler: StandardScaler = None):
    cols_present = [c for c in NUMERIC_FEATURES if c in df.columns]
    if fit:
        scaler = StandardScaler()
        df[cols_present] = scaler.fit_transform(df[cols_present])
    else:
        df[cols_present] = scaler.transform(df[cols_present])
    return df, scaler


def run_feature_engineering() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df = add_engineered_features(df)
    df, scaler = scale_numeric(df, fit=True)

    joblib.dump(scaler, SCALER_PATH)
    df.to_csv(FEATURED_DATA_PATH, index=False)

    print(f"Feature-engineered dataset shape: {df.shape}")
    print(f"Saved to {FEATURED_DATA_PATH}")
    print(f"Saved fitted scaler to {SCALER_PATH}")
    return df


if __name__ == "__main__":
    run_feature_engineering()
