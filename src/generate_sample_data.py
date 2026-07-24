"""
generate_sample_data.py
------------------------
Generates a synthetic dataset that mimics the structure of the public
"CDC Diabetes Health Indicators" (BRFSS 2015) dataset: demographic,
clinical, and lifestyle features used to predict diabetes risk.

WHY THIS EXISTS
----------------
This lets the whole pipeline (preprocessing -> features -> training ->
evaluation) run immediately, with zero setup, for testing/demo purposes.

FOR YOUR REAL PROJECT
----------------------
Replace data/diabetes_raw.csv with the real dataset before you write it
up on your resume/GitHub. Good public sources:
  - Kaggle: "Diabetes Health Indicators Dataset" (CDC BRFSS 2015)
    https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
  - UCI ML Repository: "CDC Diabetes Health Indicators"
    https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
Both use the same column schema this script simulates, so the rest of
the pipeline (preprocessing.py, feature_engineering.py, train_models.py)
will work unchanged on the real file.
"""

import numpy as np
import pandas as pd
from config import RAW_DATA_PATH, RANDOM_STATE


def generate_sample_data(n_rows: int = 20000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age_group = rng.integers(1, 14, n_rows)          # 13 BRFSS age buckets
    sex = rng.integers(0, 2, n_rows)                  # 0 female, 1 male
    bmi = np.clip(rng.normal(28, 6, n_rows), 15, 60).round(1)
    high_bp = rng.binomial(1, 0.35, n_rows)
    high_chol = rng.binomial(1, 0.38, n_rows)
    chol_check = rng.binomial(1, 0.95, n_rows)
    smoker = rng.binomial(1, 0.40, n_rows)
    stroke = rng.binomial(1, 0.05, n_rows)
    heart_disease = rng.binomial(1, 0.10, n_rows)
    phys_activity = rng.binomial(1, 0.70, n_rows)
    fruits = rng.binomial(1, 0.60, n_rows)
    veggies = rng.binomial(1, 0.75, n_rows)
    hvy_alcohol = rng.binomial(1, 0.06, n_rows)
    healthcare = rng.binomial(1, 0.94, n_rows)
    no_doc_cost = rng.binomial(1, 0.10, n_rows)
    gen_health = rng.integers(1, 6, n_rows)           # 1 excellent - 5 poor
    ment_health_days = rng.integers(0, 31, n_rows)
    phys_health_days = rng.integers(0, 31, n_rows)
    diff_walk = rng.binomial(1, 0.17, n_rows)
    education = rng.integers(1, 7, n_rows)
    income = rng.integers(1, 9, n_rows)

    # Build a latent risk score so the target is actually learnable
    # (mirrors real epidemiological relationships, not just noise)
    risk_score = (
        0.05 * (bmi - 25)
        + 0.9 * high_bp
        + 0.7 * high_chol
        + 0.5 * heart_disease
        + 0.4 * stroke
        + 0.35 * (gen_health - 1)
        + 0.03 * age_group
        + 0.3 * diff_walk
        - 0.4 * phys_activity
        - 0.2 * veggies
        - 0.2 * fruits
        + rng.normal(0, 1.0, n_rows)
    )
    prob = 1 / (1 + np.exp(-(risk_score - 3)))
    diabetes_binary = rng.binomial(1, prob)

    df = pd.DataFrame({
        "Diabetes_binary": diabetes_binary,
        "HighBP": high_bp,
        "HighChol": high_chol,
        "CholCheck": chol_check,
        "BMI": bmi,
        "Smoker": smoker,
        "Stroke": stroke,
        "HeartDiseaseorAttack": heart_disease,
        "PhysActivity": phys_activity,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": hvy_alcohol,
        "AnyHealthcare": healthcare,
        "NoDocbcCost": no_doc_cost,
        "GenHlth": gen_health,
        "MentHlth": ment_health_days,
        "PhysHlth": phys_health_days,
        "DiffWalk": diff_walk,
        "Sex": sex,
        "Age": age_group,
        "Education": education,
        "Income": income,
    })

    # sprinkle a few missing values to make preprocessing.py meaningful
    for col in ["BMI", "GenHlth", "Income", "Education"]:
        mask = rng.random(n_rows) < 0.02
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_sample_data()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Sample dataset written to {RAW_DATA_PATH}")
    print(df.shape)
    print(df["Diabetes_binary"].value_counts(normalize=True))
