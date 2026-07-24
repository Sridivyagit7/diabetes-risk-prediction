"""
config.py
---------
Central place for file paths and constants used across the pipeline.
Keeping these in one file means you only change a path once if your
folder layout changes.
"""

import os

# Project root = one level up from this file (src/ -> project root)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

RAW_DATA_PATH = os.path.join(DATA_DIR, "diabetes_raw.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "diabetes_processed.csv")
FEATURED_DATA_PATH = os.path.join(DATA_DIR, "diabetes_features.csv")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "diabetes.db")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(REPORTS_DIR, "model_metrics.csv")
ROC_CURVE_PATH = os.path.join(REPORTS_DIR, "roc_curves.png")
FEATURE_IMPORTANCE_PATH = os.path.join(REPORTS_DIR, "feature_importance.png")
CONFUSION_MATRIX_PATH = os.path.join(REPORTS_DIR, "confusion_matrix.png")
BI_EXPORT_PATH = os.path.join(REPORTS_DIR, "powerbi_patient_risk_export.csv")

TARGET_COL = "Diabetes_binary"
RANDOM_STATE = 42

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    os.makedirs(_d, exist_ok=True)
