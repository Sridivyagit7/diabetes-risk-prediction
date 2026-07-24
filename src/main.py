"""
main.py
--------
Runs the full end-to-end pipeline in order:

  1. (optional) generate_sample_data.py  -> data/diabetes_raw.csv
  2. load_to_sql.py                      -> data/diabetes.db (SQL cohort extraction)
  3. preprocessing.py                    -> data/diabetes_processed.csv
  4. feature_engineering.py              -> data/diabetes_features.csv
  5. evaluate.py (which also trains)     -> models/best_model.joblib, reports/*

Usage:
    python main.py                # uses existing data/diabetes_raw.csv if present
    python main.py --use-sample    # (re)generates synthetic sample data first
"""

import argparse
import os

from config import RAW_DATA_PATH


def main():
    parser = argparse.ArgumentParser(description="Diabetes risk prediction pipeline")
    parser.add_argument(
        "--use-sample", action="store_true",
        help="Generate a synthetic sample dataset before running the pipeline",
    )
    args = parser.parse_args()

    if args.use_sample or not os.path.exists(RAW_DATA_PATH):
        print("=== STEP 0: Generating sample data ===")
        from generate_sample_data import generate_sample_data
        generate_sample_data().to_csv(RAW_DATA_PATH, index=False)

    print("\n=== STEP 1: Loading data to SQL & extracting cohort ===")
    from load_to_sql import load_csv_to_sqlite
    load_csv_to_sqlite()

    print("\n=== STEP 2: Preprocessing ===")
    from preprocessing import run_preprocessing
    run_preprocessing()

    print("\n=== STEP 3: Feature engineering ===")
    from feature_engineering import run_feature_engineering
    run_feature_engineering()

    print("\n=== STEP 4: Training models & evaluating ===")
    from evaluate import run_evaluation
    run_evaluation()

    print("\nPipeline complete. See reports/ and models/ for outputs.")


if __name__ == "__main__":
    main()
