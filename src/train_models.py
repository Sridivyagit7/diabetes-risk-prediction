"""
train_models.py
-----------------
Trains and compares four classifiers for diabetes risk prediction:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost

For each model:
  - 5-fold stratified cross-validation on ROC-AUC
  - Fit on the training split, evaluate on a held-out test split
  - Record ROC-AUC, precision, recall, F1

The best model (by test ROC-AUC) is saved to models/best_model.joblib,
and a metrics table is saved to reports/model_metrics.csv for the README
and Power BI dashboard.
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from config import (
    FEATURED_DATA_PATH, TARGET_COL, RANDOM_STATE,
    BEST_MODEL_PATH, METRICS_PATH,
)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def run_training():
    df = pd.read_csv(FEATURED_DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Keep only numeric feature columns (drop any leftover ID-like columns if present)
    X = X.select_dtypes(include=["number", "bool"]).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    fitted_models = {}

    for name, model in get_models().items():
        print(f"\nTraining {name}...")

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        test_auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "model": name,
            "cv_auc_mean": round(cv_scores.mean(), 4),
            "cv_auc_std": round(cv_scores.std(), 4),
            "test_roc_auc": round(test_auc, 4),
            "test_precision": round(precision, 4),
            "test_recall": round(recall, 4),
            "test_f1": round(f1, 4),
        })
        fitted_models[name] = model

        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Test ROC-AUC: {test_auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    results_df = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False)
    results_df.to_csv(METRICS_PATH, index=False)
    print(f"\nMetrics saved to {METRICS_PATH}")
    print(results_df)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"\nBest model: {best_name} -> saved to {BEST_MODEL_PATH}")

    return results_df, fitted_models, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run_training()
