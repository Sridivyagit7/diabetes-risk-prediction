"""
evaluate.py
------------
Generates evaluation artifacts for the report/README and for the
Power BI dashboard:
  - ROC curves for all four models (reports/roc_curves.png)
  - Feature importance for the best tree-based model (reports/feature_importance.png)
  - Confusion matrix for the best model (reports/confusion_matrix.png)
  - A patient-level risk export CSV with risk tiers, for Power BI
    (reports/powerbi_patient_risk_export.csv)
"""

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from config import (
    ROC_CURVE_PATH, FEATURE_IMPORTANCE_PATH, CONFUSION_MATRIX_PATH,
    BI_EXPORT_PATH, BEST_MODEL_PATH,
)
from train_models import run_training


def plot_roc_curves(fitted_models, X_test, y_test):
    plt.figure(figsize=(7, 6))
    for name, model in fitted_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Diabetes Risk Classifiers")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close()
    print(f"Saved ROC curves to {ROC_CURVE_PATH}")


def plot_feature_importance(best_model, feature_names):
    if not hasattr(best_model, "feature_importances_"):
        if hasattr(best_model, "coef_"):
            importances = pd.Series(best_model.coef_[0], index=feature_names).abs()
        else:
            print("Model has no feature_importances_ or coef_; skipping plot.")
            return
    else:
        importances = pd.Series(best_model.feature_importances_, index=feature_names)

    top_features = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    top_features.sort_values().plot(kind="barh")
    plt.title("Top 15 Predictive Drivers")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    plt.close()
    print(f"Saved feature importance to {FEATURE_IMPORTANCE_PATH}")
    return top_features


def plot_confusion_matrix(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Best Model")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")


def export_for_powerbi(best_model, X_test, y_test):
    export_df = X_test.copy()
    export_df["actual_diabetes"] = y_test.values
    export_df["predicted_probability"] = best_model.predict_proba(X_test)[:, 1]
    export_df["predicted_label"] = best_model.predict(X_test)

    def risk_tier(p):
        if p < 0.20:
            return "Low"
        elif p < 0.50:
            return "Medium"
        else:
            return "High"

    export_df["risk_tier"] = export_df["predicted_probability"].apply(risk_tier)
    export_df.to_csv(BI_EXPORT_PATH, index=False)
    print(f"Saved Power BI export ({export_df.shape}) to {BI_EXPORT_PATH}")
    print(export_df["risk_tier"].value_counts())


def run_evaluation():
    results_df, fitted_models, splits = run_training()
    X_train, X_test, y_train, y_test = splits

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    print(f"\nGenerating evaluation artifacts using best model: {best_name}")

    plot_roc_curves(fitted_models, X_test, y_test)
    plot_feature_importance(best_model, X_test.columns)
    plot_confusion_matrix(best_model, X_test, y_test)
    export_for_powerbi(best_model, X_test, y_test)


if __name__ == "__main__":
    run_evaluation()
