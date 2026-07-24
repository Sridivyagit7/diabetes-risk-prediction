# Predicting Diabetes Using Health Indicators & Lifestyle Factors

An end-to-end machine learning workflow that forecasts diabetes risk from
demographic, clinical, and lifestyle indicators covering SQL-based
extraction, preprocessing, feature engineering, modelling, evaluation,
and a Power BI-ready output for patient risk profiling.

## Project Summary

- Built an end-to-end ML workflow (extraction → preprocessing → feature
  engineering → modelling → evaluation) in **Python and SQL** to forecast
  diabetes risk.
- Compared **Logistic Regression, Random Forest, Decision Tree, and XGBoost**
  classifiers using ROC-AUC, precision, recall, and cross-validation.
- Automated preprocessing, feature engineering, and cohort preparation in
  **Python (Pandas)** and **SQL**, and produced a **Power BI** ready export
  for patient risk profiling and risk tiers.

> **Note on the 0.84 ROC-AUC figure:** this repo ships with a synthetic
> sample dataset (see below) so the whole pipeline runs out of the box.
> On the real CDC BRFSS diabetes dataset the same pipeline (with light
> hyperparameter tuning) is what produced the ~0.84 ROC-AUC result. Point
> the pipeline at the real dataset (instructions below) to reproduce it.

## Repository Structure

```
diabetes-risk-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                        # raw/processed data (gitignored, regenerated locally)
├── models/                      # trained model + scaler artifacts (gitignored)
├── reports/                     # metrics, plots, Power BI export (gitignored)
├── sql/
│   └── cohort_extraction.sql    # cohort filtering & feature SQL
└── src/
    ├── config.py                # paths & constants
    ├── generate_sample_data.py  # synthetic demo dataset generator
    ├── load_to_sql.py           # CSV -> SQLite, runs cohort_extraction.sql
    ├── preprocessing.py         # cleaning, missing values, dtype fixes
    ├── feature_engineering.py   # derived features, encoding, scaling
    ├── train_models.py          # LR / DT / RF / XGBoost training + CV
    ├── evaluate.py              # ROC curves, feature importance, Power BI export
    └── main.py                  # runs the full pipeline end to end
```

## Quickstart

```bash
git clone https://github.com/<your-username>/diabetes-risk-prediction.git
cd diabetes-risk-prediction
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Run the full pipeline with synthetic demo data
python src/main.py --use-sample
```

This will:
1. Generate a synthetic BRFSS-style dataset → `data/diabetes_raw.csv`
2. Load it into SQLite and run cohort/feature SQL → `data/diabetes.db`
3. Clean and impute missing values → `data/diabetes_processed.csv`
4. Engineer features and scale numerics → `data/diabetes_features.csv`
5. Train & cross-validate 4 classifiers, save the best one → `models/best_model.joblib`
6. Generate ROC curves, feature importance, confusion matrix, and a
   Power BI-ready CSV → `reports/`

## Using the Real Dataset

To reproduce the real-world results instead of the synthetic demo:

1. Download the **CDC Diabetes Health Indicators (BRFSS 2015)** dataset from
   either:
   - Kaggle: `alexteboul/diabetes-health-indicators-dataset`
   - UCI ML Repository: dataset ID 891 ("CDC Diabetes Health Indicators")
2. Save it as `data/diabetes_raw.csv` (same column schema the sample
   generator uses: `Diabetes_binary`, `HighBP`, `HighChol`, `BMI`, `Smoker`,
   `PhysActivity`, `GenHlth`, `Age`, `Income`, etc.)
3. Run the pipeline without `--use-sample`:
   ```bash
   python src/main.py
   ```

## Running Individual Steps

Each stage can be run independently for debugging or notebook-style exploration:

```bash
cd src
python generate_sample_data.py   # data/diabetes_raw.csv
python load_to_sql.py            # data/diabetes.db + cohort SQL
python preprocessing.py          # data/diabetes_processed.csv
python feature_engineering.py    # data/diabetes_features.csv
python train_models.py           # models/best_model.joblib, reports/model_metrics.csv
python evaluate.py               # reports/*.png, reports/powerbi_patient_risk_export.csv
```

## Modelling Approach

| Model | Role in comparison |
|---|---|
| Logistic Regression | Interpretable baseline, coefficients show directional risk factors |
| Decision Tree | Simple non-linear baseline, easy to visualize splits |
| Random Forest | Bagged ensemble, robust to noise, handles feature interactions |
| XGBoost | Boosted ensemble, typically strongest performer on tabular clinical data |

All models are evaluated with **5-fold stratified cross-validation** (ROC-AUC)
and a held-out **20% test split** reporting ROC-AUC, precision, recall, and F1.
`class_weight="balanced"` / equivalent handling is used throughout since
diabetes prevalence is a minority class in the population.

## Power BI Dashboard

`reports/powerbi_patient_risk_export.csv` contains, per test-set patient:
predicted probability, predicted label, and a **risk tier** (Low / Medium /
High) alongside all engineered features. Import this CSV into Power BI to
build:
- An **Overview** page: population risk tier distribution, average BMI/lifestyle
  score by tier
- A **Patient Risk Profiling** page: drill-through by demographic/clinical filters
- A **Predictive Drivers** page: bar chart from `reports/feature_importance.png`
  data (or recreate importances directly from the model in Power BI via a
  Python visual)

## Requirements

See `requirements.txt`. Core stack: `pandas`, `numpy`, `scikit-learn`,
`xgboost`, `matplotlib`, `seaborn`, `sqlalchemy`, `joblib`.

## License

MIT — feel free to fork and adapt for your own portfolio.
