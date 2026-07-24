-- cohort_extraction.sql
-- ----------------------
-- Runs against the SQLite database created by src/load_to_sql.py
-- (table: raw_patients). Demonstrates the SQL side of the workflow:
-- cohort filtering, derived clinical flags, and aggregate summaries
-- that feed both the Python feature-engineering step and the Power BI
-- dashboard export.

-- 1) Basic cohort: drop rows with missing BMI/GenHlth, keep only checked cholesterol
DROP TABLE IF EXISTS cohort_patients;
CREATE TABLE cohort_patients AS
SELECT *
FROM raw_patients
WHERE BMI IS NOT NULL
  AND GenHlth IS NOT NULL
  AND CholCheck = 1;

-- 2) Derived clinical/lifestyle risk flags used later as engineered features
DROP TABLE IF EXISTS cohort_features;
CREATE TABLE cohort_features AS
SELECT
    *,
    CASE
        WHEN BMI < 18.5 THEN 'Underweight'
        WHEN BMI < 25   THEN 'Normal'
        WHEN BMI < 30   THEN 'Overweight'
        ELSE 'Obese'
    END AS bmi_category,
    CASE
        WHEN HighBP = 1 AND HighChol = 1 THEN 1
        ELSE 0
    END AS metabolic_risk_flag,
    (PhysActivity + Fruits + Veggies) AS healthy_lifestyle_score,
    CASE
        WHEN Age >= 10 THEN 1   -- BRFSS age bucket 10+ ~= 60 years and older
        ELSE 0
    END AS senior_flag
FROM cohort_patients;

-- 3) Population-level summary (this is what powers a Power BI "Overview" page)
DROP TABLE IF EXISTS risk_tier_summary;
CREATE TABLE risk_tier_summary AS
SELECT
    bmi_category,
    senior_flag,
    COUNT(*) AS patient_count,
    ROUND(AVG(Diabetes_binary) * 100, 2) AS diabetes_rate_pct,
    ROUND(AVG(BMI), 1) AS avg_bmi,
    ROUND(AVG(healthy_lifestyle_score), 2) AS avg_lifestyle_score
FROM cohort_features
GROUP BY bmi_category, senior_flag
ORDER BY diabetes_rate_pct DESC;

-- Sanity check queries (run manually / for QA)
-- SELECT COUNT(*) FROM raw_patients;
-- SELECT COUNT(*) FROM cohort_patients;
-- SELECT * FROM risk_tier_summary;
