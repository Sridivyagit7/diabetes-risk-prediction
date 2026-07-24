"""
load_to_sql.py
---------------
Loads the raw CSV into a local SQLite database and executes
sql/cohort_extraction.sql to build the cohort/feature tables using SQL
(as opposed to pandas). This is the "SQL" half of the extraction step
described in the project summary. Swap SQLite for Postgres/MySQL/etc.
in a real deployment by changing only the SQLAlchemy connection string.
"""

import os
import sqlite3

import pandas as pd

from config import RAW_DATA_PATH, SQLITE_DB_PATH, ROOT_DIR

SQL_SCRIPT_PATH = os.path.join(ROOT_DIR, "sql", "cohort_extraction.sql")


def load_csv_to_sqlite():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"{RAW_DATA_PATH} not found. Run generate_sample_data.py first, "
            "or place the real dataset there."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    conn = sqlite3.connect(SQLITE_DB_PATH)
    df.to_sql("raw_patients", conn, if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into raw_patients table ({SQLITE_DB_PATH})")

    with open(SQL_SCRIPT_PATH, "r") as f:
        sql_script = f.read()
    conn.executescript(sql_script)
    conn.commit()
    print("Executed cohort_extraction.sql -> cohort_patients, cohort_features, risk_tier_summary")

    summary = pd.read_sql("SELECT * FROM risk_tier_summary", conn)
    print("\nRisk tier summary (top rows):")
    print(summary.head(10))

    conn.close()
    return SQLITE_DB_PATH


if __name__ == "__main__":
    load_csv_to_sqlite()
