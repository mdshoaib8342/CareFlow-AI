"""
CareFlow AI - Shared Preprocessing
--------------------------------------
Common data loading logic used by every module. Currently minimal
(this dataset is synthetic and pre-cleaned), but centralizing it means
if you later add real data validation - missing value handling, date
parsing, outlier checks - it happens once, consistently, for every
model instead of being reimplemented per script.
"""

import pandas as pd


def load_admissions_data(path: str) -> pd.DataFrame:
    """Loads the admissions CSV with correct dtypes for date columns."""
    df = pd.read_csv(path)

    required_cols = [
        "admission_id", "patient_id", "age", "gender", "admission_type",
        "diagnosis_code", "department", "length_of_stay", "comorbidity_count",
        "num_prior_admissions", "insurance_type", "treatment_sequence",
        "discharge_pathway", "readmitted_30_days", "admission_date", "discharge_date"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Loaded data is missing expected columns: {missing}")

    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])

    return df
