"""
CareFlow AI - Tests: Data Pipeline
--------------------------------------
Focused on feature_engineering.py and preprocess.py deliberately - these
are shared by every model, so a bug here silently corrupts all four
modules at once. This is the highest-value place in the whole project
to have test coverage, even with just a handful of tests.

Run: pytest tests/
"""

import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_pipeline.feature_engineering import (
    encode_features, get_los_features, get_readmission_features,
    LOS_NUMERIC_FEATURES, LOS_CATEGORICAL_FEATURES
)
from src.data_pipeline.preprocess import load_admissions_data


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "admission_id": ["A1", "A2", "A3"],
        "patient_id": ["P1", "P2", "P3"],
        "age": [30, 65, 80],
        "gender": ["Male", "Female", "Male"],
        "admission_type": ["Emergency", "Elective", "Urgent"],
        "diagnosis_code": ["Cardiac", "Renal", "Cardiac"],
        "department": ["ICU", "General Ward", "ICU"],
        "length_of_stay": [5, 3, 9],
        "comorbidity_count": [1, 0, 3],
        "num_prior_admissions": [0, 1, 2],
        "insurance_type": ["Private", "Government", "Private"],
        "treatment_sequence": ["admit discharge"] * 3,
        "discharge_pathway": ["Home", "Home", "Rehab"],
        "readmitted_30_days": [0, 0, 1],
        "admission_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "discharge_date": ["2026-01-06", "2026-01-05", "2026-01-12"],
    })


def test_encode_features_row_count_preserved(sample_df):
    """The encoded feature matrix must have exactly one row per input row -
    a silent row drop here would misalign features with labels."""
    X, encoder = encode_features(sample_df, LOS_NUMERIC_FEATURES, LOS_CATEGORICAL_FEATURES)
    assert len(X) == len(sample_df)


def test_encode_features_no_missing_values(sample_df):
    """One-hot encoding + numeric passthrough should never introduce NaNs
    on clean input data."""
    X, _ = encode_features(sample_df, LOS_NUMERIC_FEATURES, LOS_CATEGORICAL_FEATURES)
    assert X.isnull().sum().sum() == 0


def test_encode_features_unseen_category_does_not_crash(sample_df):
    """handle_unknown='ignore' must actually prevent crashes on categories
    not seen during fit - this matters at inference time when new data
    might contain a diagnosis code the encoder wasn't fit on."""
    X, encoder = encode_features(sample_df, LOS_NUMERIC_FEATURES, LOS_CATEGORICAL_FEATURES)
    new_row = sample_df.iloc[[0]].copy()
    new_row["diagnosis_code"] = "NeverSeenBefore"
    # Should not raise
    encoded = encoder.transform(new_row[LOS_CATEGORICAL_FEATURES])
    assert encoded.shape[0] == 1


def test_get_los_features_excludes_leakage_columns(sample_df):
    """Regression guard: LOS features must never include discharge_pathway
    or readmitted_30_days, since those are only known after the stay ends."""
    X, y, _ = get_los_features(sample_df)
    leakage_cols = ["discharge_pathway", "readmitted_30_days", "discharge_date"]
    assert not any(col in X.columns for col in leakage_cols)


def test_get_readmission_features_includes_los(sample_df):
    """Readmission features SHOULD include length_of_stay, since it's known
    by discharge time (unlike discharge_pathway, which we exclude)."""
    X, y, _ = get_readmission_features(sample_df)
    assert any(col.startswith("length_of_stay") or col == "length_of_stay" for col in X.columns) or \
           "length_of_stay" in X.columns


def test_load_admissions_data_raises_on_missing_columns(tmp_path):
    """A malformed CSV missing required columns should fail loudly and
    early, not silently produce a broken feature matrix three steps later."""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"admission_id": ["A1"], "age": [30]}).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="missing expected columns"):
        load_admissions_data(str(bad_csv))


def test_load_admissions_data_parses_dates(sample_df, tmp_path):
    csv_path = tmp_path / "sample.csv"
    sample_df.to_csv(csv_path, index=False)

    df = load_admissions_data(str(csv_path))
    assert pd.api.types.is_datetime64_any_dtype(df["admission_date"])
    assert pd.api.types.is_datetime64_any_dtype(df["discharge_date"])
