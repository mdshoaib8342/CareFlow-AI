"""
CareFlow AI - Shared Feature Engineering
--------------------------------------------
This was previously copy-pasted (with the two feature sets diverging
slightly) inside los_prediction.py, readmission_risk.py, and db_utils.py.
Centralizing it here means a change to how a feature is encoded only
needs to happen once, and training/inference are guaranteed to use
identical logic.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Feature sets per model. Centralizing these definitions (not just the
# encoding function) means the "what features does each model use" question
# has one authoritative answer instead of being implicitly defined by
# whatever each script happened to select.
LOS_NUMERIC_FEATURES = ["age", "comorbidity_count", "num_prior_admissions"]
LOS_CATEGORICAL_FEATURES = ["gender", "admission_type", "diagnosis_code", "department", "insurance_type"]

READMISSION_NUMERIC_FEATURES = ["age", "comorbidity_count", "num_prior_admissions", "length_of_stay"]
READMISSION_CATEGORICAL_FEATURES = ["gender", "admission_type", "diagnosis_code", "department", "insurance_type"]


def encode_features(df: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    """
    One-hot encodes categorical columns and concatenates with numeric
    columns. Returns (X, fitted_encoder).

    Note: the encoder is fit fresh on whatever df is passed in. This is
    fine for this project since training and inference both run against
    the full historical dataset (so the same categories are always
    present), but in a true production system you'd persist the fitted
    encoder from training (e.g. via joblib) and reuse it unchanged at
    inference time, rather than refitting - otherwise a new category
    seen only at inference time could silently shift column ordering.
    """
    X_numeric = df[numeric_cols].reset_index(drop=True)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_encoded = encoder.fit_transform(df[categorical_cols])
    cat_feature_names = encoder.get_feature_names_out(categorical_cols)

    X = pd.concat([X_numeric, pd.DataFrame(X_cat_encoded, columns=cat_feature_names)], axis=1)
    return X, encoder


def get_los_features(df: pd.DataFrame):
    """Feature matrix + target for LOS prediction (Module 1)."""
    X, encoder = encode_features(df, LOS_NUMERIC_FEATURES, LOS_CATEGORICAL_FEATURES)
    y = df["length_of_stay"]
    return X, y, encoder


def get_readmission_features(df: pd.DataFrame):
    """Feature matrix + target for readmission risk prediction (Module 2)."""
    X, encoder = encode_features(df, READMISSION_NUMERIC_FEATURES, READMISSION_CATEGORICAL_FEATURES)
    y = df["readmitted_30_days"]
    return X, y, encoder
