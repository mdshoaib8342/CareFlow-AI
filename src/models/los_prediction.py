"""
CareFlow AI - Module 1: Length of Stay (LOS) Prediction
----------------------------------------------------------
Trains an XGBoost regression model to predict a patient's length
of stay (in days) at the time of admission, using only information
that would realistically be available on Day 1 of admission.

Run: python src/models/los_prediction.py
Input:  data/raw/hospital_admissions.csv
Output: models_saved/los_xgboost_model.json
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data_pipeline.preprocess import load_admissions_data
from src.data_pipeline.feature_engineering import get_los_features
from src import config

RANDOM_STATE = config.RANDOM_STATE
DATA_PATH = config.DATA_RAW_PATH
MODEL_OUT_PATH = config.LOS_MODEL_PATH


def load_data(path: str) -> pd.DataFrame:
    return load_admissions_data(path)


def engineer_features(df: pd.DataFrame):
    """Delegates to the shared feature engineering module (src/data_pipeline/
    feature_engineering.py) so training and inference (db_utils.py) can
    never drift out of sync - see the design note there for why we still
    fit the encoder fresh each run rather than persisting it."""
    return get_los_features(df)


def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        objective="reg:squarederror"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n--- Model Evaluation ---")
    print(f"MAE  (avg days off):  {mae:.2f} days")
    print(f"RMSE (penalizes big misses): {rmse:.2f} days")
    print(f"R^2  (variance explained):   {r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_feature_importance(model, X, top_n=15):
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    feat_imp[::-1].plot(kind="barh")
    plt.title("Top Feature Importances - LOS Prediction")
    plt.xlabel("Importance")
    plt.tight_layout()
    os.makedirs("notebooks/figures", exist_ok=True)
    plt.savefig("notebooks/figures/los_feature_importance.png")
    print("\nSaved feature importance plot -> notebooks/figures/los_feature_importance.png")
    return feat_imp


def main():
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} records")

    X, y, encoder = engineer_features(df)
    print(f"Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    feat_imp = plot_feature_importance(model, X)
    print("\nTop 5 features driving LOS:\n", feat_imp.head(5))

    os.makedirs("models_saved", exist_ok=True)
    model.save_model(MODEL_OUT_PATH)
    print(f"\nModel saved -> {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
