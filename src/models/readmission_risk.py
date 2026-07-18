"""
CareFlow AI - Module 2: 30-Day Readmission Risk Prediction
---------------------------------------------------------------
Trains an XGBoost classifier to predict whether a patient will be
readmitted within 30 days of discharge, using only information
available at (or before) discharge time.

Run: python src/models/readmission_risk.py
Input:  data/raw/hospital_admissions.csv
Output: models_saved/readmission_xgboost_model.json
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data_pipeline.preprocess import load_admissions_data
from src.data_pipeline.feature_engineering import get_readmission_features
from src import config

RANDOM_STATE = config.RANDOM_STATE
DATA_PATH = config.DATA_RAW_PATH
MODEL_OUT_PATH = config.READMISSION_MODEL_PATH


def load_data(path: str) -> pd.DataFrame:
    return load_admissions_data(path)


def engineer_features(df: pd.DataFrame):
    """Delegates to the shared feature engineering module - see
    src/data_pipeline/feature_engineering.py."""
    return get_readmission_features(df)


def train_model(X_train, y_train):
    # scale_pos_weight compensates for class imbalance:
    # ratio of negative to positive class, so the model doesn't just
    # learn to predict "not readmitted" every time.
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"Class balance -> Not readmitted: {n_neg}, Readmitted: {n_pos} "
          f"(scale_pos_weight={scale_pos_weight:.2f})")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, preds, target_names=["Not Readmitted", "Readmitted"]))

    auc = roc_auc_score(y_test, probs)
    print(f"ROC-AUC: {auc:.3f}")

    cm = confusion_matrix(y_test, preds)
    print("\nConfusion Matrix:")
    print("                 Predicted No   Predicted Yes")
    print(f"Actual No        {cm[0][0]:<14} {cm[0][1]}")
    print(f"Actual Yes       {cm[1][0]:<14} {cm[1][1]}")

    return preds, probs, auc, cm


def plot_diagnostics(y_test, probs, cm):
    os.makedirs("notebooks/figures", exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Readmission Risk")
    plt.legend()
    plt.tight_layout()
    plt.savefig("notebooks/figures/readmission_roc_curve.png")
    plt.close()

    # Confusion matrix heatmap
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Not Readmitted", "Readmitted"])
    plt.yticks([0, 1], ["Not Readmitted", "Readmitted"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Readmission Risk")
    plt.tight_layout()
    plt.savefig("notebooks/figures/readmission_confusion_matrix.png")
    plt.close()

    print("\nSaved plots -> notebooks/figures/readmission_roc_curve.png, "
          "notebooks/figures/readmission_confusion_matrix.png")


def plot_feature_importance(model, X, top_n=15):
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    feat_imp[::-1].plot(kind="barh")
    plt.title("Top Feature Importances - Readmission Risk")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("notebooks/figures/readmission_feature_importance.png")
    plt.close()
    print("Saved -> notebooks/figures/readmission_feature_importance.png")
    return feat_imp


def main():
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} records")

    X, y, encoder = engineer_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Overall readmission rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = train_model(X_train, y_train)
    preds, probs, auc, cm = evaluate_model(model, X_test, y_test)
    plot_diagnostics(y_test, probs, cm)
    feat_imp = plot_feature_importance(model, X)
    print("\nTop 5 features driving readmission risk:\n", feat_imp.head(5))

    os.makedirs("models_saved", exist_ok=True)
    model.save_model(MODEL_OUT_PATH)
    print(f"\nModel saved -> {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
