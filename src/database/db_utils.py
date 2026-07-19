"""
CareFlow AI - Database Utilities
--------------------------------------
Creates the PostgreSQL database schema, loads the synthetic admissions
CSV into normalized tables (departments, patients, admissions), and runs
the actual trained models (Modules 1-3) to populate the prediction tables
with real inference output - not ground truth relabeled as predictions.

Run: python src/database/db_utils.py
Requires: a running PostgreSQL instance (see README's PostgreSQL setup
section) and the CAREFLOW_DB_* environment variables set if you're not
using the local defaults in src/config.py.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src import config

SCHEMA_PATH = config.SCHEMA_PATH
DATA_PATH = config.DATA_RAW_PATH
MODEL_VERSION = config.MODEL_VERSION
DEPARTMENT_META = config.DEPARTMENT_META


def get_engine():
    return create_engine(config.DB_CONNECTION_STRING)


def reset_database(engine):
    """Drops and recreates the public schema for a clean, reproducible
    demo run - the PostgreSQL equivalent of deleting the SQLite .db file."""
    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.commit()


def init_db(engine):
    """Creates all tables/views from schema.sql."""
    with open(SCHEMA_PATH, "r") as f:
        schema_sql = f.read()

    with engine.connect() as conn:
        conn.execute(text(schema_sql))
        conn.commit()
    print(f"Database schema initialized -> {config.DB_NAME} @ {config.DB_HOST}:{config.DB_PORT}")


def load_departments(engine):
    with engine.connect() as conn:
        for name, meta in DEPARTMENT_META.items():
            conn.execute(text("""
                INSERT INTO departments (department_name, nurse_ratio, bed_daily_cost, min_service_level)
                VALUES (:name, :ratio, :cost, :service_level)
                ON CONFLICT (department_name) DO NOTHING
            """), {"name": name, "ratio": meta["nurse_ratio"],
                   "cost": meta["bed_daily_cost"], "service_level": meta["min_service_level"]})
        conn.commit()
    print("Loaded departments dimension table")


def load_patients_and_admissions(engine, df: pd.DataFrame):
    # Ensure dates are proper datetime objects, not strings, before to_sql -
    # avoids relying on implicit text->date casting during insert.
    df = df.copy()
    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])

    # patients: dedupe on patient_id (in this synthetic dataset each row is
    # a unique patient, but real data may have repeat visits per patient)
    patients_df = df[["patient_id", "gender", "insurance_type"]].drop_duplicates(subset="patient_id")
    patients_df.to_sql("patients", engine, if_exists="append", index=False)
    print(f"Loaded {len(patients_df)} patients")

    # Map department name -> department_id for the FK
    dept_map = pd.read_sql("SELECT department_id, department_name FROM departments", engine)
    dept_lookup = dict(zip(dept_map["department_name"], dept_map["department_id"]))

    admissions_df = df.copy()
    admissions_df["department_id"] = admissions_df["department"].map(dept_lookup)
    admissions_df = admissions_df.rename(columns={"age": "age_at_admission"})

    cols = [
        "admission_id", "patient_id", "department_id", "age_at_admission",
        "admission_type", "diagnosis_code", "admission_date", "discharge_date",
        "length_of_stay", "comorbidity_count", "num_prior_admissions",
        "treatment_sequence", "discharge_pathway", "readmitted_30_days"
    ]
    admissions_df[cols].to_sql("admissions", engine, if_exists="append", index=False)
    print(f"Loaded {len(admissions_df)} admissions")


def populate_los_predictions(engine, df: pd.DataFrame):
    """Loads the trained XGBoost LOS model and runs REAL inference on the
    dataset, then stores predictions - distinct from the ground-truth
    length_of_stay already sitting in the admissions table."""
    import xgboost as xgb
    from src.data_pipeline.feature_engineering import get_los_features

    X, _, _ = get_los_features(df)

    model = xgb.XGBRegressor()
    model.load_model(config.LOS_MODEL_PATH)
    preds = model.predict(X)

    now = datetime.now()
    rows = [{"admission_id": aid, "predicted_los": float(p),
             "model_version": MODEL_VERSION, "predicted_at": now}
            for aid, p in zip(df["admission_id"], preds)]

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO los_predictions (admission_id, predicted_los, model_version, predicted_at)
            VALUES (:admission_id, :predicted_los, :model_version, :predicted_at)
        """), rows)
        conn.commit()
    print(f"Populated {len(rows)} LOS predictions")


def populate_readmission_predictions(engine, df: pd.DataFrame):
    import xgboost as xgb
    from src.data_pipeline.feature_engineering import get_readmission_features

    X, _, _ = get_readmission_features(df)

    model = xgb.XGBClassifier()
    model.load_model(config.READMISSION_MODEL_PATH)
    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= 0.5).astype(int)

    now = datetime.now()
    rows = [{"admission_id": aid, "predicted_probability": float(p), "predicted_label": int(l),
             "model_version": MODEL_VERSION, "predicted_at": now}
            for aid, p, l in zip(df["admission_id"], probs, labels)]

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO readmission_predictions (admission_id, predicted_probability, predicted_label, model_version, predicted_at)
            VALUES (:admission_id, :predicted_probability, :predicted_label, :model_version, :predicted_at)
        """), rows)
        conn.commit()
    print(f"Populated {len(rows)} readmission predictions")


def populate_discharge_pathway_predictions(engine, df: pd.DataFrame):
    import torch
    import torch.nn.functional as F
    from src.models.discharge_pathway_cnn import DischargePathwayCNN, SequenceVocab

    checkpoint = torch.load(config.DISCHARGE_CNN_MODEL_PATH, map_location="cpu", weights_only=False)

    vocab = SequenceVocab()
    vocab.token_to_id = checkpoint["vocab"]
    label_classes = checkpoint["label_classes"]
    max_len = checkpoint["max_seq_len"]
    embed_dim = checkpoint["embed_dim"]

    model = DischargePathwayCNN(
        vocab_size=len(vocab.token_to_id),
        embed_dim=embed_dim,
        num_classes=len(label_classes),
        max_len=max_len
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    encoded = [vocab.encode(seq, max_len) for seq in df["treatment_sequence"]]
    X = torch.tensor(encoded, dtype=torch.long)

    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1)
        confidences, pred_idx = probs.max(dim=1)

    predicted_pathways = [label_classes[i] for i in pred_idx.numpy()]

    now = datetime.now()
    rows = [{"admission_id": aid, "predicted_pathway": pathway, "confidence": float(conf),
             "model_version": MODEL_VERSION, "predicted_at": now}
            for aid, pathway, conf in zip(df["admission_id"], predicted_pathways, confidences.numpy())]

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO discharge_pathway_predictions (admission_id, predicted_pathway, confidence, model_version, predicted_at)
            VALUES (:admission_id, :predicted_pathway, :confidence, :model_version, :predicted_at)
        """), rows)
        conn.commit()
    print(f"Populated {len(rows)} discharge pathway predictions")


def populate_resource_allocation(engine, plan_date: str = None):
    """Re-runs Module 4's LP optimizer and stores the resulting plan.

    Passes `engine` through so department parameters (nurse ratios, bed
    costs, service levels) are read live from the departments table -
    the real source of truth - rather than from config.py's seed values."""
    from src.optimization.resource_optimizer import (
        compute_daily_department_demand, build_and_solve_lp, get_department_params
    )

    df = pd.read_csv(DATA_PATH)
    demand = compute_daily_department_demand(df)

    nurse_ratio, bed_cost, min_service_level = get_department_params(engine)
    prob, beds, nurses, shortage, departments = build_and_solve_lp(
        demand, nurse_ratio, bed_cost, min_service_level
    )

    plan_date = plan_date or datetime.now().date()

    dept_map = pd.read_sql("SELECT department_id, department_name FROM departments", engine)
    dept_lookup = dict(zip(dept_map["department_name"], dept_map["department_id"]))

    now = datetime.now()
    rows = []
    for d in departments:
        rows.append({
            "department_id": int(dept_lookup[d]), "plan_date": plan_date,
            "predicted_demand": int(demand[d]), "beds_allocated": int(beds[d].value()),
            "nurses_allocated": int(nurses[d].value()), "shortage": round(shortage[d].value(), 1),
            "generated_at": now
        })

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO resource_allocation_plans
            (department_id, plan_date, predicted_demand, beds_allocated, nurses_allocated, shortage, generated_at)
            VALUES (:department_id, :plan_date, :predicted_demand, :beds_allocated, :nurses_allocated, :shortage, :generated_at)
        """), rows)
        conn.commit()
    print(f"Populated resource allocation plan for {plan_date}")


def print_summary(engine):
    print("\n--- Database Summary ---")
    for table in ["departments", "patients", "admissions", "los_predictions",
                  "readmission_predictions", "discharge_pathway_predictions",
                  "resource_allocation_plans"]:
        count = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", engine)["n"][0]
        print(f"{table:35s}: {count} rows")

    print("\n--- Sample query: vw_department_summary ---")
    print(pd.read_sql("SELECT * FROM vw_department_summary", engine).to_string(index=False))


def main():
    engine = get_engine()

    reset_database(engine)  # fresh start for reproducible demo runs
    init_db(engine)
    load_departments(engine)

    df = pd.read_csv(DATA_PATH)
    load_patients_and_admissions(engine, df)

    populate_los_predictions(engine, df)
    populate_readmission_predictions(engine, df)
    populate_discharge_pathway_predictions(engine, df)
    populate_resource_allocation(engine)

    print_summary(engine)


if __name__ == "__main__":
    main()
