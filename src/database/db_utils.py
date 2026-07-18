"""
CareFlow AI - Database Utilities
--------------------------------------
Creates the SQLite database from schema.sql, loads the synthetic admissions
CSV into normalized tables (departments, patients, admissions), and runs
the actual trained models (Modules 1-3) to populate the prediction tables
with real inference output - not ground truth relabeled as predictions.

Run: python src/database/db_utils.py
Output: careflow.db (SQLite database file in project root)

Note: uses SQLite for local development simplicity (zero server setup,
works out of the box with Power BI's built-in ODBC/SQLite connectors).
For a production deployment, swap the SQLAlchemy connection string for
PostgreSQL/MySQL - the schema and code above it don't need to change.
"""

import sqlite3
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src import config

DB_PATH = config.DB_PATH
SCHEMA_PATH = config.SCHEMA_PATH
DATA_PATH = config.DATA_RAW_PATH
MODEL_VERSION = config.MODEL_VERSION
DEPARTMENT_META = config.DEPARTMENT_META


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Creates all tables/views from schema.sql (idempotent - safe to re-run)."""
    with open(SCHEMA_PATH, "r") as f:
        schema_sql = f.read()

    conn = get_connection()
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()
    print(f"Database initialized -> {DB_PATH}")


def load_departments():
    conn = get_connection()
    cur = conn.cursor()
    for name, meta in DEPARTMENT_META.items():
        cur.execute("""
            INSERT OR IGNORE INTO departments (department_name, nurse_ratio, bed_daily_cost, min_service_level)
            VALUES (?, ?, ?, ?)
        """, (name, meta["nurse_ratio"], meta["bed_daily_cost"], meta["min_service_level"]))
    conn.commit()
    conn.close()
    print("Loaded departments dimension table")


def load_patients_and_admissions(df: pd.DataFrame):
    conn = get_connection()

    # patients: dedupe on patient_id (in this synthetic dataset each row is
    # a unique patient, but real data may have repeat visits per patient)
    patients_df = df[["patient_id", "gender", "insurance_type"]].drop_duplicates(subset="patient_id")
    patients_df.to_sql("patients", conn, if_exists="append", index=False)
    print(f"Loaded {len(patients_df)} patients")

    # Map department name -> department_id for the FK
    dept_map = pd.read_sql("SELECT department_id, department_name FROM departments", conn)
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
    admissions_df[cols].to_sql("admissions", conn, if_exists="append", index=False)
    print(f"Loaded {len(admissions_df)} admissions")

    conn.close()


def populate_los_predictions(df: pd.DataFrame):
    """Loads the trained XGBoost LOS model and runs REAL inference on the
    dataset, then stores predictions - distinct from the ground-truth
    length_of_stay already sitting in the admissions table."""
    import xgboost as xgb
    from src.data_pipeline.feature_engineering import get_los_features

    X, _, _ = get_los_features(df)

    model = xgb.XGBRegressor()
    model.load_model(config.LOS_MODEL_PATH)
    preds = model.predict(X)

    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    rows = [(aid, float(p), MODEL_VERSION, now) for aid, p in zip(df["admission_id"], preds)]
    cur.executemany("""
        INSERT INTO los_predictions (admission_id, predicted_los, model_version, predicted_at)
        VALUES (?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()
    print(f"Populated {len(rows)} LOS predictions")


def populate_readmission_predictions(df: pd.DataFrame):
    import xgboost as xgb
    from src.data_pipeline.feature_engineering import get_readmission_features

    X, _, _ = get_readmission_features(df)

    model = xgb.XGBClassifier()
    model.load_model(config.READMISSION_MODEL_PATH)
    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= 0.5).astype(int)

    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    rows = [(aid, float(p), int(l), MODEL_VERSION, now)
            for aid, p, l in zip(df["admission_id"], probs, labels)]
    cur.executemany("""
        INSERT INTO readmission_predictions (admission_id, predicted_probability, predicted_label, model_version, predicted_at)
        VALUES (?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()
    print(f"Populated {len(rows)} readmission predictions")


def populate_discharge_pathway_predictions(df: pd.DataFrame):
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

    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    rows = [(aid, pathway, float(conf), MODEL_VERSION, now)
            for aid, pathway, conf in zip(df["admission_id"], predicted_pathways, confidences.numpy())]
    cur.executemany("""
        INSERT INTO discharge_pathway_predictions (admission_id, predicted_pathway, confidence, model_version, predicted_at)
        VALUES (?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()
    print(f"Populated {len(rows)} discharge pathway predictions")


def populate_resource_allocation(plan_date: str = None):
    """Re-runs Module 4's LP optimizer and stores the resulting plan."""
    from src.optimization.resource_optimizer import (
        compute_daily_department_demand, build_and_solve_lp
    )

    df = pd.read_csv(DATA_PATH)
    demand = compute_daily_department_demand(df)
    prob, beds, nurses, shortage, departments = build_and_solve_lp(demand)

    plan_date = plan_date or datetime.now().date().isoformat()

    conn = get_connection()
    dept_map = pd.read_sql("SELECT department_id, department_name FROM departments", conn)
    dept_lookup = dict(zip(dept_map["department_name"], dept_map["department_id"]))

    cur = conn.cursor()
    now = datetime.now().isoformat()
    rows = []
    for d in departments:
        rows.append((
            dept_lookup[d], plan_date, int(demand[d]),
            int(beds[d].value()), int(nurses[d].value()),
            round(shortage[d].value(), 1), now
        ))
    cur.executemany("""
        INSERT INTO resource_allocation_plans
        (department_id, plan_date, predicted_demand, beds_allocated, nurses_allocated, shortage, generated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()
    print(f"Populated resource allocation plan for {plan_date}")


def print_summary():
    conn = get_connection()
    print("\n--- Database Summary ---")
    for table in ["departments", "patients", "admissions", "los_predictions",
                  "readmission_predictions", "discharge_pathway_predictions",
                  "resource_allocation_plans"]:
        count = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", conn)["n"][0]
        print(f"{table:35s}: {count} rows")

    print("\n--- Sample query: vw_department_summary ---")
    print(pd.read_sql("SELECT * FROM vw_department_summary", conn).to_string(index=False))
    conn.close()


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)  # fresh start for reproducible demo runs

    init_db()
    load_departments()

    df = pd.read_csv(DATA_PATH)
    load_patients_and_admissions(df)

    populate_los_predictions(df)
    populate_readmission_predictions(df)
    populate_discharge_pathway_predictions(df)
    populate_resource_allocation()

    print_summary()


if __name__ == "__main__":
    main()
