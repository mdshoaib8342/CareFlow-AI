-- ============================================================================
-- CareFlow AI - Database Schema
-- ============================================================================
-- Design principles:
--   1. Raw operational data (admissions) is kept separate from ML model
--      outputs (prediction tables) - ground truth is never overwritten by
--      a prediction, and multiple model versions can coexist.
--   2. Dimension tables (departments) are normalized out so department
--      metadata (capacity, nurse ratios) lives in one place, not repeated
--      on every row.
--   3. Every prediction table records model_version + predicted_at, so
--      you can audit "what did the model think on this date" - essential
--      for tracking model drift over time in a real deployment.
--
-- Written in PostgreSQL syntax.
-- ============================================================================

-- Foreign keys are enforced by default in PostgreSQL (unlike SQLite, where
-- PRAGMA foreign_keys = ON was needed) - no equivalent statement required.

-- ----------------------------------------------------------------------------
-- Dimension table: departments
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS departments (
    department_id     SERIAL PRIMARY KEY,
    department_name    TEXT NOT NULL UNIQUE,
    nurse_ratio         REAL NOT NULL,        -- required nurses per bed
    bed_daily_cost      REAL NOT NULL,        -- operating cost per bed/day
    min_service_level   REAL NOT NULL         -- min fraction of demand that must be met
);

-- ----------------------------------------------------------------------------
-- patients: demographic info that's stable across visits
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS patients (
    patient_id     TEXT PRIMARY KEY,
    gender          TEXT NOT NULL,
    insurance_type  TEXT NOT NULL
);

-- ----------------------------------------------------------------------------
-- admissions: one row per hospital stay (the operational "fact" table)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS admissions (
    admission_id            TEXT PRIMARY KEY,
    patient_id                TEXT NOT NULL REFERENCES patients(patient_id),
    department_id             INTEGER NOT NULL REFERENCES departments(department_id),
    age_at_admission           INTEGER NOT NULL,
    admission_type              TEXT NOT NULL CHECK (admission_type IN ('Emergency', 'Elective', 'Urgent')),
    diagnosis_code               TEXT NOT NULL,
    admission_date                 DATE NOT NULL,
    discharge_date                  DATE NOT NULL,
    length_of_stay                    INTEGER NOT NULL,
    comorbidity_count                  INTEGER NOT NULL DEFAULT 0,
    num_prior_admissions                INTEGER NOT NULL DEFAULT 0,
    treatment_sequence                    TEXT,             -- space-separated event tokens
    discharge_pathway                       TEXT,             -- actual outcome (ground truth)
    readmitted_30_days                        INTEGER NOT NULL DEFAULT 0 CHECK (readmitted_30_days IN (0, 1)),
    CHECK (discharge_date >= admission_date)
);

CREATE INDEX IF NOT EXISTS idx_admissions_patient   ON admissions(patient_id);
CREATE INDEX IF NOT EXISTS idx_admissions_department ON admissions(department_id);
CREATE INDEX IF NOT EXISTS idx_admissions_dates       ON admissions(admission_date, discharge_date);

-- ----------------------------------------------------------------------------
-- Prediction tables: one per model, kept separate from ground truth
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS los_predictions (
    prediction_id     SERIAL PRIMARY KEY,
    admission_id        TEXT NOT NULL REFERENCES admissions(admission_id),
    predicted_los          REAL NOT NULL,
    model_version             TEXT NOT NULL,
    predicted_at                TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_los_pred_admission ON los_predictions(admission_id);

CREATE TABLE IF NOT EXISTS readmission_predictions (
    prediction_id     SERIAL PRIMARY KEY,
    admission_id        TEXT NOT NULL REFERENCES admissions(admission_id),
    predicted_probability   REAL NOT NULL,
    predicted_label            INTEGER NOT NULL CHECK (predicted_label IN (0, 1)),
    model_version                 TEXT NOT NULL,
    predicted_at                    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_readmit_pred_admission ON readmission_predictions(admission_id);

CREATE TABLE IF NOT EXISTS discharge_pathway_predictions (
    prediction_id     SERIAL PRIMARY KEY,
    admission_id        TEXT NOT NULL REFERENCES admissions(admission_id),
    predicted_pathway      TEXT NOT NULL,
    confidence                REAL NOT NULL,
    model_version                TEXT NOT NULL,
    predicted_at                    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pathway_pred_admission ON discharge_pathway_predictions(admission_id);

-- ----------------------------------------------------------------------------
-- resource_allocation_plans: output of Module 4's LP optimizer
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS resource_allocation_plans (
    plan_id            SERIAL PRIMARY KEY,
    department_id        INTEGER NOT NULL REFERENCES departments(department_id),
    plan_date               DATE NOT NULL,
    predicted_demand           INTEGER NOT NULL,
    beds_allocated                INTEGER NOT NULL,
    nurses_allocated                 INTEGER NOT NULL,
    shortage                            REAL NOT NULL DEFAULT 0,
    generated_at                           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_resource_plan_dept_date ON resource_allocation_plans(department_id, plan_date);

-- ----------------------------------------------------------------------------
-- Views: convenience layers for Power BI / reporting, so the dashboard
-- queries a clean flat view instead of joining 6 tables every time.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_admissions_full AS
SELECT
    a.admission_id,
    a.patient_id,
    p.gender,
    p.insurance_type,
    d.department_name,
    a.age_at_admission,
    a.admission_type,
    a.diagnosis_code,
    a.admission_date,
    a.discharge_date,
    a.length_of_stay,
    a.comorbidity_count,
    a.num_prior_admissions,
    a.discharge_pathway,
    a.readmitted_30_days
FROM admissions a
JOIN patients p ON a.patient_id = p.patient_id
JOIN departments d ON a.department_id = d.department_id;

CREATE OR REPLACE VIEW vw_department_summary AS
SELECT
    d.department_name,
    COUNT(a.admission_id)             AS total_admissions,
    ROUND(AVG(a.length_of_stay)::numeric, 2)    AS avg_length_of_stay,
    ROUND((AVG(a.readmitted_30_days) * 100)::numeric, 1) AS readmission_rate_pct,
    ROUND(AVG(a.comorbidity_count)::numeric, 2)          AS avg_comorbidity_count
FROM admissions a
JOIN departments d ON a.department_id = d.department_id
GROUP BY d.department_name;

CREATE OR REPLACE VIEW vw_latest_resource_plan AS
SELECT
    d.department_name,
    r.plan_date,
    r.predicted_demand,
    r.beds_allocated,
    r.nurses_allocated,
    r.shortage,
    ROUND((100.0 * (r.predicted_demand - r.shortage) / r.predicted_demand)::numeric, 1) AS pct_demand_met
FROM resource_allocation_plans r
JOIN departments d ON r.department_id = d.department_id
WHERE r.plan_date = (SELECT MAX(plan_date) FROM resource_allocation_plans);
