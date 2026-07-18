"""
CareFlow AI - Centralized Configuration
------------------------------------------
Single source of truth for paths, model versioning, and department
metadata that were previously duplicated across los_prediction.py,
readmission_risk.py, resource_optimizer.py, and db_utils.py.
"""

import os

# ---------------------------------------------------------------
# Paths (relative to project root - all scripts should be run
# from the project root, e.g. `python src/models/los_prediction.py`)
# ---------------------------------------------------------------
DATA_RAW_PATH = "data/raw/hospital_admissions.csv"
MODELS_SAVED_DIR = "models_saved"
FIGURES_DIR = "notebooks/figures"
DB_PATH = "careflow.db"
SCHEMA_PATH = "src/database/schema.sql"

LOS_MODEL_PATH = os.path.join(MODELS_SAVED_DIR, "los_xgboost_model.json")
READMISSION_MODEL_PATH = os.path.join(MODELS_SAVED_DIR, "readmission_xgboost_model.json")
DISCHARGE_CNN_MODEL_PATH = os.path.join(MODELS_SAVED_DIR, "discharge_pathway_cnn.pt")

# ---------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------
RANDOM_STATE = 42
MODEL_VERSION = "v1.0"

# ---------------------------------------------------------------
# Department metadata (used by Module 4's optimizer AND db_utils
# when loading the departments dimension table - previously these
# numbers were typed out separately in both places)
# ---------------------------------------------------------------
DEPARTMENT_META = {
    "ICU":          {"nurse_ratio": 0.50, "bed_daily_cost": 2200, "min_service_level": 0.90},
    "Cardiology":   {"nurse_ratio": 0.34, "bed_daily_cost": 1400, "min_service_level": 0.75},
    "Surgery":      {"nurse_ratio": 0.34, "bed_daily_cost": 1600, "min_service_level": 0.75},
    "Pediatrics":   {"nurse_ratio": 0.25, "bed_daily_cost": 1100, "min_service_level": 0.70},
    "General Ward": {"nurse_ratio": 0.20, "bed_daily_cost": 800,  "min_service_level": 0.60},
}

NURSE_COST_PER_SHIFT = 450
SHORTAGE_PENALTY = 5000
TOTAL_BEDS_AVAILABLE = 48
TOTAL_NURSES_AVAILABLE = 20
DAILY_BUDGET = 85_000
