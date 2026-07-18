# 🏥 CareFlow AI — Hospital Operations Optimization System

CareFlow AI is an end-to-end system that combines **machine learning, deep learning, and prescriptive optimization** to improve hospital patient flow: predicting how long patients will stay, who's at risk of readmission, where they'll be discharged to, and how to allocate limited hospital beds and nursing staff accordingly.

This isn't four disconnected models — it's a pipeline where **predictions feed directly into an optimization layer** that makes an actual operational decision (bed/staff allocation), backed by a relational database and an interactive dashboard.

## Architecture

```
Raw hospital admissions data
        │
        ▼
┌────────────────────────────┐
│  Length of Stay Prediction  │   XGBoost regression
├────────────────────────────┤
│  Readmission Risk           │   XGBoost classification (imbalance-aware)
├────────────────────────────┤
│  Discharge Pathway          │   1D CNN (PyTorch) on treatment sequences
└────────────────────────────┘
        │  predictions feed into →
        ▼
┌────────────────────────────┐
│  Resource Optimization      │   Linear Programming (PuLP)
│  (beds + nurses allocation) │   cost-minimizing, priority-constrained
└────────────────────────────┘
        │
        ▼
   PostgreSQL database (ground truth + predictions, versioned)
        │
        ▼
   Streamlit + Plotly interactive dashboard
```

## Key Results

| Module | Metric | Result |
|---|---|---|
| Length of Stay Prediction | MAE | ~1.15 days |
| Length of Stay Prediction | R² | ~0.62 |
| Readmission Risk | ROC-AUC | ~0.65 |
| Readmission Risk | Recall (readmitted class) | ~0.56 |
| Discharge Pathway (1D CNN) | Accuracy | ~0.39 (vs. 0.20 random baseline, 5 classes) |
| Resource Optimization | Solves LP with priority constraints | Ensures ICU coverage ≥90% even under budget pressure |

Full evaluation details, confusion matrices, and ROC curves are in `notebooks/figures/`.

## Why These Numbers, Not Higher Ones

This project uses **synthetic data with intentional, realistic noise** — not hand-tuned to produce impressive-looking metrics. A few honest notes worth knowing before you dig into the code:

- The discharge pathway CNN initially collapsed to predicting only the majority class due to class imbalance and a feature/target mismatch in the synthetic data generator (sequence data didn't encode the severity signal driving the target). This was diagnosed and fixed — see the comments in `data/generate_synthetic_data.py` and `src/models/discharge_pathway_cnn.py`.
- The resource optimizer initially, under pure cost-minimization, allocated almost no beds to the ICU (the most expensive department) — a real and important failure mode in naive cost-minimizing LPs. Fixed with explicit minimum service-level constraints per department (see `src/optimization/resource_optimizer.py`).

Both of these are documented as genuine debugging stories, not smoothed over — see the "Design Decisions & Lessons Learned" section below.

## Tech Stack

- **ML:** XGBoost, scikit-learn
- **Deep Learning:** PyTorch (1D CNN)
- **Optimization:** PuLP (Linear Programming)
- **Data:** Pandas, NumPy, Faker (synthetic data generation)
- **Database:** PostgreSQL, SQLAlchemy
- **Dashboard:** Streamlit + Plotly
- **Testing:** pytest

## Project Structure

```
CareFlow-AI/
├── data/
│   ├── raw/hospital_admissions.csv       # generated synthetic data
│   └── generate_synthetic_data.py
├── src/
│   ├── config.py                          # centralized paths/constants
│   ├── data_pipeline/
│   │   ├── preprocess.py                  # data loading + validation
│   │   └── feature_engineering.py         # shared feature encoding
│   ├── models/
│   │   ├── los_prediction.py              # Module 1: XGBoost regression
│   │   ├── readmission_risk.py            # Module 2: XGBoost classification
│   │   └── discharge_pathway_cnn.py       # Module 3: PyTorch 1D CNN
│   ├── optimization/
│   │   └── resource_optimizer.py          # Module 4: PuLP linear programming
│   └── database/
│       ├── schema.sql                     # normalized schema + views
│       └── db_utils.py                    # DB creation + population
├── dashboard/
│   └── app.py                             # Streamlit + Plotly dashboard
├── models_saved/                          # trained model artifacts (gitignored)
├── notebooks/figures/                     # evaluation plots
├── tests/
│   └── test_feature_engineering.py
├── requirements.txt
└── README.md
```

## Setup & Running the Pipeline

**Database:** this project uses PostgreSQL (not SQLite). Install and start it first:

```bash
brew install postgresql@16
brew services start postgresql@16
createdb careflow
```

Set your database password as an environment variable (or use the default below for local dev):
```bash
export CAREFLOW_DB_PASSWORD=postgres
```

Trained models are committed to this repo, so once the database is populated you can jump straight to the dashboard without retraining anything:

```bash
git clone https://github.com/<your-username>/CareFlow-AI.git
cd CareFlow-AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build and populate the database (loads data + runs real inference from trained models)
python src/database/db_utils.py

# Launch the dashboard
streamlit run dashboard/app.py
```

To regenerate everything from scratch (new synthetic data, retrained models), run each stage in order:

```bash
# 1. Generate synthetic hospital data
python data/generate_synthetic_data.py

# 2. Train the three prediction models
python src/models/los_prediction.py
python src/models/readmission_risk.py
python src/models/discharge_pathway_cnn.py

# 3. Run the resource allocation optimizer
python src/optimization/resource_optimizer.py

# 4. Build the database (loads data + runs real inference from trained models)
python src/database/db_utils.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

Run tests with:
```bash
pytest tests/ -v
```

## macOS / Apple Silicon Setup Notes

If you're on an M-series Mac, three native (non-pip) dependencies are needed:

```bash
brew install postgresql@16   # the database itself
brew install libomp           # required for xgboost
brew install cbc              # required for pulp's linear programming solver
```

## Design Decisions & Lessons Learned

- **No data leakage:** LOS predictions only use information available at admission time (never discharge outcomes). Readmission risk uses information known by discharge time, but deliberately excludes `discharge_pathway` to avoid circular reasoning between the two models.
- **Predictions are stored separately from ground truth** in the database (`los_predictions`, `readmission_predictions`, `discharge_pathway_predictions` tables), each tagged with a `model_version`. This means ground truth is never overwritten, and multiple model versions could coexist for comparison over time.
- **Class imbalance is handled explicitly**, not ignored: `scale_pos_weight` for the readmission XGBoost model, class-weighted `CrossEntropyLoss` for the CNN.
- **The resource optimizer allows unmet demand (soft constraints) rather than failing outright**, mirroring how real hospitals operate under real capacity limits — but with explicit minimum service-level floors so cost minimization can't silently abandon high-acuity departments like the ICU.
- **Feature engineering is centralized** in `src/data_pipeline/feature_engineering.py` rather than duplicated across model scripts, so training and inference (in `db_utils.py`) can't drift out of sync.

## Future Improvements

- Persist fitted encoders (e.g. via `joblib`) instead of refitting at inference time, for true production-safe inference on unseen categories
- Add MLflow or similar for experiment tracking across model versions
- Add confidence intervals / prediction intervals to LOS forecasts
- Deploy the optimizer as a scheduled job (e.g. daily) rather than a manual script run

## License

MIT
