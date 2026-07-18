"""
CareFlow AI - Operations Dashboard (Streamlit + Plotly)
--------------------------------------------------------------
Reads directly from careflow.db (populated by src/database/db_utils.py)
and visualizes: KPIs, department breakdown, LOS prediction accuracy,
readmission risk distribution, discharge pathway prediction accuracy,
and the current resource allocation plan.

Run: streamlit run dashboard/app.py
Requires: careflow.db to already exist in the project root
          (run `python src/database/db_utils.py` first)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import config

st.set_page_config(page_title="CareFlow AI Dashboard", layout="wide", page_icon="🏥")


@st.cache_resource
def get_engine():
    return create_engine(config.DB_CONNECTION_STRING)


@st.cache_data(ttl=300)
def load_data():
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        return None

    conn = engine

    kpis = pd.read_sql("""
        SELECT COUNT(*) as total_admissions,
               ROUND(AVG(length_of_stay), 2) as avg_los,
               ROUND(AVG(readmitted_30_days) * 100, 1) as readmit_rate
        FROM admissions
    """, conn)

    dept_summary = pd.read_sql("SELECT * FROM vw_department_summary", conn)

    los_compare = pd.read_sql("""
        SELECT a.admission_id, a.length_of_stay AS actual_los, l.predicted_los,
               d.department_name
        FROM admissions a
        JOIN los_predictions l ON a.admission_id = l.admission_id
        JOIN departments d ON a.department_id = d.department_id
    """, conn)

    readmit_risk = pd.read_sql("""
        SELECT r.predicted_probability, a.readmitted_30_days AS actual_readmit
        FROM readmission_predictions r
        JOIN admissions a ON r.admission_id = a.admission_id
    """, conn)

    pathway_compare = pd.read_sql("""
        SELECT a.discharge_pathway AS actual, dp.predicted_pathway AS predicted, dp.confidence
        FROM admissions a
        JOIN discharge_pathway_predictions dp ON a.admission_id = dp.admission_id
    """, conn)

    resource_plan = pd.read_sql("SELECT * FROM vw_latest_resource_plan", conn)

    return {
        "kpis": kpis, "dept_summary": dept_summary, "los_compare": los_compare,
        "readmit_risk": readmit_risk, "pathway_compare": pathway_compare,
        "resource_plan": resource_plan
    }


data = load_data()

st.title("🏥 CareFlow AI — Hospital Operations Dashboard")

if data is None:
    st.error(
        "Could not connect to the PostgreSQL database, or it hasn't been populated yet. "
        "Make sure PostgreSQL is running and the CAREFLOW_DB_* environment variables are "
        "set correctly, then run `python src/database/db_utils.py` from the project root."
    )
    st.stop()

# ---------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------
kpi = data["kpis"].iloc[0]
mae = (data["los_compare"]["actual_los"] - data["los_compare"]["predicted_los"]).abs().mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Admissions", f"{int(kpi['total_admissions']):,}")
c2.metric("Avg Length of Stay", f"{kpi['avg_los']} days")
c3.metric("30-Day Readmission Rate", f"{kpi['readmit_rate']}%")
c4.metric("LOS Model MAE", f"{mae:.2f} days")

st.divider()

# ---------------------------------------------------------------
# Department Breakdown
# ---------------------------------------------------------------
st.subheader("Department Overview")
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        data["dept_summary"].sort_values("total_admissions", ascending=True),
        x="total_admissions", y="department_name", orientation="h",
        title="Admissions by Department", color="department_name",
        labels={"total_admissions": "Admissions", "department_name": "Department"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')

with col2:
    fig = px.bar(
        data["dept_summary"].sort_values("avg_length_of_stay", ascending=True),
        x="avg_length_of_stay", y="department_name", orientation="h",
        title="Avg Length of Stay by Department", color="department_name",
        labels={"avg_length_of_stay": "Avg LOS (days)", "department_name": "Department"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')

st.divider()

# ---------------------------------------------------------------
# LOS Prediction Accuracy
# ---------------------------------------------------------------
st.subheader("Length of Stay: Predicted vs. Actual")
sample = data["los_compare"].sample(min(1000, len(data["los_compare"])), random_state=42)
fig = px.scatter(
    sample, x="actual_los", y="predicted_los", color="department_name",
    opacity=0.5, labels={"actual_los": "Actual LOS (days)", "predicted_los": "Predicted LOS (days)"}
)
max_val = max(sample["actual_los"].max(), sample["predicted_los"].max())
fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                          line=dict(dash="dash", color="gray"), name="Perfect Prediction"))
st.plotly_chart(fig, width='stretch')
st.caption("Points closer to the dashed line indicate more accurate predictions.")

st.divider()

# ---------------------------------------------------------------
# Readmission Risk Distribution
# ---------------------------------------------------------------
st.subheader("Readmission Risk Distribution")
fig = px.histogram(
    data["readmit_risk"], x="predicted_probability", color="actual_readmit",
    nbins=30, barmode="overlay", opacity=0.6,
    labels={"predicted_probability": "Predicted Readmission Probability",
            "actual_readmit": "Actually Readmitted"},
    color_discrete_map={0: "steelblue", 1: "crimson"}
)
st.plotly_chart(fig, width='stretch')
st.caption("A well-calibrated model shows red (actually readmitted) skewed toward higher probabilities.")

st.divider()

# ---------------------------------------------------------------
# Discharge Pathway: Actual vs. Predicted
# ---------------------------------------------------------------
st.subheader("Discharge Pathway: Actual vs. Predicted")
crosstab = pd.crosstab(data["pathway_compare"]["actual"], data["pathway_compare"]["predicted"])
fig = px.imshow(
    crosstab, text_auto=True, aspect="auto", color_continuous_scale="Blues",
    labels=dict(x="Predicted Pathway", y="Actual Pathway", color="Count")
)
st.plotly_chart(fig, width='stretch')
st.caption(
    "Diagonal = correct predictions. Off-diagonal cells show where the model confuses "
    "clinically similar outcomes (e.g. Rehab vs. Home Health Care)."
)

st.divider()

# ---------------------------------------------------------------
# Resource Allocation Plan
# ---------------------------------------------------------------
st.subheader("Current Resource Allocation Plan")
rp = data["resource_plan"]

fig = go.Figure()
fig.add_bar(x=rp["department_name"], y=rp["predicted_demand"], name="Predicted Demand")
fig.add_bar(x=rp["department_name"], y=rp["beds_allocated"], name="Beds Allocated")
fig.update_layout(barmode="group", title="Predicted Demand vs. Optimized Bed Allocation")
st.plotly_chart(fig, width='stretch')

shortage_depts = rp[rp["shortage"] > 0]
if len(shortage_depts) > 0:
    st.warning(
        f"⚠️ Unmet demand in: "
        + ", ".join(f"{r.department_name} ({r.shortage:.0f} beds/day, "
                     f"{r.pct_demand_met}% covered)" for r in shortage_depts.itertuples())
    )
else:
    st.success("✓ All predicted demand met within current budget and capacity.")

st.dataframe(rp, width='stretch')
