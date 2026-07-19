"""
CareFlow AI - Module 4: Hospital Resource Optimization (Linear Programming)
--------------------------------------------------------------------------------
Given predicted daily patient demand per department (derived from admission
data / LOS predictions) and department-specific staffing ratios, this module
uses Linear Programming (PuLP) to allocate beds and nurses across departments
at minimum cost, while respecting hospital-wide capacity and budget limits.

Unmet demand is allowed but heavily penalized in the objective, so the
optimizer will tell you WHERE a hospital is understaffed rather than simply
failing when perfect coverage isn't affordable - this mirrors real hospital
operations, where trade-offs are the norm, not the exception.

Run: python src/optimization/resource_optimizer.py
Input:  data/raw/hospital_admissions.csv
Output: notebooks/figures/resource_allocation.png, printed allocation plan
"""

import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src import config

DATA_PATH = config.DATA_RAW_PATH

# ---------------------------------------------------------------
# Hospital-wide operating policy - these are genuinely config, not data
# that lives in a table (they represent budget/capacity DECISIONS, not
# department master data), so they stay in config.py.
# ---------------------------------------------------------------
NURSE_COST_PER_SHIFT = config.NURSE_COST_PER_SHIFT
SHORTAGE_PENALTY = config.SHORTAGE_PENALTY
TOTAL_BEDS_AVAILABLE = config.TOTAL_BEDS_AVAILABLE
TOTAL_NURSES_AVAILABLE = config.TOTAL_NURSES_AVAILABLE
DAILY_BUDGET = config.DAILY_BUDGET


def get_department_params(engine=None):
    """
    Returns (nurse_ratio, bed_cost, min_service_level) dicts.

    If a DB engine is provided, reads these LIVE from the departments
    table - the actual source of truth once the pipeline is running
    end-to-end, since that's the same table an ops team would update
    directly (e.g. via an admin tool) without touching code.

    config.DEPARTMENT_META is used only as a fallback for standalone
    script runs with no DB available (e.g. `python
    src/optimization/resource_optimizer.py` on its own), and as the
    seed data db_utils.py uses to populate the table the FIRST time.
    After that first seed, the table - not this dict - is authoritative.
    """
    if engine is not None:
        import pandas as pd
        dept_df = pd.read_sql(
            "SELECT department_name, nurse_ratio, bed_daily_cost, min_service_level FROM departments",
            engine
        )
        nurse_ratio = dict(zip(dept_df["department_name"], dept_df["nurse_ratio"]))
        bed_cost = dict(zip(dept_df["department_name"], dept_df["bed_daily_cost"]))
        min_service_level = dict(zip(dept_df["department_name"], dept_df["min_service_level"]))
        return nurse_ratio, bed_cost, min_service_level

    # Fallback: no DB engine given (standalone run)
    nurse_ratio = {d: meta["nurse_ratio"] for d, meta in config.DEPARTMENT_META.items()}
    bed_cost = {d: meta["bed_daily_cost"] for d, meta in config.DEPARTMENT_META.items()}
    min_service_level = {d: meta["min_service_level"] for d, meta in config.DEPARTMENT_META.items()}
    return nurse_ratio, bed_cost, min_service_level


def compute_daily_department_demand(df: pd.DataFrame) -> pd.Series:
    """
    Estimates average daily patient census per department by expanding
    each admission into its individual occupied days, then averaging
    across the full date range in the dataset.

    In a full production pipeline, `length_of_stay` here would come from
    Module 1's XGBoost predictions for *upcoming* admissions rather than
    historical actuals - this function would just take a dataframe of
    predicted admissions instead. The optimization logic itself doesn't
    change either way.
    """
    df = df.copy()
    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])

    patient_days = []
    for _, row in df.iterrows():
        n_days = max(1, (row["discharge_date"] - row["admission_date"]).days)
        patient_days.append(n_days)

    total_span_days = (df["admission_date"].max() - df["admission_date"].min()).days + 1

    # total patient-days per department / total days in dataset = avg daily census
    df["patient_days"] = patient_days
    dept_patient_days = df.groupby("department")["patient_days"].sum()
    avg_daily_census = (dept_patient_days / total_span_days).round().astype(int)

    return avg_daily_census


def build_and_solve_lp(demand: pd.Series, nurse_ratio: dict = None, bed_cost: dict = None,
                        min_service_level: dict = None, engine=None):
    """
    nurse_ratio, bed_cost, min_service_level can be passed explicitly, or
    left as None to auto-fetch via get_department_params(engine) - pass
    `engine` when calling from the full pipeline (db_utils.py) so this
    reads live department data from the database rather than config.py.
    """
    if nurse_ratio is None or bed_cost is None or min_service_level is None:
        nurse_ratio, bed_cost, min_service_level = get_department_params(engine)

    departments = list(demand.index)

    prob = pulp.LpProblem("Hospital_Resource_Allocation", pulp.LpMinimize)

    beds = pulp.LpVariable.dicts("beds", departments, lowBound=0, cat="Integer")
    nurses = pulp.LpVariable.dicts("nurses", departments, lowBound=0, cat="Integer")
    shortage = pulp.LpVariable.dicts("shortage", departments, lowBound=0, cat="Continuous")

    # Objective: minimize operating cost + shortage penalty
    prob += (
        pulp.lpSum(bed_cost[d] * beds[d] for d in departments)
        + pulp.lpSum(NURSE_COST_PER_SHIFT * nurses[d] for d in departments)
        + pulp.lpSum(SHORTAGE_PENALTY * shortage[d] for d in departments)
    ), "Total_Cost_Plus_Shortage_Penalty"

    # Demand constraint: beds allocated + shortage must cover predicted demand
    for d in departments:
        prob += beds[d] + shortage[d] >= demand[d], f"Demand_{d}"

    # Staffing ratio constraint: enough nurses for the beds allocated
    for d in departments:
        prob += nurses[d] >= nurse_ratio[d] * beds[d], f"Staffing_Ratio_{d}"

    # Minimum service level: cap how much of a department's demand can go
    # unmet, so the optimizer can't "solve" high cost by abandoning
    # critical-care coverage entirely.
    for d in departments:
        max_allowed_shortage = demand[d] * (1 - min_service_level[d])
        prob += shortage[d] <= max_allowed_shortage, f"Max_Shortage_{d}"

    # Hospital-wide capacity constraints
    prob += pulp.lpSum(beds[d] for d in departments) <= TOTAL_BEDS_AVAILABLE, "Total_Bed_Capacity"
    prob += pulp.lpSum(nurses[d] for d in departments) <= TOTAL_NURSES_AVAILABLE, "Total_Nurse_Capacity"

    # Budget constraint
    prob += (
        pulp.lpSum(bed_cost[d] * beds[d] for d in departments)
        + pulp.lpSum(NURSE_COST_PER_SHIFT * nurses[d] for d in departments)
        <= DAILY_BUDGET
    ), "Daily_Budget"

    # PuLP bundles an x86_64 CBC binary that fails with "Bad CPU type in
    # executable" on Apple Silicon (M1/M2/M3) Macs. We instead look for a
    # native arm64 CBC installed via Homebrew (`brew install cbc`), and
    # fall back to PuLP's default bundled solver on other platforms
    # (Windows/Intel Mac/Linux), where it works fine.
    import shutil
    import platform

    homebrew_cbc = shutil.which("cbc")
    if platform.system() == "Darwin" and homebrew_cbc:
        solver = pulp.COIN_CMD(path=homebrew_cbc, msg=False)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)

    prob.solve(solver)

    return prob, beds, nurses, shortage, departments


def summarize_solution(prob, beds, nurses, shortage, departments, demand, bed_cost):
    print(f"\nSolver status: {pulp.LpStatus[prob.status]}")

    results = []
    for d in departments:
        results.append({
            "department": d,
            "demand": demand[d],
            "beds_allocated": int(beds[d].value()),
            "nurses_allocated": int(nurses[d].value()),
            "shortage": round(shortage[d].value(), 1),
        })

    results_df = pd.DataFrame(results)
    total_cost = pulp.value(prob.objective)

    print("\n--- Optimal Resource Allocation Plan ---")
    print(results_df.to_string(index=False))
    print(f"\nTotal objective value (cost + shortage penalty): ${total_cost:,.0f}")

    actual_op_cost = sum(
        bed_cost[row["department"]] * row["beds_allocated"] + NURSE_COST_PER_SHIFT * row["nurses_allocated"]
        for _, row in results_df.iterrows()
    )
    print(f"Actual daily operating cost (beds + nurses only): ${actual_op_cost:,.0f} "
          f"(budget cap: ${DAILY_BUDGET:,.0f})")

    total_shortage = results_df["shortage"].sum()
    if total_shortage > 0:
        print(f"\n⚠ Unmet demand: {total_shortage:.0f} beds/day across departments "
              f"- hospital cannot fully staff predicted demand within current budget/capacity.")
    else:
        print("\n✓ All predicted demand met within budget and capacity limits.")

    return results_df


def plot_allocation(results_df):
    os.makedirs("notebooks/figures", exist_ok=True)
    x = np.arange(len(results_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, results_df["demand"], width, label="Predicted Demand")
    ax.bar(x + width/2, results_df["beds_allocated"], width, label="Beds Allocated")

    for i, row in results_df.iterrows():
        if row["shortage"] > 0:
            ax.annotate(f"-{row['shortage']:.0f}", (x[i] + width/2, row["beds_allocated"]),
                        textcoords="offset points", xytext=(0, 5), ha="center", color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(results_df["department"], rotation=15)
    ax.set_ylabel("Beds")
    ax.set_title("Predicted Demand vs. Optimized Bed Allocation")
    ax.legend()
    plt.tight_layout()
    plt.savefig("notebooks/figures/resource_allocation.png")
    plt.close()
    print("\nSaved -> notebooks/figures/resource_allocation.png")


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records")

    demand = compute_daily_department_demand(df)
    print("\nEstimated average daily patient demand by department:")
    print(demand.to_string())

    # Standalone run: no DB engine, so this falls back to config.py's
    # seed values. When called from db_utils.py instead, an engine is
    # passed and department params come live from the departments table.
    nurse_ratio, bed_cost, min_service_level = get_department_params(engine=None)
    print("\n(Running standalone - department params sourced from config.py, "
          "not the database. Run via db_utils.py for DB-sourced values.)")

    prob, beds, nurses, shortage, departments = build_and_solve_lp(
        demand, nurse_ratio, bed_cost, min_service_level
    )
    results_df = summarize_solution(prob, beds, nurses, shortage, departments, demand, bed_cost)
    plot_allocation(results_df)


if __name__ == "__main__":
    main()
