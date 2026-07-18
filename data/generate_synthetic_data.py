"""
CareFlow AI - Synthetic Hospital Data Generator
-------------------------------------------------
Generates realistic (fake) hospital admission records for building
and testing the LOS prediction, readmission risk, discharge pathway,
and resource optimization modules.

Run: python data/generate_synthetic_data.py
Output: data/raw/hospital_admissions.csv
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import timedelta
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

N_PATIENTS = 5000

DEPARTMENTS = ["ICU", "General Ward", "Surgery", "Pediatrics", "Cardiology"]
ADMISSION_TYPES = ["Emergency", "Elective", "Urgent"]
DIAGNOSIS_CODES = [
    "Cardiac", "Respiratory", "Orthopedic", "Neurological",
    "Gastrointestinal", "Renal", "Infectious Disease", "Oncology"
]
INSURANCE_TYPES = ["Private", "Government", "Self-Pay", "Uninsured"]
DISCHARGE_PATHWAYS = ["Home", "Rehab", "Nursing Facility", "Transfer", "Home Health Care"]

# Treatment event vocabulary used to build a per-patient sequence.
# This sequence is what the 1D CNN will learn from later.
EVENT_VOCAB = ["admit", "triage", "labs", "imaging", "surgery",
               "icu_stay", "ward_stay", "physio", "discharge_prep", "discharge"]


def generate_treatment_sequence(department, admission_type, age, comorbidity_count):
    """Builds a plausible ordered sequence of treatment events for a patient.

    Importantly, this now reflects age/comorbidity severity too (via
    geriatric_consult / case_management / social_work_referral events),
    since discharge_pathway is driven by those same factors. Without this,
    the sequence wouldn't actually carry the signal needed to predict
    discharge pathway - a classic case of features not matching the target."""
    seq = ["admit", "triage"]

    if admission_type == "Emergency":
        seq.append("labs")

    if department in ("Surgery", "Cardiology"):
        seq += ["imaging", "surgery", "icu_stay" if random.random() < 0.4 else "ward_stay"]
    elif department == "ICU":
        seq += ["imaging", "icu_stay"]
    else:
        seq += ["ward_stay"]

    if random.random() < 0.3:
        seq.append("physio")

    # High-severity patients (elderly + multiple comorbidities) get
    # additional care-coordination steps - this is what actually
    # correlates with non-Home discharge pathways in real hospitals.
    high_severity = age > 70 and comorbidity_count >= 3
    moderate_severity = age > 60 and comorbidity_count >= 2

    if high_severity:
        seq += ["geriatric_consult", "social_work_referral", "case_management"]
    elif moderate_severity:
        seq += ["social_work_referral"] if random.random() < 0.6 else []

    seq += ["discharge_prep", "discharge"]
    return seq


def assign_discharge_pathway(department, age, comorbidity_count, admission_type):
    """Discharge pathway is influenced by age, comorbidities, and department -
    this creates a learnable signal for the CNN rather than pure randomness."""
    if age > 70 and comorbidity_count >= 3:
        weights = [0.25, 0.30, 0.30, 0.05, 0.10]
    elif department == "ICU":
        weights = [0.35, 0.25, 0.20, 0.10, 0.10]
    elif admission_type == "Elective":
        weights = [0.70, 0.10, 0.05, 0.05, 0.10]
    else:
        weights = [0.55, 0.15, 0.10, 0.10, 0.10]
    return random.choices(DISCHARGE_PATHWAYS, weights=weights, k=1)[0]


def generate_los(department, admission_type, age, comorbidity_count):
    """Length of stay (in days) with realistic dependencies baked in."""
    base = {
        "ICU": 6, "Surgery": 4, "Cardiology": 5,
        "General Ward": 3, "Pediatrics": 2
    }[department]

    if admission_type == "Emergency":
        base += 1.5
    age_factor = (age / 100) * 3
    comorbidity_factor = comorbidity_count * 0.8

    noise = np.random.gamma(shape=2, scale=1.0)
    los = base + age_factor + comorbidity_factor + noise
    return max(1, round(los))


records = []
for i in range(N_PATIENTS):
    patient_id = f"P{100000 + i}"
    age = int(np.clip(np.random.normal(55, 20), 0, 95))
    gender = random.choice(["Male", "Female"])
    department = random.choice(DEPARTMENTS)
    admission_type = random.choices(ADMISSION_TYPES, weights=[0.5, 0.3, 0.2])[0]
    diagnosis = random.choice(DIAGNOSIS_CODES)
    comorbidity_count = np.random.poisson(1.5 if age > 60 else 0.5)
    insurance = random.choices(INSURANCE_TYPES, weights=[0.5, 0.3, 0.15, 0.05])[0]
    num_prior_admissions = np.random.poisson(0.8)

    los = generate_los(department, admission_type, age, comorbidity_count)
    admission_date = fake.date_between(start_date="-2y", end_date="today")
    discharge_date = admission_date + timedelta(days=los)

    pathway = assign_discharge_pathway(department, age, comorbidity_count, admission_type)
    treatment_sequence = generate_treatment_sequence(department, admission_type, age, comorbidity_count)

    # Readmission risk: higher with more comorbidities, prior admissions, older age
    readmit_prob = (
        0.05 + 0.05 * comorbidity_count + 0.08 * num_prior_admissions
        + (0.1 if age > 70 else 0) + (0.05 if pathway == "Nursing Facility" else 0)
    )
    readmitted_30_days = 1 if random.random() < min(readmit_prob, 0.85) else 0

    records.append({
        "patient_id": patient_id,
        "admission_id": f"A{200000 + i}",
        "age": age,
        "gender": gender,
        "admission_type": admission_type,
        "diagnosis_code": diagnosis,
        "department": department,
        "admission_date": admission_date,
        "discharge_date": discharge_date,
        "length_of_stay": los,
        "num_prior_admissions": num_prior_admissions,
        "comorbidity_count": comorbidity_count,
        "insurance_type": insurance,
        "treatment_sequence": " ".join(treatment_sequence),
        "discharge_pathway": pathway,
        "readmitted_30_days": readmitted_30_days,
    })

df = pd.DataFrame(records)

output_path = "data/raw/hospital_admissions.csv"
import os
os.makedirs("data/raw", exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} synthetic admission records -> {output_path}")
print(df.head())
print("\nColumn dtypes:\n", df.dtypes)
