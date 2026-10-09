# ============================================================
# INSURANCE FRAUD DETECTION - STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os


# ============================================================
# 0. PROJECT PATH SETUP
# ============================================================

# Current App folder
app_dir = os.path.dirname(
    os.path.abspath(__file__)
)

# Main project folder
project_root = os.path.dirname(
    app_dir
)

# Development folder
development_path = os.path.join(
    project_root,
    "Development"
)

# Allow imports from Development folder
sys.path.insert(
    0,
    development_path
)


# ============================================================
# IMPORT PROJECT FUNCTIONS
# ============================================================

from feature_enggineering import create_features
from model_monitoring import save_predictions


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Insurance Fraud Detection")

st.write(
    "Enter insurance policy and incident details to predict "
    "whether the insurance claim may be fraudulent."
)


# ============================================================
# 2. LOAD PREPROCESSING PIPELINE AND MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Path of preprocessing pipeline
    preprocessing_path = os.path.join(
        project_root,
        "preprocessing_pipeline.pkl"
    )

    # Path of trained model
    model_path = os.path.join(
        project_root,
        "fraud_detection_model.pkl"
    )

    # Load preprocessing
    preprocessing_pipeline = joblib.load(
        preprocessing_path
    )

    # Load trained model
    model = joblib.load(
        model_path
    )

    return preprocessing_pipeline, model


preprocessing_pipeline, model = load_model()


# ============================================================
# 3. POLICY INFORMATION
# ============================================================

st.header("1. Policy Information")

col1, col2, col3 = st.columns(3)


with col1:

    months_as_customer = st.number_input(
        "Months as Customer",
        min_value=0,
        value=50
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40
    )

    policy_state = st.selectbox(
        "Policy State",
        ["OH", "IN", "IL"]
    )


with col2:

    policy_csl = st.selectbox(
        "Policy CSL",
        [
            "100/300",
            "250/500",
            "500/1000"
        ]
    )

    policy_deductable = st.number_input(
        "Policy Deductible",
        min_value=0,
        value=1000
    )

    policy_annual_premium = st.number_input(
        "Annual Premium",
        min_value=0.0,
        value=1200.0
    )


with col3:

    umbrella_limit = st.number_input(
        "Umbrella Limit",
        value=0
    )

    policy_bind_date = st.date_input(
        "Policy Bind Date"
    )

    insured_sex = st.selectbox(
        "Insured Sex",
        ["MALE", "FEMALE"]
    )


# ============================================================
# 4. CUSTOMER INFORMATION
# ============================================================

st.header("2. Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    insured_education_level = st.selectbox(
        "Education Level",
        [
            "High School",
            "Associate",
            "College",
            "Masters",
            "JD",
            "MD",
            "PhD"
        ]
    )

    insured_occupation = st.text_input(
        "Occupation",
        value="craft-repair"
    )


with col2:

    insured_hobbies = st.text_input(
        "Hobby",
        value="reading"
    )

    insured_relationship = st.selectbox(
        "Relationship",
        [
            "husband",
            "wife",
            "own-child",
            "unmarried",
            "other-relative",
            "not-in-family"
        ]
    )


with col3:

    capital_gains = st.number_input(
        "Capital Gains",
        value=0
    )

    capital_loss = st.number_input(
        "Capital Loss",
        value=0
    )


# ============================================================
# 5. INCIDENT INFORMATION
# ============================================================

st.header("3. Incident Information")

col1, col2, col3 = st.columns(3)


with col1:

    incident_date = st.date_input(
        "Incident Date"
    )

    incident_type = st.selectbox(
        "Incident Type",
        [
            "Single Vehicle Collision",
            "Multi-vehicle Collision",
            "Vehicle Theft",
            "Parked Car"
        ]
    )

    collision_type = st.selectbox(
        "Collision Type",
        [
            "Rear Collision",
            "Side Collision",
            "Front Collision",
            "?"
        ]
    )


with col2:

    incident_severity = st.selectbox(
        "Incident Severity",
        [
            "Trivial Damage",
            "Minor Damage",
            "Major Damage",
            "Total Loss"
        ]
    )

    authorities_contacted = st.selectbox(
        "Authorities Contacted",
        [
            "Police",
            "Fire",
            "Ambulance",
            "Other",
            "None"
        ]
    )

    incident_state = st.text_input(
        "Incident State",
        value="NY"
    )


with col3:

    incident_city = st.text_input(
        "Incident City",
        value="Springfield"
    )

    incident_hour = st.slider(
        "Incident Hour",
        min_value=0,
        max_value=23,
        value=12
    )

   