import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PART 1: LOAD THE BACKEND ---
# Using the cache to load the model faster
@st.cache_resource
def load_models():
    # Load the files
    model = joblib.load('xgb_final_99acc.joblib')
    le = joblib.load('label_encoder.joblib')
    imputer = joblib.load('imputer.joblib')
    return model, le, imputer

try:
    model, le, imputer = load_models()
    st.toast("System Loaded")
except FileNotFoundError:
    st.error("Error.")
    st.stop()

# --- PART 2: FRINTEND ---
st.set_page_config(page_title="Smart CBC Diagnosis", page_icon="🩸")

st.title("🏥 AI-Powered Disease Diagnosis System")
st.markdown("Enter the patient's CBC (Complete Blood Count) values below.")

col1, col2, col3 = st.columns(3)

with col1:
    hgb = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=13.0, step=0.1)
    rbc = st.number_input("RBC Count (m/mcL)", min_value=0.0, max_value=10.0, value=4.8, step=0.1)
    mcv = st.number_input("MCV (fL)", min_value=0.0, max_value=150.0, value=85.0, step=1.0)

with col2:
    wbc = st.number_input("WBC Count (cells/uL)", min_value=0.0, max_value=50000.0, value=7500.0, step=100.0)
    plt_count = st.number_input("Platelets (cells/uL)", min_value=0.0, max_value=1000000.0, value=250000.0, step=1000.0)
    mch = st.number_input("MCH (pg)", min_value=0.0, max_value=50.0, value=29.0, step=0.1)

with col3:
    hct = st.number_input("Hematocrit (%)", min_value=0.0, max_value=80.0, value=40.0, step=0.1)
    mchc = st.number_input("MCHC (g/dL)", min_value=0.0, max_value=50.0, value=33.0, step=0.1)

# --- PART 3: MATCH ENGINE ---
def make_prediction(hgb, wbc, rbc, hct, mcv, mch, mchc, plt_count):
    # 1. Feature Engineering 
    pwr = plt_count / (wbc + 1e-5)
    hpr = hgb / (plt_count + 1e-5)
    anemia_idx = hgb * rbc
    
    #SCALING THE VALS
    
    wbc_scaled = wbc if wbc < 100 else wbc / 1000.0 
    plt_scaled = plt_count if plt_count < 1000 else plt_count / 1000.0
    
    # Recalculate ratios with scaled values
    pwr = plt_scaled / (wbc_scaled + 1e-5)
    hpr = hgb / (plt_scaled + 1e-5)
    
    features = pd.DataFrame([[
        hgb, wbc_scaled, rbc, hct, mcv, mch, mchc, plt_scaled, 
        pwr, hpr, anemia_idx
    ]], columns=['Hemoglobin', 'WBC', 'RBC', 'Hematocrit', 'MCV', 'MCH', 'MCHC', 'Platelets', 'PWR', 'HPR', 'Anemia_Index'])

    # 3.  AI Prediction
    prediction_idx = model.predict(features)[0]
    initial_diagnosis = le.inverse_transform([prediction_idx])[0]
    
    final_diagnosis = initial_diagnosis
    
    #  Infection Override
    if wbc_scaled > 12.0:
        final_diagnosis = "Infection"
        
    # Dengue Override 
    elif plt_scaled < 100 and initial_diagnosis == "Anemia":
        final_diagnosis = "Dengue"
        
    return final_diagnosis, initial_diagnosis

# --- PART 4: RESULT ---
if st.button(" Diagnose Patient", type="primary"):
    
    result, raw_ai = make_prediction(hgb, wbc, rbc, hct, mcv, mch, mchc, plt_count)
    
    # Display Logic
    if result == "Healthy":
        st.success(f"### Diagnosis: {result}")
        st.balloons()
    elif result == "Dengue":
        st.error(f"### ⚠️ Diagnosis: {result}")
        st.warning("Action: Check for fluid leakage. Monitor Platelets daily.")
    elif result == "Infection":
        st.error(f"### ⚠️ Diagnosis: {result}")
        st.warning("Action: Check temperature. Possible antibiotics required.")
    else:
        st.warning(f"### Diagnosis: {result}")
        
    with st.expander("See System Logic"):
        st.write(f"**Raw AI Guess:** {raw_ai}")
        st.write(f"**Final System Decision:** {result}")
        if result != raw_ai:
            st.info("Clinical Safety Rules overrode the AI decision.")