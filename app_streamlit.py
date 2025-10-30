# app_streamlit.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Construction Delay & Cost Overrun Predictor",
    page_icon="🏗️",
    layout="wide"
)

# -------------------------------------------------------------
# Custom CSS for UCC Theme
# -------------------------------------------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #e8eef9;
        }
        .stButton button {
            background-color: #002060;
            color: white;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #cc0000;
            color: white;
        }
        footer {
            text-align: center;
            color: gray;
            font-size: 14px;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Sidebar with UCC Info
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://ucc.edu.gh/sites/default/files/ucc_logo.png", width=120)
    st.markdown("### **Project Information**")
    st.markdown("""
    **Name:** Ebenezer Gyasi Attah  
    **Programme:** MSc Information Technology  
    **Index Number:** MS/ITE/24/0026  
    **Institution:** University of Cape Coast  
    **Project Title:** *AI-Based Construction Delay and Cost Overrun Predictor*
    """)
    st.markdown("---")

# -------------------------------------------------------------
# Load Models
# -------------------------------------------------------------
@st.cache_resource
def load_models():
    bundle = joblib.load("best_model.pkl")
    return bundle

try:
    bundle = load_models()
    features = bundle["features"]
    delay_model = bundle["models"]["delay"]
    cost_model = bundle["models"]["cost"]
except Exception as e:
    st.error("⚠️ Model could not be loaded. Please ensure 'best_model.pkl' is present.")
    st.stop()

# -------------------------------------------------------------
# App Header
# -------------------------------------------------------------
st.title("🏗️ AI Predictor for Construction Delays & Cost Overruns")
st.write("This application predicts **project delay (months)** and **cost overrun (%)** based on project parameters using AI models.")

# -------------------------------------------------------------
# Input Section
# -------------------------------------------------------------
st.subheader("Single Prediction Input")

col1, col2 = st.columns(2)

with col1:
    planned_duration = st.number_input("Planned Duration (months)", min_value=1, value=12)
    estimated_cost = st.number_input("Estimated Cost (USD)", min_value=1000, value=1200000)
    labour_hours = st.number_input("Labour Hours (man-days)", min_value=1, value=8000)

with col2:
    weather_severity = st.slider("Weather Severity (1–5)", 1, 5, 3)
    supply_disruption = st.selectbox("Supply Chain Disruption (0 = No, 1 = Yes)", [0, 1])
    contractor_experience = st.number_input("Contractor Experience (years)", min_value=0, value=10)

# -------------------------------------------------------------
# Prediction Logic
# -------------------------------------------------------------
if st.button("Predict for Inputs Above"):
    X = pd.DataFrame([{
        "Planned_Duration_Months": planned_duration,
        "Estimated_Cost_USD": estimated_cost,
        "Labour_Hours": labour_hours,
        "Weather_Severity": weather_severity,
        "Supply_Chain_Disruption": supply_disruption,
        "Contractor_Experience_Years": contractor_experience
    }])[features]

    delay_pred = float(delay_model.predict(X)[0])
    cost_pred = float(cost_model.predict(X)[0])

    delay_pred = max(0, delay_pred)
    cost_pred = max(0, cost_pred)

    st.success(f"**Predicted Delay:** {delay_pred:.2f} months")
    st.success(f"**Predicted Cost Overrun:** {cost_pred:.2f} %")

# -------------------------------------------------------------
# Batch CSV Upload
# -------------------------------------------------------------
st.subheader("Batch Prediction via CSV Upload")

st.markdown("""
Upload a CSV file with the following columns:  
`Planned_Duration_Months`, `Estimated_Cost_USD`, `Labour_Hours`, `Weather_Severity`,  
`Supply_Chain_Disruption`, `Contractor_Experience_Years`
""")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            delays = delay_model.predict(df[features])
            costs = cost_model.predict(df[features])
            df["Predicted_Delay_Months"] = np.maximum(0, delays)
            df["Predicted_Cost_Overrun_Percent"] = np.maximum(0, costs)

            st.dataframe(df.head())
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv", key="download-csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# -------------------------------------------------------------
# Feature Importance Visualization
# -------------------------------------------------------------
st.subheader("Model Information")

try:
    importance = delay_model.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Delay Model)")
    st.pyplot(fig)
except Exception:
    st.info("Feature importance plot unavailable for this model.")

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.markdown("""
---
<div style='text-align: center; color: gray; font-size: 14px;'>
Developed by <b>Ebenezer Gyasi Attah</b> | MSc IT | MS/ITE/24/0026 | University of Cape Coast © 2025
</div>
""", unsafe_allow_html=True)
