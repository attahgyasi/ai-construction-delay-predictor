
import streamlit as st

# --- Project Branding ---
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

>>>>>>> bb406ab9af2b5deb0202c90d0169fa8a98580ef8
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Construction Delay & Cost Overrun Predictor", layout="wide")

st.title("AI Predictor for Construction Delays & Cost Overruns")
st.markdown("Enter project parameters or upload a CSV to get predictions for delay (months) and cost overrun (%).")

model_path = Path("best_model.pkl")

@st.cache_resource
def load_bundle():
    if not model_path.exists():
        st.error("Model file not found. Please run `python train_model.py` first to generate 'best_model.pkl'.")
        st.stop()
    return joblib.load(model_path)

bundle = load_bundle()
FEATURES = bundle["features"]
delay_model = bundle["models"]["delay"]
cost_model  = bundle["models"]["cost"]

st.sidebar.header("Single Prediction Input")
def sidebar_inputs():
    inputs = {}
    inputs["Planned_Duration_Months"] = st.sidebar.slider("Planned Duration (months)", 6, 24, 12)
    inputs["Estimated_Cost_USD"] = st.sidebar.number_input("Estimated Cost (USD)", min_value=100000, max_value=10000000, value=1200000, step=50000)
    inputs["Labour_Hours"] = st.sidebar.slider("Labour Hours (man-days)", 2000, 20000, 8000, step=100)
    inputs["Weather_Severity"] = st.sidebar.slider("Weather Severity (1-5)", 1, 5, 3)
    inputs["Supply_Chain_Disruption"] = st.sidebar.selectbox("Supply Chain Disruption", [0, 1], index=0)
    inputs["Contractor_Experience_Years"] = st.sidebar.slider("Contractor Experience (years)", 1, 30, 10)
    return pd.DataFrame([inputs])

single_df = sidebar_inputs()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Single Project Prediction")
    if st.button("Predict for Inputs Above"):
        X = single_df[FEATURES]
        delay_pred = delay_model.predict(X)[0]
        cost_pred  = cost_model.predict(X)[0]
        delay_pred = max(0.0, float(delay_pred))
        cost_pred  = max(0.0, float(cost_pred))
        st.metric("Predicted Delay (months)", f"{delay_pred:.2f}")
        st.metric("Predicted Cost Overrun (%)", f"{cost_pred:.2f}%")

with col2:
    st.subheader("Batch Prediction via CSV Upload")
    uploaded = st.file_uploader("Upload CSV with columns: " + ", ".join(FEATURES), type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            delays = delay_model.predict(df[FEATURES])
            costs  = cost_model.predict(df[FEATURES])
            preds = pd.DataFrame({
                "Predicted_Delay_Months": np.maximum(0.0, delays),
                "Predicted_Cost_Overrun_Percent": np.maximum(0.0, costs),
            })
            st.write("Predictions:")
            st.dataframe(pd.concat([df.reset_index(drop=True), preds], axis=1))

st.markdown("---")
st.subheader("Model Information")
st.write(f"Delay model: **{bundle['delay_model_name']}** | Cost model: **{bundle['cost_model_name']}**")
if Path("feature_importance.png").exists():
    st.image("feature_importance.png", caption="Feature Importance (Best Model)", use_container_width=True)
else:
    st.info("Train the model to generate feature importance visualization.")

st.markdown("**How to run locally**")
st.code("""
pip install -r requirements.txt
python train_model.py
streamlit run app_streamlit.py
""")
<<<<<<< HEAD

# --- Footer Section ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: gray;'>
    Developed by <b>Ebenezer Gyasi Attah</b> | MSc IT | MS/ITE/24/0026 | University of Cape Coast © 2025
    </div>
    """,
    unsafe_allow_html=True
)
=======
>>>>>>> bb406ab9af2b5deb0202c90d0169fa8a98580ef8
