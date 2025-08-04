import streamlit as st
import joblib
import pandas as pd

# Load the model 
model = joblib.load("model/status_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

st.set_page_config(page_title="Tire Failure Predictor", layout="centered")
st.title("🔧 Predictive Tire Maintenance")
st.subheader("Enter tire info below to predict failure risk")

# User inputs
tread_depth = st.slider("Tread Depth (mm)", min_value=0.0, max_value=20.0, value=6.0, step=0.1)
pressure = st.slider("Pressure (PSI)", min_value=60.0, max_value=150.0, value=100.0, step=1.0)
mileage = st.number_input("Mileage (miles)", min_value=0, max_value=150000, value=45000)
age_months = st.slider("Tire Age (Months)", 0, 100, 24)
temperature = st.slider("Temperature (°F)", min_value=80.0, max_value=250.0, value=150.0)
status = st.selectbox("Visual Status", label_encoder.classes_)

if st.button("🔍 Predict Failure"):
    try:
        status_encoded = label_encoder.transform([status])[0]
        input_data = pd.DataFrame([[
            tread_depth, pressure, mileage, age_months, temperature, status_encoded
        ]], columns=['tread_depth', 'pressure', 'mileage', 'age_months', 'temperature', 'status_encoded'])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.success(f"Prediction: {'⚠️ INSPECT' if prediction == 1 else '✅ NO ACTION NEEDED'}")
        st.info(f"Probability of Failure: {probability:.2%}")

    except Exception as e:
        st.error(f"Error: {e}")
