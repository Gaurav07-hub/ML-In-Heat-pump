import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("cop_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Heat Pump COP Prediction")
st.write("Enter values to predict the Coefficient of Performance (COP)")

# User input
evaporator_temp = st.number_input("Evaporator Temperature (°C)", min_value=-10.0, max_value=20.0, value=5.0)
condenser_temp = st.number_input("Condenser Temperature (°C)", min_value=20.0, max_value=60.0, value=40.0)
power_input = st.number_input("Power Input (kW)", min_value=0.5, max_value=10.0, value=2.5)

# Predict COP
if st.button("Predict COP"):
    input_data = np.array([[evaporator_temp, condenser_temp, power_input]])
    input_scaled = scaler.transform(input_data)
    cop_pred = model.predict(input_scaled)[0]
    st.success(f"Predicted COP: {cop_pred:.2f}")
