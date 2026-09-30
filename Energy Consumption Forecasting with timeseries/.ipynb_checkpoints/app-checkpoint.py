import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------
model = joblib.load("energy_demand_model.pkl")

# -------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------

st.title("Energy Consumption Forecasting")

st.write("Enter the required weather and time-series information.")

# -------------------------------------------------
# WEATHER INPUTS
# -------------------------------------------------

AWND = st.number_input(
    "Average Wind Speed (AWND)",
    value=0.0
)

PRCP = st.number_input(
    "Precipitation (PRCP)",
    value=0.0
)

TMAX = st.number_input(
    "Maximum Temperature (TMAX)",
    value=0.0
)

TMIN = st.number_input(
    "Minimum Temperature (TMIN)",
    value=0.0
)

# -------------------------------------------------
# TIME-SERIES FEATURES
# -------------------------------------------------

st.subheader("Previous Demand Information")

lag_1 = st.number_input(
    "Demand at previous time (lag_1)",
    value=0.0
)

lag_7 = st.number_input(
    "Demand 7 periods ago (lag_7)",
    value=0.0
)

rolling_7 = st.number_input(
    "7-period Rolling Average",
    value=0.0
)

rolling_30 = st.number_input(
    "30-period Rolling Average",
    value=0.0
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if st.button("Predict Energy Demand"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "AWND": [AWND],
        "PRCP": [PRCP],
        "TMAX": [TMAX],
        "TMIN": [TMIN],
        "lag_1": [lag_1],
        "lag_7": [lag_7],
        "rolling_7": [rolling_7],
        "rolling_30": [rolling_30]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.success(
        f"Predicted Energy Demand: {prediction[0]:.2f}"
    )

    # Show input values
    st.subheader("Input Data")
    st.dataframe(input_data)