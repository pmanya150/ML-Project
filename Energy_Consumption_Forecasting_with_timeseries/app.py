import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="centered"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp {
    background-color: #000000;
    color: white;
}

h1 {
    text-align: center;
    color: white;
    font-size: 42px;
    font-weight: 700;
}

h2, h3 {
    color: white;
}

p {
    color: #dddddd;
    font-size: 17px;
}

label {
    color: #eeeeee !important;
}

.stNumberInput input {
    background-color: #1a1a1a !important;
    color: white !important;
    border: 1px solid #555555 !important;
    border-radius: 8px;
}

.stNumberInput input:focus {
    border-color: white !important;
}

.stButton > button {
    width: 100%;
    background-color: red;
    color: black;
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    font-size: 18px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #dddddd;
    color: black;
}

.stAlert {
    background-color: #111111;
    color: white;
    border: 1px solid #555555;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
model = joblib.load("energy_demand_model.pkl")


# ---------------- TITLE ----------------
st.title("⚡ Energy Consumption Forecasting")

st.write(
    "Enter the required weather, date and previous demand information "
    "to predict energy consumption."
)


# ---------------- DATE INFORMATION ----------------
st.subheader("📅 Date Information")

year = st.number_input(
    "Year",
    min_value=2000,
    max_value=2100,
    value=2026,
    step=1
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=1,
    step=1
)

day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=1,
    step=1
)

day_of_week = st.number_input(
    "Day of Week",
    min_value=0,
    max_value=6,
    value=0,
    step=1
)


# ---------------- WEATHER INFORMATION ----------------
st.subheader("🌦️ Weather Information")

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


# ---------------- TIME SERIES FEATURES ----------------
st.subheader("📊 Previous Demand Information")

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


st.markdown("---")


# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Energy Demand"):

    input_data = pd.DataFrame({
        "AWND": [AWND],
        "PRCP": [PRCP],
        "TMAX": [TMAX],
        "TMIN": [TMIN],

        "year": [year],
        "month": [month],
        "day": [day],
        "day_of_week": [day_of_week],

        "lag_1": [lag_1],
        "lag_7": [lag_7],
        "rolling_7": [rolling_7],
        "rolling_30": [rolling_30]
    })

    # Make sure feature order is exactly the same
    # as the features used during model training
    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Energy Demand: {prediction[0]:.2f}"
    )

    st.subheader("📋 Input Data")

    st.dataframe(
        input_data,
        use_container_width=True
    )