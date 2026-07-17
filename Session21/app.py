'''
Ford Car Price Prediction Web App
'''

# Import Streamlit for building the web application
import streamlit as st

# Import Pandas for data manipulation
import pandas as pd

# Import Joblib for loading the trained model
import joblib

# --------------------------------------------------
# Load Saved Model and Preprocessing Objects
# --------------------------------------------------

model = joblib.load("LR_ford_car.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

# --------------------------------------------------
# Configure Streamlit Page
# --------------------------------------------------

st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🚗 Ford Car Price Predictor")

st.write("Enter the car details below to predict its selling price.")

st.divider()

# --------------------------------------------------
# Numerical Inputs
# --------------------------------------------------

year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2025,
    value=2018
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=300000,
    value=30000
)

tax = st.number_input(
    "Road Tax (£)",
    min_value=0,
    max_value=600,
    value=150
)

mpg = st.number_input(
    "Miles Per Gallon (MPG)",
    min_value=0.0,
    max_value=100.0,
    value=55.0
)

engineSize = st.number_input(
    "Engine Size (L)",
    min_value=0.0,
    max_value=6.0,
    value=1.5
)

# --------------------------------------------------
# Categorical Inputs
# --------------------------------------------------

transmission = st.selectbox(
    "Transmission",
    [
        "Automatic",
        "Manual",
        "Semi-Auto"
    ]
)

fuelType = st.selectbox(
    "Fuel Type",
    [
        "Petrol",
        "Diesel",
        "Hybrid",
        "Electric",
        "Other"
    ]
)

# --------------------------------------------------
# Text Input
# --------------------------------------------------

model_name = st.text_input(
    "Car Model"
)

# --------------------------------------------------
# Predict Button
# --------------------------------------------------

predict_button = st.button("Predict Price")

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    try:

        # Create input DataFrame
        input_data = pd.DataFrame({
            "model": [model_name],
            "year": [year],
            "transmission": [transmission],
            "mileage": [mileage],
            "fuelType": [fuelType],
            "tax": [tax],
            "mpg": [mpg],
            "engineSize": [engineSize]
        })

        # One-Hot Encoding
        input_data = pd.get_dummies(input_data)

        # Match columns with training data
        input_data = input_data.reindex(
            columns=encoded_columns,
            fill_value=0
        )

        # Numerical columns to scale
        numerical_columns = [
            "year",
            "mileage",
            "tax",
            "mpg",
            "engineSize"
        ]

        # Apply Standard Scaling
        input_data[numerical_columns] = scaler.transform(
            input_data[numerical_columns]
        )

        # Predict Price
        prediction = model.predict(input_data)

        # Display Prediction
        st.success(
            f"💷 Predicted Car Price: £{prediction[0]:,.2f}"
        )

    except Exception as e:

        st.error("An error occurred while making the prediction.")

        st.exception(e)