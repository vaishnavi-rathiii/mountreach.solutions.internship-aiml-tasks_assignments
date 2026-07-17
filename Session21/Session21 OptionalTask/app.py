'''
House Price Prediction Web Application

This Streamlit application predicts the selling price of a house
using a trained Linear Regression model.
'''

# --------------------------------------------------
# Import Required Libraries
# --------------------------------------------------

# Import Streamlit for building the web application
import streamlit as st

# Import Pandas for handling data
import pandas as pd

# Import Joblib for loading the saved machine learning model
import joblib

# --------------------------------------------------
# Load Saved Model and Preprocessing Objects
# --------------------------------------------------

# Load the trained Linear Regression model
model = joblib.load("LR_House_Price.pkl")

# Load the StandardScaler object
scaler = joblib.load("scaler.pkl")

# Load the encoded column names
encoded_columns = joblib.load("columns.pkl")

# --------------------------------------------------
# Configure Streamlit Page
# --------------------------------------------------

# page_title sets the browser tab title.
# layout="centered" keeps the application centered on the page.
st.set_page_config(
    page_title="House Price Predictor",
    layout="centered"
)

# --------------------------------------------------
# Title and Description
# --------------------------------------------------

st.title("🏡 House Price Prediction")

st.write("Enter the house details below to predict its selling price.")

st.divider()

# --------------------------------------------------
# Numerical Input Fields
# --------------------------------------------------

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1.0,
    max_value=10.0,
    value=2.0
)

sqft_living = st.number_input(
    "Living Area (sqft)",
    min_value=300,
    max_value=15000,
    value=2000
)

sqft_lot = st.number_input(
    "Lot Area (sqft)",
    min_value=500,
    max_value=1000000,
    value=5000
)

floors = st.number_input(
    "Floors",
    min_value=1.0,
    max_value=5.0,
    value=1.0
)

waterfront = st.number_input(
    "Waterfront (0 = No, 1 = Yes)",
    min_value=0,
    max_value=1,
    value=0
)

view = st.number_input(
    "View Rating",
    min_value=0,
    max_value=4,
    value=0
)

condition = st.number_input(
    "Condition",
    min_value=1,
    max_value=5,
    value=3
)

sqft_above = st.number_input(
    "Sqft Above Ground",
    min_value=300,
    max_value=10000,
    value=1500
)

sqft_basement = st.number_input(
    "Sqft Basement",
    min_value=0,
    max_value=5000,
    value=0
)

yr_built = st.number_input(
    "Year Built",
    min_value=1900,
    max_value=2025,
    value=2000
)

yr_renovated = st.number_input(
    "Year Renovated (0 if Never)",
    min_value=0,
    max_value=2025,
    value=0
)

year = st.number_input(
    "Sale Year",
    min_value=2014,
    max_value=2025,
    value=2014
)

month = st.number_input(
    "Sale Month",
    min_value=1,
    max_value=12,
    value=5
)

# --------------------------------------------------
# Categorical Input Fields
# --------------------------------------------------

# Text input is used to enter city name.
city = st.text_input("City")

# Text input is used to enter state zip.
statezip = st.text_input("State ZIP")

# Selectbox provides predefined options and prevents invalid inputs.
country = st.selectbox(
    "Country",
    ["USA"]
)

# --------------------------------------------------
# Predict Button
# --------------------------------------------------

predict_button = st.button("Predict House Price")

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    try:

        # Create DataFrame from user inputs
        input_data = pd.DataFrame({
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "sqft_living": [sqft_living],
            "sqft_lot": [sqft_lot],
            "floors": [floors],
            "waterfront": [waterfront],
            "view": [view],
            "condition": [condition],
            "sqft_above": [sqft_above],
            "sqft_basement": [sqft_basement],
            "yr_built": [yr_built],
            "yr_renovated": [yr_renovated],
            "city": [city],
            "statezip": [statezip],
            "country": [country],
            "year": [year],
            "month": [month]
        })

        # Perform One-Hot Encoding
        input_data = pd.get_dummies(input_data)

        # Align columns with training data
        input_data = input_data.reindex(
            columns=encoded_columns,
            fill_value=0
        )

        # Numerical columns used for scaling
        numerical_columns = [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "waterfront",
            "view",
            "condition",
            "sqft_above",
            "sqft_basement",
            "yr_built",
            "yr_renovated",
            "year",
            "month"
        ]

        # Apply Standard Scaling
        input_data[numerical_columns] = scaler.transform(
            input_data[numerical_columns]
        )

        # Predict house price
        prediction = model.predict(input_data)

        # Display predicted price
        st.success(
            f"🏡 Predicted House Price: ${prediction[0]:,.2f}"
        )

    except Exception as e:

        st.error("An error occurred while making the prediction.")

        st.exception(e)