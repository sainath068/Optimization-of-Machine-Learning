# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib

# Load saved models and transformers
scale = joblib.load('minmax')
winsor = joblib.load('winsor')
imputation = joblib.load('meanimpute')
rf_model = joblib.load('rfc.pkl')

# Function to preprocess input data
def preprocess_input(data):
    imputed_data = pd.DataFrame(imputation.transform(data), columns=data.columns)
    winsorized_data = pd.DataFrame(winsor.transform(imputed_data), columns=imputed_data.columns)
    scaled_data = pd.DataFrame(scale.transform(winsorized_data), columns=winsorized_data.columns)
    return scaled_data, imputed_data

# Function to make predictions
def predict(data):
    prediction = rf_model.predict(data)
    return prediction

# Streamlit app
def main():
    st.title("Machine Downtime Prediction App")

    # Upload CSV file through Streamlit
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read data from the uploaded file
        input_data = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.subheader("Original Data")
        st.write(input_data)

        # Preprocess input data and make predictions
        scaled_data, imputed_data = preprocess_input(input_data)
        predictions = predict(scaled_data)

        # Combine imputed data and predictions
        result_df = pd.concat([imputed_data, pd.DataFrame({'Predictions': predictions})], axis=1)

        # Display the result
        st.subheader("Imputed Data with Predictions")
        st.write(result_df)

if __name__ == '__main__':
    main()
