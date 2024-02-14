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
    data = pd.DataFrame(imputation.transform(data), columns=data.columns)
    data = pd.DataFrame(winsor.transform(data), columns=data.columns)
    data = pd.DataFrame(scale.transform(data), columns=data.columns)
    return data

# Function to make predictions
def predict(data):
    processed_data = preprocess_input(data)
    prediction = rf_model.predict(processed_data)
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
        st.subheader("Uploaded Data")
        st.write(input_data)

        # Make predictions
        predictions = predict(input_data)

        # Display predictions
        st.subheader("Predictions")

        # Combine original data with predictions
        result_df = pd.concat([input_data, pd.DataFrame(predictions, columns=['Prediction'])], axis=1)
        st.write(result_df)

if __name__ == '__main__':
    main()
