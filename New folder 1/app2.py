import numpy as np
from joblib import load
import streamlit as st
import pandas as pd
from sklearn import preprocessing
 
# Loading the saved model
loaded_model = load('fraud_model.joblib')
 
# Function for prediction
def claim_fraud_prediction(input_data):
    # Reshaping the input data for single instance prediction
    input_data_reshaped = input_data.values.reshape(1, -1)
 
    prediction = loaded_model.predict(input_data_reshaped)
 
    if prediction[0] == 0:
        return 'The claim is legitimate'
    else:
        return 'The claim is fraudulent'
 
def main():
    # Giving a title
    st.title('Claim Fraud Prediction Web App')
 
    # Upload data file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read uploaded data
        data = pd.read_csv(uploaded_file)
        st.write(data)
 
        # Code for prediction
        diagnosis = ''
 
        # Creating a button for prediction
        if st.button('Predict Claim Status'):
            diagnosis = claim_fraud_prediction(data)
 
        st.success(diagnosis)

if __name__ == '__main__':
    main()
 
