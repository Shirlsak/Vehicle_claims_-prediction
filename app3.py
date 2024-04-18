import numpy as np
from joblib import load
import streamlit as st
import pandas as pd
from sklearn import preprocessing
from PIL import Image

# Loading the saved model
loaded_model = load('fraud_model.joblib')

# Function for prediction
def claim_fraud_prediction(input_data):
    predictions = loaded_model.predict(input_data)
    return predictions

def main():
    # Load background image
    background_image = Image.open("image2.jpg")

    # Resize image to desired dimensions
    resized_image = background_image.resize((300, 100))
    st.image(resized_image, use_column_width=True)

    # Adding styling for the title
    st.markdown(
        """
        <style>
        .title-box {
            background-color: rgba(255, 255, 255, 0.7); /* Set background color with transparency */
            padding: 10px; /* Add padding */
            border-radius: 5px; /* Add border radius */
            margin-bottom: 20px; /* Add margin */
            display: flex; /* Use flexbox for centering */
            justify-content: center; /* Horizontally center the content */
            align-items: center; /* Vertically center the content */
        }
        .title {
            text-align: center; /* Center the text */
        }
        .anomalous {
            color: red; /* Set text color to red for anomalous claims */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Creating a box around the title
    st.markdown('<div class="title-box"><h1 class="title">Automobile Insurance Claim Anomaly Prediction Web App</h1></div>', unsafe_allow_html=True)

    # Option for user to input feature names and values directly
    st.sidebar.subheader("Input Features")
    input_features = {}
    features = ['Month', 'Accident_Area', 'Month_claimed', 'Sex', 'Age', 'Fault', 'Vehicle_Category', 'Vehicle_Price', 'Past_Numbers_of_claims', 'Age_of_Vehicle', 'Age_of_Policy_Holder', 'Agent_Type', 'Number_of_Suppliments', 'Year', 'Base_Policy']
    for feature_name in features:
        feature_value = st.sidebar.number_input(f"{feature_name}")
        input_features[feature_name] = feature_value
    input_data_direct = pd.DataFrame([input_features])

    # Option for user to upload CSV file
    st.sidebar.subheader("Upload CSV file")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read uploaded data
        data = pd.read_csv(uploaded_file)
        st.write(data)

        # Code for prediction
        if st.sidebar.button('Predict Claim Status (CSV)'):
            predictions = claim_fraud_prediction(data)
            for i, prediction in enumerate(predictions):
                if prediction == 0:
                    st.success(f'The claim in row {i+1} is Normal')
                else:
                    st.markdown(f'<span class="anomalous">The claim in row {i+1} is Anomalous</span>', unsafe_allow_html=True)

    # Code for prediction using input features directly
    if st.sidebar.button('Predict Claim Status (Direct Input)'):
        predictions = claim_fraud_prediction(input_data_direct)
        if predictions[0] == 0:
            st.success('The claim is Normal')
        else:
            st.markdown('<span class="anomalous">The claim is Anomalous</span>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
