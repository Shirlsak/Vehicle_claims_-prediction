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
 
    predictions = []
 
    # Loop over the first 10 rows of the input data
    for i in range(min(10, len(input_data))):
        # Reshape the input data for single instance prediction
        input_data_reshaped = input_data.iloc[[i]].values.reshape(1, -1)
       
        prediction = loaded_model.predict(input_data_reshaped)
 
        if prediction[0] == 0:
            predictions.append('The claim in row {} is Normal'.format(i+1))
        else:
            predictions.append('The claim in row {} is Anomalous'.format(i+1))
 
    return predictions
 
def main():
    # Giving a title
    #st.title('Automobile Insurance Claim Anomally Prediction Web App')

     # Set page configuration
    #st.set_page_config(layout="wide")
 
    # Load background image
    background_image = Image.open("image2.jpg")

    # Resize image to desired dimensions
    resized_image = background_image.resize((300, 100))
    st.image(resized_image, use_column_width=True)
 
    # Adding styling for the title
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
    st.markdown('<div class="title-box"><h1 class="title">Automobile Insurance Claim Anomally Prediction</h1></div>', unsafe_allow_html=True)
 
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
            predictions = claim_fraud_prediction(data)
            for prediction in predictions:
                if 'Anomalous' in prediction:
                    st.markdown(f'<span class="anomalous">{prediction}</span>', unsafe_allow_html=True)
                else:
                    st.success(prediction)              
if __name__ == '__main__':
    main()