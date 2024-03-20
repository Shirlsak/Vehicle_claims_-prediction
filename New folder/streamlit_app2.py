import streamlit as st
import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('fraud_model.pkl', 'rb'))

# creating a function for Prediction
def fraud_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The claim is legitimate'
    else:
        return 'The claim is fraudulent'

def main():
    # giving a title
    st.title('Claim Fraud Prediction Web App')

    # getting the input data from the user
    age = st.text_input('Age')
    fault = st.text_input('Fault')
    base_policy = st.text_input('Base Policy')
    month_claimed = st.text_input('Month Claimed')
    month = st.text_input('Month')
    year = st.text_input('Year')
    past_num_claims = st.text_input('Past Number of Claims')
    vehicle_age = st.text_input('Age of Vehicle')
    vehicle_price = st.text_input('Vehicle Price')
    vehicle_category = st.text_input('Vehicle Category')
    num_supplements = st.text_input('Number of Supplements')
    policyholder_age = st.text_input('Age of Policy Holder')
    sex = st.text_input('Sex')
    accident_area = st.text_input('Accident Area')
    agent_type = st.text_input('Agent Type')

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict Claim Status'):
        diagnosis = fraud_prediction([age, fault, base_policy, month_claimed, month, year, past_num_claims, vehicle_age, vehicle_price, vehicle_category, num_supplements, policyholder_age, sex, accident_area, agent_type])

    st.success(diagnosis)

if __name__ == '__main__':
    main()