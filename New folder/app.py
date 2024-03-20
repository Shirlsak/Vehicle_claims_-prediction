import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

app = Flask(__name__)
##Load the model
model = pickle.load(open('fraud_model.pkl', 'rb'))

cols=['Age','Fault','Base_Policy','Month_Claimed','Month','Year','Past_Number_of_claims','Age_of_Vehicle', 
      'Vehicle_Price','Vehicle_category','Number_of_suppliments','Age_of_Policy_holder','Sex','Accident_Area','Agent_Type'] 
      

@app.route('/') 
def home(): 
    return render_template('index.html') 
    

@app.route('/predict',methods=['POST','GET']) 
def predict(): 
    feature_dict = request.form.to_dict() 
    feature_list = list(feature_list.values()) 
    feature_list = list(map(float, feature_list)) 
    final_features = np.array(feature_list).reshape(1, -1) 
    
    prediction = model.predict(final_features) 
    output = int(prediction[0]) 
    if output == 1: 
        text = "Frauduelent claim" 
    else: 
        text = "Legitimate claim"
    
    return render_template('index.html', prediction_text='The claim is {}'.format(text)) 
    
    
if __name__ == "__main__": 
    app.run(debug=True)