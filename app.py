import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from Titanic_test_functions import test_preprocessing

app=Flask(__name__)

## Load the ml and scaler model
model=pickle.load(open('Titanic_Model.pkl','rb'))
encoder=pickle.load(open("Titanic_Encoder.pkl","rb"))

## For the default home page
@app.route('/')
def home():
    return render_template('index.html')

## For the api
@app.route('/predict_api',methods=['POST'])
def predict_api():

    #Requesting the data in json format
    data = request.json['data']

    #Converting json->list->np array and reshaping the array
    #So that it can be fed into model for prediction
    x_test = test_preprocessing(list(data.values()),encoder)

    #Doing the prediction
    y_pred = model.predict(x_test)
    print(y_pred[0])
    
    return jsonify(int(y_pred[0]))

@app.route('/predict',methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    final_input=test_preprocessing(np.array(data).reshape(1,-1),encoder)
    y_pred = model.predict(final_input)
    if int(y_pred[0])==1:
        output="Survive!"
    else:
        output="Die."
    return render_template("index.html",prediction_text=f"The passenger will {output}")


if __name__=="__main__":
    app.run(debug=True)