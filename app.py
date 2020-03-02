# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:03:51 2020

@author: hp
"""

import numpy as np
from flask import Flask, request, render_template,jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST']) 
def predict():
    '''

    For rendering results on HTML GUI

    '''

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    print(prediction)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Purchased'.format(output))

    
if __name__ == "__main__":
    app.run(debug=True)