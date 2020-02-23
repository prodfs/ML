# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:03:51 2020

@author: hp
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_assets import Bundle,Enviroment
import pickle

app = Flask(__name__)

#js=Bundle('jquery-3.2.1.min.js','popper.js','bootstrap.min.js','select2.min.js','tilt.jquery.min.js',output='gen/main.js')
#assets=Enviroment(app)
#assets.register('main.js',js)
#
model = pickle.load(open('model.pkl', 'rb'))

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
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Purchased 0 for NO and 1 for YES === {}'.format(output))





if __name__ == "__main__":

    app.run(debug=True)