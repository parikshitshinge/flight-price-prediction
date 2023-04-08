from flask import Flask, request, render_template
import numpy as np
import pandas as pd 

from src.Pipelines.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(airline, ch_code, from_, stop, to_, type_, dep_time_phase, arr_time_phase)
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results = results[0])