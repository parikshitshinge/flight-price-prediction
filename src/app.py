import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from flask import Flask, request, render_template
import numpy as np
import pandas as pd 

from pipelines.predict_pipeline import CustomData, PredictPipeline

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
        data = CustomData(
            airline = request.form.get('airline'),
            from_ = request.form.get('from'),
            to_ = request.form.get('to'),
            type_ = request.form.get('type'),
            dep_time_phase = request.form.get('departure_phase'),
            arr_time_phase = request.form.get('departure_phase')
            )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', prediction_result = results[0])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)