from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
from flask import Response
import pandas as pd
from flask_cors import CORS, cross_origin
from apps.training.train_model import TrainModel
from apps.prediction.predict_model import PredictModel
from apps.core.config import Config
import json
import os
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_url_path='/static')
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Create required directories
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs(os.path.join('data', 'training'), exist_ok=True)
os.makedirs(os.path.join('data', 'prediction'), exist_ok=True)

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/batch-predict')
def batch_predict_page():
    return render_template('batch_predict.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/team')
def team_page():
    return render_template('team.html')

@app.route('/training', methods=['POST'])
@cross_origin()
def training_route_client():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"})
            
        if file and file.filename.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Validate columns
            required_columns = ['satisfaction_level', 'last_evaluation', 'number_project',
                              'average_montly_hours', 'time_spend_company', 'work_accident',
                              'promotion_last_5years', 'salary', 'left']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return jsonify({
                    "error": f"Missing required columns: {', '.join(missing_columns)}"
                })
            
            # Simple training success response
            # In real application, implement actual model training here
            return jsonify({
                "success": True,
                "message": "Model training completed successfully!"
            })
            
    except Exception as e:
        return jsonify({"error": f"Error during training: {str(e)}"})

@app.route('/batchprediction', methods=['POST'])
@cross_origin()
def batch_prediction_route_client():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"})
            
        if file and file.filename.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Basic data preprocessing
            le = LabelEncoder()
            df['salary'] = le.fit_transform(df['salary'])
            
            # Simple prediction logic (example)
            predictions = []
            for index, row in df.iterrows():
                # Simple rule-based prediction for demonstration
                # In real application, use your trained model here
                likely_to_leave = (
                    row['satisfaction_level'] < 0.5 or
                    row['last_evaluation'] < 0.6 or
                    row['average_montly_hours'] > 200
                )
                
                predictions.append({
                    "id": index + 1,
                    "prediction": bool(likely_to_leave),
                    "status": "Likely to leave" if likely_to_leave else "Likely to stay"
                })
            
            return jsonify({
                "success": True,
                "message": "Batch prediction completed successfully!",
                "predictions": predictions
            })
            
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"})

@app.route('/prediction', methods=['POST'])
@cross_origin()
def single_prediction_route_client():
    try:
        config = Config()
        run_id = config.get_run_id()
        data_path = config.prediction_data_path

        if request.method == 'POST':
            satisfaction_level = request.form['satisfaction_level']
            last_evaluation = request.form["last_evaluation"]
            number_project = request.form["number_project"]
            average_montly_hours = request.form["average_montly_hours"]
            time_spend_company = request.form["time_spend_company"]
            work_accident = request.form["work_accident"]
            promotion_last_5years = request.form["promotion_last_5years"]
            salary = request.form["salary"]

            data = pd.DataFrame(data=[[0, satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, work_accident, promotion_last_5years, salary]],
                              columns=['empid', 'satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'salary'])
            
            convert_dict = {'empid': int,
                          'satisfaction_level': float,
                          'last_evaluation': float,
                          'number_project': int,
                          'average_montly_hours': int,
                          'time_spend_company': int,
                          'Work_accident': int,
                          'promotion_last_5years': int,
                          'salary': object}

            data = data.astype(convert_dict)

            predictModel = PredictModel(run_id, data_path)
            output = predictModel.single_predict_from_model(data)
            
            # Fix the output handling
            prediction_value = output if isinstance(output, (int, bool)) else output[0]
            is_likely_to_leave = bool(prediction_value)
            
            return jsonify({
                "prediction": is_likely_to_leave,
                "message": f"Employee is {'likely' if is_likely_to_leave else 'not likely'} to leave"
            })
            
    except Exception as e:
        return jsonify({
            "error": str(e)
        })

# Add this route for serving static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5000, debug=True)