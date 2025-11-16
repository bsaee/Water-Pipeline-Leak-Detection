import pandas as pd
import joblib
from flask import Flask, request, jsonify
import numpy as np

# 1. Initialize the Flask App
app = Flask(__name__)

# 2. Load Our Trained Model and Scaler
try:
    model = joblib.load('leak_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: 'leak_model.pkl' or 'scaler.pkl' not found.")
    print("Please make sure both files are in the same folder as app.py")
    exit()

# 3. Define our feature list
# This MUST be in the exact same order as it was during training
feature_columns = [
    'Pressure (bar)', 'Flow Rate (L/s)', 'Temperature (°C)',
       'Pressure (bar)_roll_mean', 'Pressure (bar)_roll_std','Pressure (bar)_lag_1', 'Pressure (bar)_diff',
       'Flow Rate (L/s)_roll_mean', 'Flow Rate (L/s)_roll_std','Flow Rate (L/s)_lag_1', 'Flow Rate (L/s)_diff',
       'Temperature (°C)_roll_mean', 'Temperature (°C)_roll_std','Temperature (°C)_lag_1', 'Temperature (°C)_diff'
]

# 4. Define the Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives sensor data in JSON format and returns a leak prediction.
    """
    try:
        # Get the JSON data sent to the endpoint
        data = request.get_json(force=True)
        
        # We expect the data to be a dictionary with all 15 feature names:
        # {
        #   "pressure": 1.2, "flow_rate": 3.4, "temperature": 25.1,
        #   "pressure_roll_mean": 1.1, "pressure_roll_std": 0.2, ... etc.
        # }
        
        # Convert the dictionary to a DataFrame in the correct column order
        features_df = pd.DataFrame([data], columns=feature_columns)
        
        # --- 1. Scale the Data ---
        # Use the *loaded* scaler to transform the new data
        features_scaled = scaler.transform(features_df)
        
        # --- 2. Make the Prediction ---
        prediction_raw = model.predict(features_scaled)
        
        # Convert from numpy.int64 to a standard Python int for JSON
        prediction = int(prediction_raw[0]) 
        
        # Map to human-readable string
        status_map = {0: "No Leak", 1: "Minor Leak", 2: "Major Leak/Burst"}
        status_text = status_map.get(prediction, "Unknown Status")
        
        response = {'prediction': prediction, 'status': status_text}
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e), 'message': 'Invalid input data'})

# 5. Run the App
if __name__ == '__main__':
    print("Starting Flask server... Access it at http://127.0.0.1:5000")
    # debug=False is better for production/simulation
    app.run(host='0.0.0.0', port=5000, debug=False)