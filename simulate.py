import pandas as pd
import requests
import time
import json

print("Simulator started...")

# --- 1. Load Data ---
try:
    df = pd.read_csv('water_leak_detection.csv')
    print("Original data loaded.")
except FileNotFoundError:
    print("Error: 'water_data.csv' not found. Make sure it's in the folder.")
    exit()

# --- 2. Feature Engineering (MUST be identical to training) ---
print("Re-creating features for simulation...")
features = ['Pressure (bar)', 'Flow Rate (L/s)', 'Temperature (°C)']
window_size = 5

for col in features:
    df[f'{col}_roll_mean'] = df[col].rolling(window=window_size).mean()
    df[f'{col}_roll_std'] = df[col].rolling(window=window_size).std()
    df[f'{col}_lag_1'] = df[col].shift(1)
    df[f'{col}_diff'] = df[col].diff()

# We must drop the NaNs, just like in training
df = df.dropna()
print("Features re-created. Starting simulation.")

# --- 3. Define Features to Send ---
# This is the full list of 15 features our API expects
feature_columns = [
       'Pressure (bar)', 'Flow Rate (L/s)', 'Temperature (°C)',
       'Pressure (bar)_roll_mean', 'Pressure (bar)_roll_std','Pressure (bar)_lag_1', 'Pressure (bar)_diff',
       'Flow Rate (L/s)_roll_mean', 'Flow Rate (L/s)_roll_std','Flow Rate (L/s)_lag_1', 'Flow Rate (L/s)_diff',
       'Temperature (°C)_roll_mean', 'Temperature (°C)_roll_std','Temperature (°C)_lag_1', 'Temperature (°C)_diff'
]

# This is the URL of our running Flask app's endpoint
API_URL = 'http://127.0.0.1:5000/predict'

# --- 4. Run the Simulation Loop ---
# We will simulate just the first 100 rows for this test
for i in range(150, 200):
    try:
        # Get one row of data
        row_features = df.iloc[i][feature_columns]
        
        # Convert the row (which is a pandas Series) to a standard dict
        # This is the JSON payload we will send
        payload = row_features.to_dict()

        # Send the data to the API as a POST request
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() # Raise an error for bad responses (4xx, 5xx)
        
        # Get the prediction back from the API
        result = response.json()
        
        # --- Print a nice log ---
        # Get the original sensor data for a cleaner log
        original_pressure = payload['Pressure (bar)']
        original_flow = payload['Flow Rate (L/s)']
        
        print(f"Time Step {i}")
        print(f"  Sending: Pressure={original_pressure:.2f}, Flow={original_flow:.2f}")
        print(f"  Prediction: {result['status']} (Class {result['prediction']})")
        
        # Wait for 2 seconds to simulate a real-time interval
        time.sleep(2)

    except requests.exceptions.RequestException as e:
        print(f"\nError: Could not connect to the API at {API_URL}.")
        print("Is the 'app.py' server running in another terminal?")
        print(f"Details: {e}")
        break
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        break

print("Simulation complete.")