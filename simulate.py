import pandas as pd
import requests
import time
import json
import os 

print("Simulator started...")

# --- 1. Load Data ---
try:
    df = pd.read_csv('water_leak_detection.csv')
except FileNotFoundError:
    print("Error: 'water_data.csv' not found.")
    exit()

# Re-create features (Must match training exactly)
features = ['Pressure (bar)', 'Flow Rate (L/s)', 'Temperature (°C)']
window_size = 5
for col in features:
    df[f'{col}_roll_mean'] = df[col].rolling(window=window_size).mean()
    df[f'{col}_roll_std'] = df[col].rolling(window=window_size).std()
    df[f'{col}_lag_1'] = df[col].shift(1)
    df[f'{col}_diff'] = df[col].diff()
df = df.dropna().reset_index()

feature_columns = [
    'Pressure (bar)', 'Flow Rate (L/s)', 'Temperature (°C)',
       'Pressure (bar)_roll_mean', 'Pressure (bar)_roll_std','Pressure (bar)_lag_1', 'Pressure (bar)_diff',
       'Flow Rate (L/s)_roll_mean', 'Flow Rate (L/s)_roll_std','Flow Rate (L/s)_lag_1', 'Flow Rate (L/s)_diff',
       'Temperature (°C)_roll_mean', 'Temperature (°C)_roll_std','Temperature (°C)_lag_1', 'Temperature (°C)_diff'
]
API_URL = 'http://127.0.0.1:5000/predict'

# --- 2. SETUP LOG FILE (THE UPDATE YOU NEED) ---
LOG_FILE = 'prediction_log.csv'

# Delete old log file if it exists, so we start fresh
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

# Create the new file with headers
with open(LOG_FILE, 'w') as f:
    f.write('time,pressure,flow_rate,prediction_status,prediction_class\n')

# --- 3. Run the Simulation Loop ---
for i in range(100, 200): 
    try:
        row_features = df.iloc[i][feature_columns]
        payload = row_features.to_dict()

        # Get prediction from API
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # --- 4. WRITE TO LOG FILE (THE UPDATE YOU NEED) ---
        # Extract data
        timestamp = df.iloc[i].get('Timestamp', str(i))
        pressure = payload['Pressure (bar)']
        flow = payload['Flow Rate (L/s)']
        status = result['status']
        pred_class = result['prediction']
        
        # Append this row to the CSV file
        with open(LOG_FILE, 'a') as f:
            f.write(f'{timestamp},{pressure},{flow},{status},{pred_class}\n')
        
        # Print to console
        print(f"--- [Time Step {i}] ---")
        print(f"  Logged: Pressure={pressure:.2f}, Flow={flow:.2f}, Status={status}")
        
        time.sleep(2)

    except requests.exceptions.RequestException as e:
        print(f"Error: API at {API_URL} not reachable.")
        break
    except KeyboardInterrupt:
        break

print("Simulation complete.")