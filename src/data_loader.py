import os
import requests
import zipfile
import scipy.io
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
NASA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
BATTERY_IDS = ['B0005', 'B0006', 'B0007', 'B0018']

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    zip_path = os.path.join(DATA_DIR, 'battery_data_set.zip')
    
    # Check if files already exist to avoid re-downloading
    if all([os.path.exists(os.path.join(DATA_DIR, f"{bid}.mat")) for bid in BATTERY_IDS]):
        print("Data already exists.")
        return

    print(f"Downloading data from {NASA_URL}...")
    try:
        # Download outer zip
        if not os.path.exists(zip_path):
            response = requests.get(NASA_URL, stream=True)
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete. Extracting...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            
        # The data is inside a nested zip: "5. Battery Data Set/1. BatteryAgingARC-FY08Q4.zip"
        nested_dir = os.path.join(DATA_DIR, "5. Battery Data Set")
        inner_zip_name = "1. BatteryAgingARC-FY08Q4.zip"
        inner_zip_path = os.path.join(nested_dir, inner_zip_name)
        
        if os.path.exists(inner_zip_path):
            print(f"Extracting inner zip: {inner_zip_name}...")
            with zipfile.ZipFile(inner_zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
        else:
            print(f"Warning: Inner zip {inner_zip_path} not found.")

        print("Extraction complete.")
    except Exception as e:
        print(f"Error downloading/extracting data: {e}")

def load_battery_data(battery_id):
    mat_file = os.path.join(DATA_DIR, f"{battery_id}.mat")
    data = scipy.io.loadmat(mat_file)
    
    # The data structure is nested: data[battery_id]['cycle'][0][0][0]...
    # We need to iterate through cycles
    cycles = data[battery_id]['cycle'][0][0][0]
    
    parsed_data = []
    
    for i, cycle in enumerate(cycles):
        cycle_type = cycle['type'][0]
        ambient_temp = cycle['ambient_temperature'][0][0]
        time_str = cycle['time'][0]
        
        # We are primarily interested in discharge cycles for SOH (Capacity)
        # But we might want charge data for features later.
        # For now, let's extract everything but focus on 'discharge' for capacity.
        
        if cycle_type == 'discharge':
            data_struct = cycle['data']
            capacity = data_struct['Capacity'][0][0][0][0] if data_struct['Capacity'][0][0].size > 0 else np.nan
            
            # Detailed measurements (Voltage, Current, etc. over time for this cycle)
            # These are arrays. For summary dataframe, we might just want capacity.
            # For deep learning, we will need the full arrays.
            
            # Extract scalar features from the curves for ML models
            voltage = data_struct['Voltage_load'][0][0].flatten()
            current = data_struct['Current_load'][0][0].flatten()
            temp = data_struct['Temperature_measured'][0][0].flatten()
            time = data_struct['Time'][0][0].flatten()
            
            # Calculate features
            avg_voltage = np.mean(voltage) if len(voltage) > 0 else 0
            min_voltage = np.min(voltage) if len(voltage) > 0 else 0
            max_temp = np.max(temp) if len(temp) > 0 else 0
            discharge_time = time[-1] - time[0] if len(time) > 0 else 0
            
            parsed_data.append({
                'battery_id': battery_id,
                'cycle': i + 1,
                'type': cycle_type,
                'ambient_temperature': ambient_temp,
                'capacity': capacity,
                'avg_voltage': avg_voltage,
                'min_voltage': min_voltage,
                'max_temp': max_temp,
                'discharge_time': discharge_time,
                # Retain arrays for Deep Learning
                'voltage_load': voltage,
                'current_load': current,
                'temperature_measured': temp,
                'time': time
            })
        elif cycle_type == 'charge':
             # Similar extraction for charge if needed
             pass
             
    return pd.DataFrame(parsed_data)

def get_all_data():
    download_data()
    all_dfs = []
    for bid in BATTERY_IDS:
        print(f"Processing {bid}...")
        df = load_battery_data(bid)
        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True)

if __name__ == "__main__":
    df = get_all_data()
    print(f"Loaded {len(df)} discharge cycles.")
    print(df.head())
    
    # Save a processed summary CSV for easy inspection
    # We now include the new features
    summary_cols = ['battery_id', 'cycle', 'type', 'ambient_temperature', 'capacity', 
                    'avg_voltage', 'min_voltage', 'max_temp', 'discharge_time']
    df[summary_cols].to_csv(os.path.join(DATA_DIR, 'processed_summary.csv'), index=False)
    print("Saved processed_summary.csv")
