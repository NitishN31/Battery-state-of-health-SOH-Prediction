import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys

# Add src to path to import from other scripts
sys.path.append(os.path.dirname(__file__))

from data_loader import get_all_data
from train_lstm import BatteryDataset, LSTMModel, SEQUENCE_LENGTH

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'notebooks')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def compare_models(test_battery='B0005'):
    print(f"--- Starting Model Comparison on {test_battery} ---")
    
    # 1. Load Data
    print("Loading data...")
    df = get_all_data()
    
    # 2. Prepare Data for XGBoost
    print("Training XGBoost...")
    features = ['cycle', 'avg_voltage', 'min_voltage', 'max_temp', 'discharge_time', 'ambient_temperature']
    target = 'capacity'
    
    train_df = df[df['battery_id'] != test_battery]
    test_df = df[df['battery_id'] == test_battery]
    
    X_train_xgb = train_df[features]
    y_train_xgb = train_df[target]
    X_test_xgb = test_df[features]
    y_test_xgb = test_df[target]
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    xgb_model.fit(X_train_xgb, y_train_xgb)
    y_pred_xgb = xgb_model.predict(X_test_xgb)
    
    rmse_xgb = np.sqrt(mean_squared_error(y_test_xgb, y_pred_xgb))
    r2_xgb = r2_score(y_test_xgb, y_pred_xgb)
    print(f"XGBoost - RMSE: {rmse_xgb:.4f}, R2: {r2_xgb:.4f}")

    # 3. Prepare Data for LSTM
    print("Training LSTM...")
    train_dataset = BatteryDataset(df, seq_len=SEQUENCE_LENGTH, mode='train', test_battery=test_battery)
    test_dataset = BatteryDataset(df, seq_len=SEQUENCE_LENGTH, mode='test', test_battery=test_battery)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Using device: {device}")
    lstm_model = LSTMModel().to(device)
    print("Model created")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    print("Optimizer created")
    
    epochs = 15
    print("Starting training loop...")
    for epoch in range(epochs):
        lstm_model.train()
        batch_idx = 0
        for X_batch, y_batch in train_loader:
            if batch_idx == 0: print(f"Epoch {epoch+1} Batch {batch_idx} started")
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = lstm_model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            batch_idx += 1
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    lstm_model.eval()
    y_pred_lstm = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = lstm_model(X_batch)
            y_pred_lstm.extend(outputs.cpu().numpy().flatten())
            
    y_pred_lstm = np.array(y_pred_lstm)
    # Actuals for LSTM (might differ slightly in length if dataset class filters short cycles, 
    # but typically should match test_df if no filtering happened. 
    # BatteryDataset filters len(time) < 10. Let's assume it's consistent for now or use test_dataset.y)
    y_test_lstm = test_dataset.y
    
    rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm))
    r2_lstm = r2_score(y_test_lstm, y_pred_lstm)
    print(f"LSTM - RMSE: {rmse_lstm:.4f}, R2: {r2_lstm:.4f}")

    # 4. Plotting
    NOMINAL_CAPACITY = 2.0
    
    # Align cycles for plotting
    # XGBoost uses test_df['cycle']
    # LSTM uses test_dataset.data['cycle']
    # They should be identical if filtering is consistent.
    
    cycles = test_df['cycle'].values
    # Ensure lengths match (in case LSTM filtered something)
    if len(y_pred_lstm) != len(cycles):
        print(f"Warning: Length mismatch. XGB: {len(cycles)}, LSTM: {len(y_pred_lstm)}")
        # Fallback to using LSTM's internal cycles
        cycles = test_dataset.data['cycle'].values
        # And slice XGB to match if needed, but usually LSTM is the one dropping.
        # For this plot, let's just plot what we have.
    
    plt.figure(figsize=(12, 6))
    plt.plot(cycles, y_test_xgb / NOMINAL_CAPACITY, label='Actual SOH', color='black', linewidth=2)
    plt.plot(cycles, y_pred_xgb / NOMINAL_CAPACITY, label=f'XGBoost (R2={r2_xgb:.2f})', color='blue', linestyle='--')
    plt.plot(cycles, y_pred_lstm / NOMINAL_CAPACITY, label=f'LSTM (R2={r2_lstm:.2f})', color='green', linestyle='-.')
    
    plt.title(f'Model Comparison: SOH Estimation for Battery {test_battery}')
    plt.xlabel('Cycle Number')
    plt.ylabel('State of Health (SOH)')
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(PLOTS_DIR, f'comparison_{test_battery}.png')
    plt.savefig(output_path)
    print(f"Saved comparison plot to {output_path}")

if __name__ == "__main__":
    compare_models('B0018')
