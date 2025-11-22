import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os
from scipy.interpolate import interp1d
from data_loader import get_all_data

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'notebooks')
SEQUENCE_LENGTH = 200 # Resample curves to this length

class BatteryDataset(Dataset):
    def __init__(self, df, seq_len=200, mode='train', test_battery='B0018'):
        self.seq_len = seq_len
        
        # Filter for discharge cycles only
        self.data = df[df['type'] == 'discharge'].reset_index(drop=True)
        
        # Split based on battery_id
        if mode == 'train':
            self.data = self.data[self.data['battery_id'] != test_battery]
        else:
            self.data = self.data[self.data['battery_id'] == test_battery]
            
        # Preprocess data
        self.X, self.y = self._process_data()
        
    def _process_data(self):
        X_list = []
        y_list = []
        
        for idx, row in self.data.iterrows():
            if idx % 50 == 0:
                print(f"Processing cycle {idx}/{len(self.data)}")
            # Extract arrays
            # Note: In the dataframe returned by get_all_data, these are numpy arrays
            voltage = row['voltage_load']
            current = row['current_load']
            temp = row['temperature_measured']
            time = row['time']
            
            # Skip empty or too short cycles
            if len(time) < 10:
                continue
                
            # Resample to fixed length
            # We use linear interpolation
            new_time = np.linspace(time[0], time[-1], self.seq_len)
            
            f_v = interp1d(time, voltage, kind='linear', fill_value="extrapolate")
            f_c = interp1d(time, current, kind='linear', fill_value="extrapolate")
            f_t = interp1d(time, temp, kind='linear', fill_value="extrapolate")
            
            v_resampled = f_v(new_time)
            c_resampled = f_c(new_time)
            t_resampled = f_t(new_time)
            
            # Stack features: (Seq_Len, 3) -> Voltage, Current, Temp
            features = np.stack([v_resampled, c_resampled, t_resampled], axis=1)
            
            # Normalize (Simple Min-Max or Standardization could be better, 
            # but for now let's do basic scaling based on typical Li-ion values)
            # Voltage: 2.5-4.2V -> Map to 0-1 approx
            features[:, 0] = (features[:, 0] - 2.5) / (4.2 - 2.5)
            # Current: -2A to 0A (discharge) -> Map to 0-1
            features[:, 1] = (features[:, 1] + 2.0) / 2.0 
            # Temp: 20-50C -> Map to 0-1
            features[:, 2] = (features[:, 2] - 20) / 30
            
            X_list.append(features)
            y_list.append(row['capacity'])
            
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train_lstm():
    print("Loading data...")
    # We need to call get_all_data to get the dataframes with arrays
    # Note: data_loader.py logic needs to be importable without running main
    df = get_all_data()
    
    train_dataset = BatteryDataset(df, seq_len=SEQUENCE_LENGTH, mode='train')
    test_dataset = BatteryDataset(df, seq_len=SEQUENCE_LENGTH, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    epochs = 100
    train_losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
    # Evaluation
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy())
            
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    print(f"LSTM Test RMSE: {rmse:.4f}")
    print(f"LSTM Test R2: {r2:.4f}")
    
    # Plotting
    NOMINAL_CAPACITY = 2.0
    plt.figure(figsize=(12, 6))
    cycles = test_dataset.data['cycle'].values
    
    plt.plot(cycles, actuals / NOMINAL_CAPACITY, label='Actual SOH', color='blue')
    plt.plot(cycles, predictions / NOMINAL_CAPACITY, label='Predicted SOH (LSTM)', color='green', linestyle='--')
    plt.title('SOH Estimation using LSTM (Raw Curves)')
    plt.xlabel('Cycle Number')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, 'lstm_prediction_B0018.png'))
    print("Saved LSTM prediction plot")

if __name__ == "__main__":
    train_lstm()
