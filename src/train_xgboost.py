import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'notebooks')

def train_xgboost():
    csv_path = os.path.join(DATA_DIR, 'processed_summary.csv')
    df = pd.read_csv(csv_path)
    
    # Define features and target
    # We use 'cycle' as a feature too, as it's a strong predictor of aging
    features = ['cycle', 'avg_voltage', 'min_voltage', 'max_temp', 'discharge_time', 'ambient_temperature']
    target = 'capacity' # We predict Capacity, then calculate SOH
    
    # Leave-One-Group-Out Cross-Validation
    # Test on B0018, Train on others
    test_battery = 'B0018'
    train_df = df[df['battery_id'] != test_battery]
    test_df = df[df['battery_id'] == test_battery]
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    print(f"Training on {train_df['battery_id'].unique()}")
    print(f"Testing on {test_battery}")
    
    # Initialize and train XGBoost
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Calculate SOH (Nominal Capacity = 2.0Ah)
    NOMINAL_CAPACITY = 2.0
    y_test_soh = y_test / NOMINAL_CAPACITY
    y_pred_soh = y_pred / NOMINAL_CAPACITY
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['cycle'], y_test_soh, label='Actual SOH', color='blue')
    plt.plot(test_df['cycle'], y_pred_soh, label='Predicted SOH', color='red', linestyle='--')
    plt.title(f'SOH Prediction for Battery {test_battery} (XGBoost)')
    plt.xlabel('Cycle Number')
    plt.ylabel('State of Health (SOH)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f'xgboost_prediction_{test_battery}.png'))
    print(f"Saved prediction plot to {PLOTS_DIR}")
    
    # Feature Importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10)
    plt.title('XGBoost Feature Importance')
    plt.savefig(os.path.join(PLOTS_DIR, 'xgboost_feature_importance.png'))
    print("Saved feature importance plot")

if __name__ == "__main__":
    train_xgboost()
