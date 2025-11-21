import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'notebooks')

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def run_eda():
    csv_path = os.path.join(DATA_DIR, 'processed_summary.csv')
    if not os.path.exists(csv_path):
        print("processed_summary.csv not found. Run data_loader.py first.")
        return

    df = pd.read_csv(csv_path)
    
    # Nominal capacity for these batteries is 2.0 Ah
    NOMINAL_CAPACITY = 2.0
    df['SOH'] = df['capacity'] / NOMINAL_CAPACITY
    
    # Plot Capacity vs Cycle
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='cycle', y='capacity', hue='battery_id', marker='o')
    plt.title('Battery Capacity Degradation over Cycles')
    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity (Ah)')
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, 'capacity_vs_cycle.png'))
    print("Saved capacity_vs_cycle.png")
    
    # Plot SOH vs Cycle
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='cycle', y='SOH', hue='battery_id', marker='o')
    plt.axhline(y=0.7, color='r', linestyle='--', label='End of Life (70%)') # Typically 70-80% is EOL
    plt.title('State of Health (SOH) Degradation over Cycles')
    plt.xlabel('Cycle Number')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, 'soh_vs_cycle.png'))
    print("Saved soh_vs_cycle.png")

if __name__ == "__main__":
    run_eda()
