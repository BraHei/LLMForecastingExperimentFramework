import pandas as pd
import matplotlib.pyplot as plt
import os

# Prompt for the directory
data_dir = input("Enter the directory path containing df_y.csv and df_x.csv: ").strip()

# Paths to the files
df_y_path = os.path.join(data_dir, 'df_y.csv')
df_x_path = os.path.join(data_dir, 'df_x.csv')

# Check existence
if not os.path.isfile(df_y_path):
    print(f"Missing df_y.csv in {data_dir}")
    exit()

# Load target series (df_y)
df_y = pd.read_csv(df_y_path)
df_y['ds'] = pd.to_datetime(df_y['ds'])

# Plot target series
plt.figure(figsize=(14, 5))
plt.plot(df_y['ds'], df_y['y'], label='Target Series (y)')
plt.title(f"Long-Horizon Time Series (from {os.path.basename(data_dir)})")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.legend()

# Save plot
save_path = os.path.join(data_dir, "long_horizon_plot.png")
plt.savefig(save_path)
plt.show()

print(f"\nPlot saved to: {save_path}")

