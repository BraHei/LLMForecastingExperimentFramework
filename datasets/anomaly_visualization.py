import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Ask for file input
file_path = input("Enter the full path to your time series file: ").strip()

if not os.path.isfile(file_path):
    print("Invalid file path.")
    exit()

# Parse metadata from the filename
filename = os.path.basename(file_path)
# Ensure it's a .txt file
if not filename.endswith(".txt"):
    print("This script expects a .txt file, not an image or other type.")
    exit()

# Extract metadata from filename
match = re.search(r'_(\d+)_(\d+)_(\d+)\.txt$', filename)

if not match:
    print("Filename format incorrect or missing metadata. Expected format like ..._1500_2764_2995.txt")
    exit()

train_end = int(match.group(1))
anomaly_start = int(match.group(2))
anomaly_end = int(match.group(3))

# Load the time series data
try:
    data = np.loadtxt(file_path)

    # Sanity check for bounds
    if anomaly_end >= len(data):
        print("Anomaly end index exceeds data length.")
        exit()

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(data, label='Time Series', color='gray')

    # Highlight training data
    plt.axvspan(0, train_end, color='green', alpha=0.1, label='Training Data')

    # Highlight anomaly
    plt.axvspan(anomaly_start, anomaly_end, color='red', alpha=0.3, label='Anomaly')

    plt.title(f'Time Series with Anomaly: {filename}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Save plot
    save_path = os.path.splitext(file_path)[0] + '_annotated_plot.png'
    plt.savefig(save_path)
    plt.show()

    print(f"Plot saved to {save_path}")

except Exception as e:
    print(f"Error loading or plotting data: {e}")

