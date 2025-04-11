import numpy as np
import matplotlib.pyplot as plt
import os

def load_tsv_series(file_path):
    try:
        data = np.loadtxt(file_path, delimiter='\t')
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Get one file (train or test)
input_path = input("Enter path to a UCR Forecasting .tsv file (either TRAIN or TEST): ").strip()

if not os.path.isfile(input_path):
    print("File not found.")
    exit()

base_path, filename = os.path.split(input_path)

# Determine train/test pair
if '_TRAIN' in filename:
    train_file = input_path
    test_file = os.path.join(base_path, filename.replace('_TRAIN', '_TEST'))
elif '_TEST' in filename:
    test_file = input_path
    train_file = os.path.join(base_path, filename.replace('_TEST', '_TRAIN'))
else:
    print("Filename should include _TRAIN or _TEST.")
    exit()

# Load both files
train_data = load_tsv_series(train_file)
test_data = load_tsv_series(test_file)

if train_data is None or test_data is None:
    print("Failed to load one or both files.")
    exit()

# Plot a few series from train and test
plt.figure(figsize=(14, 6))

# Plot first 3 train series
for i in range(min(3, len(train_data))):
    plt.plot(train_data[i], label=f'Train #{i}', alpha=0.6)

# Plot first 3 test series
for i in range(min(3, len(test_data))):
    plt.plot(test_data[i], label=f'Test #{i}', linestyle='--')

plt.title(f"UCR Forecasting Preview: {filename.split('_')[0]}")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# Save the plot
save_path = os.path.splitext(input_path)[0].replace('_TRAIN', '').replace('_TEST', '') + '_preview_plot.png'
plt.savefig(save_path)
plt.show()

print(f"Plot saved to: {save_path}")

