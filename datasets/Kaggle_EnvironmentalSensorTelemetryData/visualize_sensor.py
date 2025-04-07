import os
import pyarrow.ipc as ipc
import pyarrow as pa
import pandas as pd
import matplotlib.pyplot as plt

# Settings
arrow_dir = "arrow_by_device"

# Step 1: List available devices
device_files = [f for f in os.listdir(arrow_dir) if f.endswith(".arrow")]
device_ids = [f.replace(".arrow", "") for f in device_files]

if not device_ids:
    raise FileNotFoundError("No Arrow files found in 'arrow_by_device/'")

print("Available device IDs:")
for i, device_id in enumerate(device_ids):
    print(f"{i}: {device_id}")

# Step 2: User selects device
selection = input(f"Select a device (0 - {len(device_ids)-1}): ")
try:
    selected_index = int(selection)
    selected_device = device_ids[selected_index]
except (ValueError, IndexError):
    raise ValueError("Invalid selection.")

# Step 3: Load the Arrow file for selected device
file_path = os.path.join(arrow_dir, f"{selected_device}.arrow")
with pa.memory_map(file_path, 'r') as source:
    reader = ipc.RecordBatchFileReader(source)
    table = reader.read_all()

df = table.to_pandas()

# Step 4: Check timestamp and convert if needed
if 'ts' in df.columns:
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)
else:
    print("No timestamp column 'ts' found. Using index for x-axis.")

# Step 5: Select numeric columns to plot
sensor_columns = df.select_dtypes(include='number').columns.tolist()
if not sensor_columns:
    raise ValueError("No numeric sensor columns found.")

# Step 6: Plot
df[sensor_columns].plot(subplots=True, figsize=(12, len(sensor_columns) * 3), title=selected_device)
plt.tight_layout()
plt.show()

