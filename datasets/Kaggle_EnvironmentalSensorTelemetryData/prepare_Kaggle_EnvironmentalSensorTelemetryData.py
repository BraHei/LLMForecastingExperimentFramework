import os
import zipfile
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

# File paths
zip_path = "Kaggle_EnvironmentalSensorTelemetryData.zip"
csv_filename = "iot_telemetry_data.csv"
output_dir = "arrow_by_device"

# Step 1: Unzip the archive
print(f"Unzipping {zip_path}...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall()

# Step 2: Create output directory
os.makedirs(output_dir, exist_ok=True)

# Step 3: Load CSV
print(f"Reading CSV: {csv_filename}...")
df = pd.read_csv(csv_filename)

# Step 4: Check available columns
print("Available columns:", df.columns.tolist())

# Use the correct column if 'device_id' doesn't exist
group_col = "device_id" if "device_id" in df.columns else (
    "device" if "device" in df.columns else None
)

if not group_col:
    raise ValueError("Could not find a device ID column like 'device_id' or 'device'.")

# Step 5: Group and write Arrow files
print(f"Processing and writing Arrow files to: {output_dir}/")
for device_id, group in df.groupby(group_col):
    table = pa.Table.from_pandas(group)
    output_file = os.path.join(output_dir, f"{device_id}.arrow")
    with pa.OSFile(output_file, 'wb') as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write(table)

# Step 6: Clean up the CSV file
os.remove(csv_filename)
print(f"Removed CSV file: {csv_filename}")

print("Done. Arrow files created for each device.")

