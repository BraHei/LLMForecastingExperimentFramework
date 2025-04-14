import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os

from src.datasets import get_dataset
from src.pretokenizer import get_pretokenizer
from src.lmwrapper import get_model

timestamp = time.strftime('%Y%m%d-%H%M%S')

# === Settings ===
EXPERMIMENT_NAME = "test_framework"
OUTPUT_CSV = "model_responses.csv"
OUTPUT_FOLDER = f"{EXPERMIMENT_NAME}_{timestamp}"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

CHECKPOINT_NAME = "smollm2-135m"
DEVICE = "cpu"
NUM_SERIES = 3
MAX_KERNELS = 1
SEQUENCE_LENGHT = 4096

# === Step 1: Generate Synthetic Time Series ===
print("[Step 1] Generating synthetic time series...")
dataset = get_dataset("kernelsynth", num_series=NUM_SERIES, max_kernels=MAX_KERNELS, sequence_lenght=SEQUENCE_LENGHT)
ts_list = dataset.load()
print(f"  -> Loaded {len(ts_list)} time series.")

# === Step 2: Initialize Pretokenizer ===
print("[Step 2] Initializing pretokenizer (fABBA)...")
tokenizer = get_pretokenizer("fABBA", tol=0.1, alpha=0.1, sorting='2-norm', scl=1, verbose=0)

# === Step 3: Load LLM ===
print(f"[Step 3] Loading language model: {CHECKPOINT_NAME} on device '{DEVICE}'...")
model = get_model(CHECKPOINT_NAME, device=DEVICE, max_new_tokens=100, temperature=1.0, top_p=0.9)
print("  -> Model loaded successfully.")

# === Prompt LLM ===
def prompt_model(data_string):
    return model.generate_response(data_string)

# === Safe Inverse Transform ===
def inverse_transform_safe(encoder, encoded_str, start_value):
    try:
        return encoder.decode(encoded_str, reference_point=start_value), True
    except Exception:
        return None, False

# === Plotting ===
def plot_series(idx, original, reconstruction, prediction, success):
    plt.figure(figsize=(10, 4))
    plt.plot(original, label="Original")
    plt.plot(reconstruction, label="Reconstruction")
    if success and prediction is not None:
        plt.plot(range(len(original), len(original) + len(prediction)), prediction, label="Prediction")
    else:
        plt.title("Prediction failed (malformed output)")
    plt.legend()
    plt.grid(True)
    path = f"{OUTPUT_FOLDER}/plot_{idx}.png"
    plt.savefig(path)
    plt.close()
    return path

# === Run Experiment ===

def main():
    print("[Step 4] Starting main experiment loop...\n")
    results = []

    csv_path = os.path.join(OUTPUT_FOLDER, OUTPUT_CSV)
    # Write header once at the beginning
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "original_string", "model_response", "inverse_success", "plot_path"
        ])
        writer.writeheader()

    for idx, ts in enumerate(ts_list):
        print(f"Processing time series {idx + 1}/{len(ts_list)}...")
        ts_data = ts["target"]
        data_string = tokenizer.encode(ts_data)
        model_response = prompt_model(data_string)

        reconstructed, rec_success = inverse_transform_safe(tokenizer, data_string, ts_data[0])
        predicted, pred_success = inverse_transform_safe(tokenizer, model_response, ts_data[-1])

        plot_path = plot_series(idx, ts_data, reconstructed if rec_success else ts, predicted, pred_success)

        result = {
            "id": idx,
            "original_string": data_string,
            "model_response": model_response,
            "inverse_success": pred_success,
            "plot_path": plot_path
        }

        results.append(result)

        # Append each result to the CSV
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "original_string", "model_response", "inverse_success", "plot_path"
            ])
            writer.writerow(result)

    print(f"\nDone. Results appended to '{OUTPUT_CSV}', plots saved to '{OUTPUT_FOLDER}/'.")
    
if __name__ == "__main__":
    main()
