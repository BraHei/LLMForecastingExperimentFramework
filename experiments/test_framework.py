import csv
import json
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
OUTPUT_JSONL = "model_responses.jsonl"
OUTPUT_FOLDER = f"{EXPERMIMENT_NAME}_{timestamp}"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

CHECKPOINT_NAME = "smollm2-135m"
NUM_SERIES = 3
MAX_KERNELS = 5
SEQUENCE_LENGHT = 1024

# === Step 1: Generate Synthetic Time Series ===
print("[Step 1] Generating synthetic time series...")
dataset = get_dataset("kernelsynth", num_series=NUM_SERIES, max_kernels=MAX_KERNELS, sequence_lenght=SEQUENCE_LENGHT)
ts_list = dataset.load()
print(f"  -> Loaded {len(ts_list)} time series.")

# === Step 2: Initialize Pretokenizer ===
print("[Step 2] Initializing pretokenizer (fABBA)...")
tokenizer = get_pretokenizer("fABBA", tol=0.1, alpha=0.1, sorting='2-norm', scl=1, verbose=0)

# === Step 3: Load LLM ===
print(f"[Step 3] Loading language model: {CHECKPOINT_NAME}...")
model = get_model(CHECKPOINT_NAME, max_new_tokens=100, temperature=1.0, top_p=0.9)
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

# === Save the experiment settings ===
def save_experiment_settings(output_folder, model, tokenizer, dataset):
    settings = {
        "model": {
            "checkpoint": model.checkpoint,
            "max_new_tokens": model.max_new_tokens,
            "temperature": model.temperature,
            "top_p": model.top_p,
            "context_window": model.context_window,
            "device": model.device
        },
        "tokenizer": {
            "type": tokenizer.tokenizer_type,
        },
        "dataset": dataset.metadata()
    }

    # Add encoder-specific parameters if available
    if hasattr(tokenizer, "encoder_params"):
        settings["tokenizer"]["params"] = tokenizer.encoder_params
    if hasattr(tokenizer, "settings"):
        # Convert settings object to dict if it has __dict__
        settings["tokenizer"]["settings"] = (
            vars(tokenizer.settings) if hasattr(tokenizer.settings, "__dict__") else tokenizer.settings
        )

    settings_path = Path(output_folder) / "settings_overview.json"
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)

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

# === Fix output folder ownership ===

def fix_output_ownership(folder: Path):
    """If HOST_UID and HOST_GID are set, change ownership of OUTPUT_FOLDER recursively."""
    try:
        uid = int(os.environ.get("HOST_UID", -1))
        gid = int(os.environ.get("HOST_GID", -1))
        if uid < 0 or gid < 0:
            print("[Info] HOST_UID or HOST_GID not set; skipping ownership fix.")
            return

        print(f"[Info] Fixing ownership of {folder} to UID={uid}, GID={gid}...")
        for root, dirs, files in os.walk(folder):
            for name in dirs + files:
                path = os.path.join(root, name)
                try:
                    os.chown(path, uid, gid)
                except PermissionError as e:
                    print(f"[Warning] Could not change ownership of {path}: {e}")
        os.chown(folder, uid, gid)
    except Exception as e:
        print(f"[Warning] Ownership fix failed: {e}")

# === Run Experiment ===
def main():
    print("[Step 4] Starting main experiment loop...\n")
    results = []

    jsonl_path = os.path.join(OUTPUT_FOLDER, OUTPUT_JSONL)

    # Save settings overview
    save_experiment_settings(OUTPUT_FOLDER, model, tokenizer, dataset)

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

        # Append each result to the JSON Line
        with open(jsonl_path, mode="a") as f:
            f.write(json.dumps(result) + "\n")

    fix_output_ownership(OUTPUT_FOLDER)
    
    print(f"\nDone. Results saved to '{OUTPUT_JSONL}', plots saved to '{OUTPUT_FOLDER}/'.")
    
if __name__ == "__main__":
    main()
