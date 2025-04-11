import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.datasets import get_dataset
from src.pretokenizer import get_pretokenizer
from src.lmwrapper import get_model

# === Settings ===
CHECKPOINT_NAME = "smollm2-135m"
DEVICE = "cpu"
NUM_SERIES = 10
MAX_KERNELS = 2
SEQUENCE_LENGHT = 4096
OUTPUT_CSV = "model_responses.csv"
PLOT_FOLDER = "plots"
Path(PLOT_FOLDER).mkdir(exist_ok=True)

# === Generate Synthetic Time Series ===
dataset = get_dataset("kernelsynth", num_series=NUM_SERIES, max_kernels=MAX_KERNELS, sequence_lenght=SEQUENCE_LENGHT)
ts_list = dataset.load()

# === Initialize Pretokenizer ===
tokenizer = get_pretokenizer("fABBA", tol=0.1, alpha=0.1, sorting='2-norm', scl=1, verbose=0)

# === Load LLM ===
model = get_model(CHECKPOINT_NAME, device=DEVICE, max_new_tokens=100, temperature=1.0, top_p=0.9)

# === Prompt LLM ===
def prompt_model(data_string):
    return model.generate_response(data_string)

# === Safe Inverse Transform ===
def inverse_transform_safe(encoder, encoded_str, start_value):
    try:
        return encoder.decode(encoded_str, reference_point=start_value), True
    except Exception as e:
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
    path = f"{PLOT_FOLDER}/plot_{idx}.png"
    plt.savefig(path)
    plt.close()
    return path

# === Run Experiment ===
def main():
    results = []

    for idx, ts in enumerate(ts_list):
        print(f"\nProcessing time series {idx}...")

        data_string = tokenizer.encode(ts)
        model_response = prompt_model(data_string)

        reconstructed, rec_success = inverse_transform_safe(tokenizer, data_string, ts[0])
        predicted, pred_success = inverse_transform_safe(tokenizer, model_response, ts[-1])

        plot_path = plot_series(idx, ts, reconstructed if rec_success else ts, predicted, pred_success)

        results.append({
            "id": idx,
            "original_string": data_string,
            "model_response": model_response,
            "inverse_success": pred_success,
            "plot_path": plot_path
        })

    # Save results
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "original_string", "model_response", "inverse_success", "plot_path"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAll done. Results saved to {OUTPUT_CSV}, plots in '{PLOT_FOLDER}/'.")

if __name__ == "__main__":
    main()
