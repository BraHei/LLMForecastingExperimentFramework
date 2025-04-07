import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fABBA import fABBA
import pyarrow.ipc as ipc
from transformers import AutoModelForCausalLM, AutoTokenizer

# Settings
CHECKPOINT = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cpu"
ARROW_PATH = "kernelsynth-data_n200_k5.arrow"
OUTPUT_CSV = "model_responses.csv"
PLOT_FOLDER = "plots"
Path(PLOT_FOLDER).mkdir(exist_ok=True)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT).to(DEVICE)

# Read all time series from Arrow file
def read_all_timeseries(file_path):
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        reader = ipc.RecordBatchFileReader(f)
        table = reader.read_all()

    if "target" in table.column_names:
        target_column = table.column("target")
        timeseries_list = [np.array(row.as_py()) for row in target_column]
        return timeseries_list
    else:
        raise ValueError("Arrow file must contain 'target' field.")

# Prompt model and return response string
def prompt_model(data_string):
    inputs = tokenizer.encode(data_string, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        temperature=1.0,
        top_p=0.9,
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

# Safe inverse_transform with error handling
def inverse_transform_safe(fabba, string, start_value):
    try:
        return fabba.inverse_transform(string, start_value), True
    except Exception as e:
        return None, False

# Plot original, reconstruction, and prediction
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

# Main processing loop
def main():
    ts_list = read_all_timeseries(ARROW_PATH)
    results = []

    for idx, ts in enumerate(ts_list):
        print(f"\nProcessing time series {idx}...")

        fabba = fABBA(tol=0.1, alpha=0.1, sorting='2-norm', scl=1, verbose=0)
        data_string = fabba.fit_transform(ts)

        # Prompt LLM with the symbolic string
        model_response = prompt_model(data_string)

        # Reconstruct original and predicted time series
        reconstructed, rec_success = inverse_transform_safe(fabba, data_string, ts[0])
        predicted, pred_success = inverse_transform_safe(fabba, model_response, ts[-1])

        plot_path = plot_series(idx, ts, reconstructed if rec_success else ts, predicted, pred_success)

        results.append({
            "id": idx,
            "original_string": data_string,
            "model_response": model_response,
            "inverse_success": pred_success,
            "plot_path": plot_path
        })

    # Save to CSV
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "original_string", "model_response", "inverse_success", "plot_path"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ All done. Output saved to {OUTPUT_CSV} and plots to '{PLOT_FOLDER}/'.")

if __name__ == "__main__":
    main()

