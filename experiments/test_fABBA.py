import os
import time
import json
from pathlib import Path
from src.datasets import get_dataset
from src.pretokenizer import get_pretokenizer
from src.lmwrapper import get_model
from experiment_utils import (
    save_experiment_settings,
    inverse_transform_safe,
    plot_series,
    fix_output_ownership
)

# === Settings ===
EXPERIMENT_NAME = "test_framework"
CHECKPOINT_NAME = "smollm2-1.7b"
OUTPUT_JSONL = "model_responses.jsonl"
NUM_SERIES = 10
MAX_KERNELS = 5
SEQUENCE_LENGTH = 4096
TOKENIZER_NAME = "fABBA"
TOLERANCE = 0.1
ALPHA = 0.1
SORTING = '2-norm'
SCL = 1

timestamp = time.strftime('%Y%m%d-%H%M%S')
OUTPUT_FOLDER = f"{EXPERIMENT_NAME}_{timestamp}"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# === Load components ===
dataset = get_dataset("kernelsynth", num_series=NUM_SERIES, max_kernels=MAX_KERNELS, sequence_lenght=SEQUENCE_LENGTH)
ts_list = dataset.load()

tokenizer = get_pretokenizer(TOKENIZER_NAME, tol=TOLERANCE, alpha=ALPHA, sorting=SORTING, scl=SCL, verbose=0)
model = get_model(CHECKPOINT_NAME, max_new_tokens=100, temperature=1.0, top_p=0.9)

def main():
    results = []
    jsonl_path = os.path.join(OUTPUT_FOLDER, OUTPUT_JSONL)
    save_experiment_settings(OUTPUT_FOLDER, model, tokenizer, dataset)

    for idx, ts in enumerate(ts_list):
        ts_data = ts["target"]
        data_string = tokenizer.encode(ts_data)
        model_response = model.generate_response(data_string)

        reconstructed, rec_success = inverse_transform_safe(tokenizer, data_string, ts_data[0])
        predicted, pred_success = inverse_transform_safe(tokenizer, model_response, ts_data[-1])

        plot_path = plot_series(idx, ts_data, reconstructed or ts_data, predicted, pred_success, OUTPUT_FOLDER)

        result = {
            "id": idx,
            "original_string": data_string,
            "model_response": model_response,
            "inverse_success": pred_success,
            "plot_path": plot_path
        }

        results.append(result)
        with open(jsonl_path, mode="a") as f:
            f.write(json.dumps(result) + "\n")

    fix_output_ownership(Path(OUTPUT_FOLDER))
    print("Experiment 1 complete.")

if __name__ == "__main__":
    main()
