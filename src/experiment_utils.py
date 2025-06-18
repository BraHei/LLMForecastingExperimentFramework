import os
import json
import matplotlib
matplotlib.use("Agg")  # Headless mode for environments without GUI
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from typing import TypeVar, Type

from src.config import ExperimentConfig

T = TypeVar('T')

# --- Utility Functions ---

def build(name: str, registry: dict[str, type[T]], **kwargs) -> T:
    """
    Instantiate an object from a registry by name.
    Raises ValueError if name not found.
    """
    try:
        return registry[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown {name=}. Available: {list(registry)}")

def safe_to_list(x):
    """Convert to list if 'tolist' method is available."""
    return x.tolist() if hasattr(x, "tolist") else x

def split_data(input_list, percent):
    """
    Returns the first `percent` of the input list.
    
    Args:
        input_list (list): The list to slice.
        percent (float): The fraction of the list to return (0 to 1).
    
    Returns:
        list: The sliced portion.
    """
    cutoff = int(len(input_list) * percent)
    return input_list[:cutoff]

def save_experiment_settings(output_folder, model, preprocessor, dataset, analyzers):
    """
    Save key components of the experiment setup as a JSON file in the output folder.
    Useful for auditing and reproducibility.
    """
    settings = {
        "model": {
            "checkpoint": model.checkpoint,
            "max_new_tokens": model.max_new_tokens,
            "temperature": model.temperature,
            "top_p": model.top_p,
            "context_window": model.context_window,
            "device": model.device,
            "precision": model.precision
        },
        "preprocessor": {
            "type": preprocessor.preprocessor_type,
        },
        "dataset": dataset.metadata(),
        "analyzers": [analyzer.AnalyzerType for analyzer in analyzers]
    }

    # Optionally include encoder and settings if present
    if hasattr(preprocessor, "encoder_params"):
        settings["preprocessor"]["params"] = preprocessor.encoder_params
    if hasattr(preprocessor, "settings"):
        settings["preprocessor"]["settings"] = (
            vars(preprocessor.settings) if hasattr(preprocessor.settings, "__dict__") else preprocessor.settings
        )

    settings_path = Path(output_folder) / "settings_overview.json"
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)

def inverse_transform_safe(encoder, encoded_str):
    """
    Try to decode a string using the encoder. Return success flag and value.
    """
    try:
        return encoder.decode(encoded_str), True
    except Exception as e:
        print(e)
        return None, False

# --- Plotting ---

def plot_series(series, reconstruction, predictions, successes, output_folder, prediction_offset=None):
    """
    Plot the original, reconstruction, and predictions of a single series.
    
    Args:
        series: dict with keys 'series' and 'metadata'
        reconstruction: list or array of reconstructed values
        predictions: list of prediction arrays
        successes: list of booleans indicating if prediction was successful
        output_folder: directory to save the output PNG
        prediction_offset: optional x-axis starting point for predictions
    """
    idx = series["metadata"]["dataset_name"]
    original = series["series"]

    if prediction_offset is None:
        prediction_offset = len(original)

    # Determine max x-axis length
    valid_pred_lengths = [
        len(pred) for pred, succ in zip(predictions, successes) if succ and pred is not None
    ]
    max_pred_length = max(valid_pred_lengths) if valid_pred_lengths else 0
    max_len = min(
        len(original),
        prediction_offset + max_pred_length if max_pred_length > 0 else len(reconstruction)
    )

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(original)), original, label="Original")
    if len(reconstruction) > 0:
        plt.plot(range(len(reconstruction)), reconstruction, label="Reconstruction")

    for i, (pred, succ) in enumerate(zip(predictions, successes)):
        if succ and pred is not None:
            pred_end = prediction_offset + len(pred)
            plt.plot(
                range(prediction_offset, pred_end),
                pred,
                label=f"Prediction {i+1}",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8
            )
        else:
            # Optional visual cue for failed prediction
            plt.axvline(prediction_offset, color='red', linestyle=':', alpha=0.5, label=f"Prediction {i+1} Failed")

    plt.xlabel("Sample (-)")
    plt.ylabel("Amplitude (-)")
    plt.title(str(idx))
    plt.legend()
    plt.grid(True)

    path = f"{output_folder}/plot_{idx}.png"
    plt.xlim(0, max_len)
    plt.savefig(path)
    plt.close()
    return path

# --- File permission management ---

def fix_output_ownership(folder: Path):
    """
    Change file ownership in the folder to match HOST_UID/HOST_GID.
    Useful in containerized environments (e.g., Docker).
    """
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

# --- Result Recording Class ---

class ResultRecorder:
    """
    Responsible for recording predictions, metrics, and metadata to disk.
    Supports tabular (TSV) and JSONL formats.
    """
    def __init__(self, out_dir: Path, jsonl_file: str):
        self.out_dir = out_dir
        self.jsonl_path = self.out_dir / jsonl_file
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def flatten_config(self, cfg: ExperimentConfig):
        """
        Flatten structured config into a flat dict for tabular output.
        """
        flat = {
            "experiment_name": cfg.experiment_name,
            "model_name": cfg.model_name,
            "preprocessor_name": cfg.preprocessor_name,
            "dataset_name": cfg.dataset_name,
            "instruction_name": cfg.instruction_object['name'] if cfg.instruction_object else None,
            "instruction_text": cfg.instruction_object['text'] if cfg.instruction_object else None,
            "input_data_length": cfg.input_data_length,
            "input_data_factor": cfg.input_data_factor,
        }
        flat.update(cfg.preprocessor_params)
        flat.update(cfg.model_parameters)
        return flat

    def record_results_to_table(self, results: list, cfg: ExperimentConfig, output_file="master_results.tsv"):
        """
        Save metrics in TSV format (one row per metric per prediction).
        Appends if file exists.
        """
        rows = []
        meta = self.flatten_config(cfg)

        for entry in results:
            for pred_idx, prediction in enumerate(entry.get("predictions", [])):
                for metric_name, metric_value in prediction["metrics"].items():
                    row = {
                        **meta,
                        "series_id": entry['id'],
                        "prediction_index": pred_idx,
                        "metric_name": metric_name,
                        "metric_value": metric_value
                    }
                    rows.append(row)

        df = pd.DataFrame(rows)
        output_path = self.out_dir / output_file

        if output_path.exists():
            df.to_csv(output_path, sep="\t", mode="a", index=False, header=False)
        else:
            df.to_csv(output_path, sep="\t", mode="w", index=False, header=True)

        fix_output_ownership(self.out_dir)

    def record_jsonl(self, result: dict) -> None:
        """
        Append a structured result entry to a .jsonl file.
        """
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        fix_output_ownership(self.out_dir)
