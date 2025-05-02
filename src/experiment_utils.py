import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from typing import TypeVar, Type

T = TypeVar('T')

def build(name: str, registry: dict[str, type[T]], **kwargs) -> T:
    try:
        return registry[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown {name=}. Available: {list(registry)}")

def safe_to_list(x):
    return x.tolist() if hasattr(x, "tolist") else x

def save_experiment_settings(output_folder, model, tokenizer, dataset, analyzers):
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
        "tokenizer": {
            "type": tokenizer.tokenizer_type,
        },
        "dataset": dataset.metadata(),
        "analyzers": [analyzer.AnalyzerType for analyzer in analyzers]
    }

    if hasattr(tokenizer, "encoder_params"):
        settings["tokenizer"]["params"] = tokenizer.encoder_params
    if hasattr(tokenizer, "settings"):
        settings["tokenizer"]["settings"] = (
            vars(tokenizer.settings) if hasattr(tokenizer.settings, "__dict__") else tokenizer.settings
        )

    settings_path = Path(output_folder) / "settings_overview.json"
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)


def inverse_transform_safe(encoder, encoded_str):
    try:
        return encoder.decode(encoded_str), True
    except Exception as e:
        print(e)
        return None, False


def plot_series(idx, original, reconstruction, prediction, success, output_folder, prediction_offset=None):
    plt.style.use('default')  # Use default matplotlib style

    if prediction_offset is None:
        prediction_offset = len(original)

    # Determine the x-axis length
    max_len = min(
        len(original),
        prediction_offset + len(prediction) if (success and prediction is not None) else len(reconstruction)
    )

    # Create figure and plot
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(original)), original, label="Original")
    plt.plot(range(len(reconstruction)), reconstruction, label="Reconstruction")

    if success and prediction is not None:
        pred_end = prediction_offset + len(prediction)
        plt.plot(range(prediction_offset, pred_end), prediction, label="Prediction")
    else:
        idx = f"{idx}: Prediction failed (malformed output)"

    # Add labels and title
    plt.xlabel("Sample (-)")
    plt.ylabel("Amplitude (-)")
    plt.title(str(idx))
    plt.legend()
    plt.grid(True)

    # Save and close
    path = f"{output_folder}/plot_{idx}.png"
    plt.xlim(0, max_len)
    plt.savefig(path)
    plt.close()
    return path

def fix_output_ownership(folder: Path):
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
