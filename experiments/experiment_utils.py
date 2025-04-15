import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def split_data(input_list, percent):
    """
    Returns the first `percent` of the input list.
    
    Args:
        input_list (list): The list to slice.
        percent (float): The fraction of the list to return (between 0 and 1).
        
    Returns:
        list: A sliced portion of the input list.
    """
    cutoff = int(len(input_list) * percent)
    return input_list[:cutoff]

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

    if hasattr(tokenizer, "encoder_params"):
        settings["tokenizer"]["params"] = tokenizer.encoder_params
    if hasattr(tokenizer, "settings"):
        settings["tokenizer"]["settings"] = (
            vars(tokenizer.settings) if hasattr(tokenizer.settings, "__dict__") else tokenizer.settings
        )

    settings_path = Path(output_folder) / "settings_overview.json"
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)


def inverse_transform_safe(encoder, encoded_str, start_value=None):
    try:
        return encoder.decode(encoded_str, reference_point=start_value), True
    except Exception as e:
        print(e)
        return None, False


def plot_series(idx, original, reconstruction, prediction, success, output_folder, prediction_offset = None):
    if (prediction_offset is None):
        prediction_offset = len(original)

    plt.figure(figsize=(10, 4))
    plt.plot(original, label="Original")
    plt.plot(reconstruction, label="Reconstruction")
    if success and prediction is not None:
        plt.plot(range(prediction_offset, prediction_offset + len(prediction)), prediction, label="Prediction")
    else:
        plt.title("Prediction failed (malformed output)")
    plt.legend()
    plt.grid(True)
    path = f"{output_folder}/plot_{idx}.png"
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
