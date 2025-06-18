from itertools import product
import argparse
import gc
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict

from src.config import load_config
from src.run_experiment import ExperimentRunner
from src.experiment_utils import ResultRecorder, fix_output_ownership

# --- Utility Functions ---

def ensure_list(x):
    """Ensure input is a list; wrap single value or return [None] if input is None."""
    if x is None:
        return [None]
    return x if isinstance(x, list) else [x]

def expand_param_grid(param_grid: dict):
    """
    Expand a dictionary of parameter lists into a list of all combinations.
    E.g., {'a': [1,2], 'b': [3,4]} → [{'a':1,'b':3}, {'a':1,'b':4}, ...]
    """
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    return [dict(zip(keys, v)) for v in product(*vals)]

# --- Core Execution Function ---

def run_grid(cfg_path: str | Path):
    """
    Run a sweep of experiments using a single base config and any *_grid entries for parameter variation.
    Each variation results in a unique sub-config and output directory.
    """
    base_cfg = load_config(cfg_path, build_experiment_name_flag=False)

    # Initialize result recorder
    recorder = ResultRecorder(
        Path(base_cfg.output_dir),
        base_cfg.extra.get("output_jsonl", "results.jsonl")
    )

    # --- Extract sweep axes from *_grid fields ---
    sweep_axes = {}
    base_dict = asdict(base_cfg) | base_cfg.extra  # merge dataclass and extra

    for k, v in base_dict.items():
        if k.endswith("_grid") and v is not None:
            base_key = k[:-5]  # e.g., model_parameters_grid → model_parameters
            if isinstance(v, dict):
                sweep_axes[base_key] = expand_param_grid(v)
            else:
                sweep_axes[base_key] = ensure_list(v)

    # Add any remaining core fields not included in sweep axes
    for key in ["model_name", "model_parameters", "preprocessor_params", "instruction_object", "input_data_factor"]:
        if key not in sweep_axes:
            sweep_axes[key] = ensure_list(getattr(base_cfg, key, None))

    base_cfg.build_experiment_name_flag = True  # Enable naming logic

    previous_model = None  # Cache model reuse across runs
    previous_alias = ""

    # --- Execute grid sweep ---
    for run_idx, values in enumerate(product(*sweep_axes.values()), start=1):
        sub_cfg = deepcopy(base_cfg)
        
        # Set values for this grid point
        for key, val in zip(sweep_axes.keys(), values):
            if val is not None:
                setattr(sub_cfg, key, val)

        sub_cfg.__post_init__()  # Enforce config validation and naming
        print(f"[{run_idx}] → {sub_cfg.output_dir}")
        sub_cfg.save()

        # Free memory if model has changed
        if previous_alias is not sub_cfg.model_name:
            previous_model = None
            gc.collect()

        # Run experiment
        runner = ExperimentRunner(sub_cfg, previous_model)
        results = runner.run()

        # Record results
        recorder.record_results_to_table(results, sub_cfg)

        # Cache model instance if reused
        previous_alias = sub_cfg.model_name
        previous_model = runner.processor.model

# --- CLI Entrypoint ---

def main():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments in sequence from a single config file."
    )
    parser.add_argument("--config", required=True, help="Path to batch experiment YAML.")
    args = parser.parse_args()

    run_grid(args.config)

if __name__ == "__main__":
    main()
