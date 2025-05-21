from itertools import product
import argparse
import gc
import torch
from copy import deepcopy
from pathlib import Path
from src.config import load_config
from src.run_experiment import ExperimentRunner
from typing import List, Optional
from src.experiment_utils import ResultRecorder, fix_output_ownership
from itertools import product
from dataclasses import asdict

def ensure_list(x):
    if x is None:
        return [None]
    return x if isinstance(x, list) else [x]

def expand_param_grid(param_grid: dict):
    """Turn {'a': [1,2], 'b': [3,4]} into [{'a':1,'b':3}, {'a':1,'b':4}, ...]"""
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    return [dict(zip(keys, v)) for v in product(*vals)]

def run_grid(cfg_path: str | Path):
    base_cfg = load_config(cfg_path, build_experiment_name_flag=False)
    recorder = ResultRecorder(Path(base_cfg.output_dir), base_cfg.extra.get("output_jsonl", "results.jsonl"))

    # --- Build sweep_axes from *_grid fields ---
    sweep_axes = {}
    base_dict = asdict(base_cfg) | base_cfg.extra  # merge base_cfg and extra

    for k, v in base_dict.items():
        if k.endswith("_grid"):
            base_key = k[:-5]  # Remove '_grid'
            if base_key == "preprocessor_params":
                # expand dict grid
                sweep_axes[base_key] = expand_param_grid(v)
            else:
                sweep_axes[base_key] = ensure_list(v)

    # For all standard fields not in sweep_axes, add as singleton
    for key in ["model_name", "model_parameters", "preprocessor_params", "instruction_object", "input_data_factor"]:
        if key not in sweep_axes:
            sweep_axes[key] = ensure_list(getattr(base_cfg, key, None))

    # Always: flip flag on for naming
    base_cfg.build_experiment_name_flag = True

    # --- Sweep over product of axes ---
    for run_idx, values in enumerate(product(*sweep_axes.values()), start=1):
        sub_cfg = deepcopy(base_cfg)
        for key, val in zip(sweep_axes.keys(), values):
            if val is not None:
                setattr(sub_cfg, key, val)
        sub_cfg.__post_init__()
        print(f"[{run_idx}] → {sub_cfg.output_dir}")
        sub_cfg.save()
        runner = ExperimentRunner(sub_cfg)
        recorder.record_results_to_table(runner.run(), sub_cfg)
        del runner.processor.model
        torch.cuda.empty_cache()
        gc.collect()

    fix_output_ownership(Path(base_cfg.output_dir))

def main():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments in sequence from a single config file."
    )
    parser.add_argument("--config", required=True, help="Path to batch experiment YAML.")
    args = parser.parse_args()

    run_grid(args.config)


if __name__ == "__main__":
    main()
