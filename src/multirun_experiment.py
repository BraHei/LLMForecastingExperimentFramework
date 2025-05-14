from itertools import product
import argparse
import gc
import torch
from copy import deepcopy
from pathlib import Path
from src.config import load_config
from src.run_experiment import ExperimentRunner
from typing import List, Optional


def ensure_list(x):
    return x if isinstance(x, list) else [x]

def run_grid(cfg_path: str | Path):
    base = load_config(cfg_path, build_experiment_name_flag=False)

    # pre‐compute instruction‐object sweep options:
    if base.instruction_object is None:
        instr_opts: list[Optional[List[dict]]] = [None]
    else:
        # each run gets a list containing exactly one instruction dict
        instr_opts = [[inst] for inst in base.instruction_object]

    sweep_axes = {
        "preprocessor_params": ensure_list(base.preprocessor_params),
        "model_name":          ensure_list(base.model_name),
        "model_parameters":    ensure_list(base.model_parameters),
        "instruction_object":  instr_opts,
        "input_data_factor":   ensure_list(base.input_data_factor),
    }

    # now flip the flag on so that __post_init__ will build a new name per sub‐cfg
    base.build_experiment_name_flag = True

    for run_idx, values in enumerate(product(*sweep_axes.values()), start=1):
        sub_cfg = deepcopy(base)
        # assign each axis value to the config
        for key, val in zip(sweep_axes.keys(), values):
            setattr(sub_cfg, key, val)

        # regenerate name & output_dir
        sub_cfg.__post_init__()
        print(f"[{run_idx}] → {sub_cfg.output_dir}")

        # save a copy of the yaml & run
        sub_cfg.save()
        runner = ExperimentRunner(sub_cfg)
        runner.run()

        # cleanup
        del runner.processor.model
        torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments in sequence from a single config file."
    )
    parser.add_argument("--config", required=True, help="Path to batch experiment YAML.")
    args = parser.parse_args()

    run_grid(args.config)


if __name__ == "__main__":
    main()
