from itertools import product
import argparse
import gc
import torch
from copy import deepcopy
from pathlib import Path
from src.config import load_config
from src.run_experiment import ExperimentRunner


def ensure_list(x):
    return x if isinstance(x, list) else [x]


def run_grid(cfg_path: str | Path):
    base = load_config(cfg_path, build_experiment_name_flag=False)

    sweep_axes = {
        "preprocessor_params": ensure_list(base.preprocessor_params),
        "model_name": ensure_list(base.model_name),
        "model_parameters": ensure_list(base.model_parameters),
        "instruction_string": ensure_list(base.instruction_string),
        "input_data_factor": ensure_list(base.input_data_factor),
    }


    base.build_experiment_name_flag = True #enable after initializing
    for run_idx, values in enumerate(product(*sweep_axes.values()), 1):
        sub_cfg = deepcopy(base)
        for k, v in zip(sweep_axes, values):
            setattr(sub_cfg, k, v)

        sub_cfg.__post_init__()  # refresh output_dir
        print(f"[{run_idx}] → {sub_cfg.output_dir}")
        sub_cfg.save()

        runner = ExperimentRunner(sub_cfg)
        runner.run()

        # ---- free GPU/CPU memory before next loop ------------
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
