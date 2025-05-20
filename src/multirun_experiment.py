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


def ensure_list(x):
    return x if isinstance(x, list) else [x]

def run_grid(cfg_path: str | Path):
    base_cfg = load_config(cfg_path, build_experiment_name_flag=False)
    recorder = ResultRecorder(Path(base_cfg.output_dir), base_cfg.extra.get("output_jsonl", "results.jsonl"))

    # pre‐compute instruction‐object sweep options:
    if base_cfg.instruction_object is None:
        instr_opts: list[Optional[List[dict]]] = [None]
    else:
        # each run gets a list containing exactly one instruction dict
        instr_opts = [[inst] for inst in base_cfg.instruction_object]

    sweep_axes = {
        "preprocessor_params": ensure_list(base_cfg.preprocessor_params),
        "model_name":          ensure_list(base_cfg.model_name),
        "model_parameters":    ensure_list(base_cfg.model_parameters),
        "instruction_object":  instr_opts,
        "input_data_factor":   ensure_list(base_cfg.input_data_factor),
    }

    # now flip the flag on so that __post_init__ will build a new name per sub‐cfg
    base_cfg.build_experiment_name_flag = True

    for run_idx, values in enumerate(product(*sweep_axes.values()), start=1):
        sub_cfg = deepcopy(base_cfg)
        # assign each axis value to the config
        for key, val in zip(sweep_axes.keys(), values):
            setattr(sub_cfg, key, val)

        # regenerate name & output_dir
        sub_cfg.__post_init__()
        print(f"[{run_idx}] → {sub_cfg.output_dir}")

        # save a copy of the yaml & run
        sub_cfg.save()
        runner = ExperimentRunner(sub_cfg)

        # We record a master file with relevant information for later extraction
        recorder.record_results_to_table(runner.run(), sub_cfg)

        # cleanup
        del runner.processor.model
        torch.cuda.empty_cache()
        gc.collect()

    fix_output_ownership(base_cfg.out_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments in sequence from a single config file."
    )
    parser.add_argument("--config", required=True, help="Path to batch experiment YAML.")
    args = parser.parse_args()

    run_grid(args.config)


if __name__ == "__main__":
    main()
