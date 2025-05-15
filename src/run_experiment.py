from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, List

from src.config import ExperimentConfig, load_config
from src.experiment_utils import (
    fix_output_ownership,
    plot_series,
    safe_to_list,
    build,
    inverse_transform_safe,
    split_data
)
from src.available_datasets import DATASET_REGISTRY
from src.preprocessor import PREPROCESSOR_REGISTRY
from src.lmwrapper import MODEL_REGISTRY
from src.data_analyzers import DATA_ANALYZER_REGISTRY


# ---------------------------------------------------------------------
class SeriesProcessor:
    """Turns one time-series into (recon, pred, metrics)."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.preprocessor = build(
            cfg.preprocessor_name,
            PREPROCESSOR_REGISTRY,
            **cfg.preprocessor_params
        )
        self.model = build(
            cfg.model_name,
            MODEL_REGISTRY,
            **cfg.model_parameters
        )
        self.analyzers = [build(name, DATA_ANALYZER_REGISTRY) for name in cfg.data_analyzers]

    # --------------------------------------------------------------
    def __call__(self, ts_name: str, ts_data: List[float]) -> dict:
        # --- split data ---------------------------------------------
        if self.cfg.input_data_length is not None:
            ts_data_split = ts_data[: self.cfg.input_data_length]
        else:
            ts_data_split = split_data(ts_data, self.cfg.input_data_factor)

        # --- encode --------------------------------------------------
        data_string = self.preprocessor.encode(ts_data_split)

        # --- reconstruct ---------------------------------------------
        reconstructed, _ = inverse_transform_safe(self.preprocessor, data_string)

        # --- prepend instruction ------------------------------------
        if self.cfg.instruction_object:
            first = self.cfg.instruction_object[0]
            instr_text = first.get("text", "")
            try:
                formatted = instr_text.format(
                    timeseries_name=ts_name,
                    input_data=data_string,
                    input_length=len(ts_data_split),
                    total_length=len(ts_data),
                    forecast_lenght=len(ts_data) - len(ts_data_split)
                )
            except KeyError:
                formatted = instr_text + data_string
            data_string = formatted

        # --- LLM interaction -----------------------------------------
        start = time.perf_counter()
        generated = self.model.generate_response(data_string)
        latency = time.perf_counter() - start

        # --- decode prediction ---------------------------------------
        predicted, pred_success = inverse_transform_safe(self.preprocessor, generated)

        # --- metrics --------------------------------------------------
        analysis_result: dict = {}
        if pred_success:
            true_seg = ts_data[len(ts_data_split) : len(ts_data_split) + len(predicted)]
            min_len = min(len(true_seg), len(predicted))
            true_seg = true_seg[:min_len]
            predicted = predicted[:min_len]
            for a in self.analyzers:
                analysis_result[a.AnalyzerType] = a.Analyze(true_seg, predicted)
        else:
            analysis_result["Malformed output"] = 0.0

        return {
            "id": ts_name,
            "inverse_success": pred_success,
            "data": {
                "original": safe_to_list(ts_data),
                "original_split": safe_to_list(ts_data_split),
                "reconstructed_split": safe_to_list(reconstructed),
                "predicted": safe_to_list(predicted),
            },
            "model": {
                "original_string": data_string,
                "generated": generated,
                "latency": latency,
            },
            "metrics": analysis_result,
        }


# ---------------------------------------------------------------------
class ResultRecorder:
    def __init__(self, out_dir: Path, jsonl_file: str):
        self.out_dir = out_dir
        self.jsonl_path = self.out_dir / jsonl_file
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def record_jsonl(self, result: dict) -> None:
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        fix_output_ownership(self.out_dir)


# ---------------------------------------------------------------------
class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.name = cfg.experiment_name
        self.out_dir = Path(cfg.output_dir)
        self.processor = SeriesProcessor(cfg)
        self.jsonl_filename = cfg.extra.get("output_jsonl", "results.jsonl")
        self.recorder = ResultRecorder(self.out_dir, self.jsonl_filename)

    # --------------------------------------------------------------
    def run(self) -> None:
        # --- dataset ------------------------------------------------
        dataset = build(
            self.cfg.dataset_name,
            DATASET_REGISTRY,
            **self.cfg.dataset_params
        )
        series_iter = list(dataset.load())

        for series in series_iter:
            ts_name = series["metadata"]["dataset_name"]
            ts_data = series["series"]
            outcome = self.processor(ts_name, ts_data)

            # plot ----------------------------------------------------
            ts_plot_path = plot_series(
                ts_name,
                ts_data,
                outcome["data"]["reconstructed_split"],
                outcome["data"]["predicted"],
                outcome["inverse_success"],
                str(self.out_dir),
                prediction_offset=len(outcome["data"]["original_split"]),
            )
            outcome["plot_path"] = ts_plot_path
            self.recorder.record_jsonl(outcome)

        print(f"Experiment '{self.name}' finished. Results in {self.out_dir}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a time-series LLM experiment")
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.save()
    ExperimentRunner(cfg).run()
