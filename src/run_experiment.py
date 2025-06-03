from __future__ import annotations

import time
from pathlib import Path
from typing import List

from src.config import ExperimentConfig, load_config
from src.experiment_utils import (
    fix_output_ownership,
    plot_series,
    safe_to_list,
    build,
    inverse_transform_safe,
    split_data,
    ResultRecorder
)
from src.available_datasets import DATASET_REGISTRY, BaseDataset
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
    def __call__(self, series : BaseDataset) -> dict:
        
        ts_name = series["metadata"]["dataset_name"]
        ts_data = series["series"]
        ts_seasonality = series["metadata"].get("seasonality", None)

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
            instr_text = self.cfg.instruction_object.get("text", "")
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
        generated_list = self.model.generate_response(data_string)
        latency = time.perf_counter() - start

        # --- decode predictions --------------------------------------
        prediction_results = []
        for generated in generated_list:
            predicted, pred_success = inverse_transform_safe(self.preprocessor, generated)

            # --- metrics ----------------------------------------------
            analysis_result: dict = {}
            if pred_success:
                true_seg = ts_data[len(ts_data_split) : len(ts_data_split) + len(predicted)]
                min_len = min(len(true_seg), len(predicted))
                true_seg = true_seg[:min_len]
                predicted_trimmed = predicted[:min_len]
                for a in self.analyzers:
                    if ts_seasonality:
                        analysis = a.Analyze(true_seg, predicted_trimmed, ts_data_split, ts_seasonality)
                        analysis_result[a.AnalyzerType] = analysis
                    else:
                        analysis_result[a.AnalyzerType] = a.Analyze(true_seg, predicted_trimmed)
            else:
                analysis_result["Malformed output"] = 0.0

            prediction_results.append({
                "generated_string": generated,
                "inverse_success": pred_success,
                "predicted": safe_to_list(predicted),
                "metrics": analysis_result,
            })

        # --- assemble return dict -------------------------------------
        return {
            "id": ts_name,
            "latency": {
                "total_seconds": latency,
                "num_predictions": len(generated_list)
            },
            "data": {
                "original": safe_to_list(ts_data),
                "original_split": safe_to_list(ts_data_split),
                "reconstructed_split": safe_to_list(reconstructed),
            },
            "predictions": prediction_results,
            "model": {
                "original_string": data_string,
            }
        }

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
    def run(self) -> List:
        # --- dataset ------------------------------------------------
        dataset = build(
            self.cfg.dataset_name,
            DATASET_REGISTRY,
            **self.cfg.dataset_params
        )
        series_iter = list(dataset.load())
        results = []
        for series in series_iter:

            outcome = self.processor(series)

            # plot ----------------------------------------------------
            predictions = [p["predicted"] for p in outcome["predictions"]]
            successes = [p["inverse_success"] for p in outcome["predictions"]]
            ts_plot_path = plot_series(
                series,
                outcome["data"]["reconstructed_split"],
                predictions,
                successes,
                str(self.out_dir),
                prediction_offset=len(outcome["data"]["original_split"]),
            )

            outcome["plot_path"] = ts_plot_path
            self.recorder.record_jsonl(outcome)
            results.append(outcome)

        fix_output_ownership(self.out_dir)
        print(f"Experiment '{self.name}' finished. Results in {self.out_dir}")
        return results


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

    recorder = ResultRecorder(Path(cfg.output_dir), "")
    recorder.record_results_to_table(ExperimentRunner(cfg).run(), cfg)
