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


class SeriesProcessor:
    """Turns one time-series into (recon, pred, metrics)."""

    def __init__(self, cfg: ExperimentConfig, preloaded_model = None):
        # Initialize preprocessor and model using registries and config
        self.cfg = cfg
        self.preprocessor = build(
            cfg.preprocessor_name,
            PREPROCESSOR_REGISTRY,
            **cfg.preprocessor_params
        )

        # Use preloaded model if available, else create new from config
        if (preloaded_model is None):
            self.model = build(
                cfg.model_name,
                MODEL_REGISTRY,
                **cfg.model_parameters
            )
        else:
            preloaded_model.update_parameters(**cfg.model_parameters)
            self.model = preloaded_model

        # Initialize data analyzers
        self.analyzers = [build(name, DATA_ANALYZER_REGISTRY) for name in cfg.data_analyzers]

    # --------------------------------------------------------------
    def __call__(self, series : BaseDataset) -> dict:
        """
        Processes a single time series through the entire experiment pipeline:
        encoding, LLM inference, decoding, analysis, and formatting results.
        """
        ts_name = series["metadata"]["dataset_name"]
        ts_data = series["series"]
        ts_seasonality = series["metadata"].get("seasonality", 1)

        # --- Data splitting ---------------------------------------------
        if self.cfg.input_data_length is not None:
            ts_train = ts_data[: self.cfg.input_data_length]
        else:
            ts_train = split_data(ts_data, self.cfg.input_data_factor)

        # --- Preprocessing: encode training data ------------------------
        data_string = self.preprocessor.encode(ts_train)

        # --- Optional reconstruction for inspection ---------------------
        reconstructed, _ = inverse_transform_safe(self.preprocessor, data_string)

        # --- Optional instruction prepending ----------------------------
        if self.cfg.instruction_object:
            instr_text = self.cfg.instruction_object.get("text", "")
            try:
                formatted = instr_text.format(
                    timeseries_name=ts_name,
                    input_data=data_string,
                    input_length=len(ts_train),
                    total_length=len(ts_data),
                    forecast_lenght=len(ts_data) - len(ts_train)
                )
            except KeyError:
                formatted = instr_text + data_string
            data_string = formatted

        # --- Model inference --------------------------------------------
        start = time.perf_counter()
        generated_list = self.model.generate_response(data_string)
        latency = time.perf_counter() - start

        # --- Decoding and Evaluation ------------------------------------
        prediction_results = []
        for generated in generated_list:
            predicted, pred_success = inverse_transform_safe(self.preprocessor, generated)

            # --- Analyze if decoding succeeded ---------------------------
            analysis_result: dict = {}
            if pred_success:
                ts_test = ts_data[len(ts_train) : len(ts_train) + len(predicted)]
                min_len = min(len(ts_test), len(predicted))
                ts_test = ts_test[:min_len]
                predicted_trimmed = predicted[:min_len]
                for a in self.analyzers:
                    analysis_result[a.AnalyzerType]  = a.Analyze(ts_test, predicted_trimmed, ts_train, ts_seasonality)
            else:
                ts_test = []
                analysis_result["Malformed output"] = 0.0

            # --- Append prediction results -------------------------------
            prediction_results.append({
                "generated_string": generated,
                "inverse_success": pred_success,
                "predicted": safe_to_list(predicted),
                "metrics": analysis_result,
            })

        # --- Return structured result ------------------------------------
        return {
            "id": ts_name,
            "latency": {
                "total_seconds": latency,
                "num_predictions": len(generated_list)
            },
            "data": {
                "original": safe_to_list(ts_data),
                "train": safe_to_list(ts_train),
                "reconstructed_train": safe_to_list(reconstructed),
                "test": safe_to_list(ts_test),
            },
            "predictions": prediction_results,
            "model": {
                "original_string": data_string,
            }
        }
    
class ExperimentRunner:
    """
    Manages running an experiment over a dataset and recording results.
    """

    def __init__(self, cfg: ExperimentConfig, preloaded_model = None):
        self.cfg = cfg
        self.name = cfg.experiment_name
        self.out_dir = Path(cfg.output_dir)
        self.processor = SeriesProcessor(cfg, preloaded_model)
        self.jsonl_filename = cfg.extra.get("output_jsonl", "results.jsonl")
        self.recorder = ResultRecorder(self.out_dir, self.jsonl_filename)

    def run(self) -> List:
        # --- Load dataset ---------------------------------------------
        dataset = build(
            self.cfg.dataset_name,
            DATASET_REGISTRY,
            **self.cfg.dataset_params
        )
        series_iter = list(dataset.load())
        results = []

        # --- Process each series ---------------------------------------
        for series in series_iter:
            outcome = self.processor(series)

            # --- Visualization -----------------------------------------
            predictions = [p["predicted"] for p in outcome["predictions"]]
            successes = [p["inverse_success"] for p in outcome["predictions"]]
            ts_plot_path = plot_series(
                series,
                outcome["data"]["reconstructed_train"],
                predictions,
                successes,
                str(self.out_dir),
                prediction_offset=len(outcome["data"]["train"]),
            )

            # --- Store result ------------------------------------------
            outcome["plot_path"] = ts_plot_path
            self.recorder.record_jsonl(outcome)
            results.append(outcome)

        # --- Finalize --------------------------------------------------
        fix_output_ownership(self.out_dir)
        print(f"Experiment '{self.name}' finished. Results in {self.out_dir}")
        return results


if __name__ == "__main__":
    import argparse

    # --- CLI interface to run experiment ------------------------------
    parser = argparse.ArgumentParser(description="Run a time-series LLM experiment")
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    # --- Load and persist config, then run ----------------------------
    cfg = load_config(args.config)
    cfg.save()

    recorder = ResultRecorder(Path(cfg.output_dir), "")
    recorder.record_results_to_table(ExperimentRunner(cfg).run(), cfg)
