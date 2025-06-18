from __future__ import annotations

import json
from pathlib import Path
from typing import List, Any
from copy import deepcopy

import numpy as np

from src.config import ExperimentConfig, load_config
from src.run_experiment import ResultRecorder
from src.experiment_utils import (
    fix_output_ownership,
    plot_series,
    safe_to_list,
    split_data,
    build,
    ResultRecorder
)
from src.available_datasets import DATASET_REGISTRY, BaseDataset
from src.data_analyzers import DATA_ANALYZER_REGISTRY

from darts import TimeSeries
from darts.models.forecasting.baselines import NaiveMean, NaiveSeasonal, NaiveDrift


def get_dynamic_baseline(model_name: str):
    """
    Return a function that builds a Darts baseline model instance
    given the model name.
    """
    if model_name == "NaiveSeasonal":
        return lambda series_length, **kwargs: NaiveSeasonal(**kwargs)
    elif model_name == "NaiveMean":
        return lambda series_length, **kwargs: NaiveMean(**kwargs)
    elif model_name == "NaiveDrift":
        return lambda series_length, **kwargs: NaiveDrift(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}, did you want to use --multirun?")


class DartsSeriesProcessor:
    """Processes a time series using a Darts baseline model."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.model_name = cfg.model_name
        self.model_builder = get_dynamic_baseline(self.model_name)
        self.analyzers = [build(name, DATA_ANALYZER_REGISTRY) for name in cfg.data_analyzers]

    def __call__(self, series: BaseDataset) -> dict:
        """
        Encodes, fits, forecasts, and evaluates a time series using the configured Darts baseline.
        """
        ts_name = series["metadata"]["dataset_name"]
        ts_data = series["series"]
        ts_seasonality = series["metadata"].get("seasonality", None)

        # --- Split data into training and testing ----------------------
        if (self.cfg.input_data_length is not None):
            ts_train = ts_data[:self.cfg.input_data_length]
        else:
            ts_train = split_data(ts_data, self.cfg.input_data_factor)
        
        # --- Prepare Darts TimeSeries objects --------------------------
        forecast_len = len(ts_data) - len(ts_train)
        series_train = TimeSeries.from_values(np.array(ts_train))

        # --- Build and fit model ---------------------------------------
        if self.model_name == "NaiveSeasonal":
            self.model = self.model_builder(len(ts_train), K=ts_seasonality, **self.cfg.model_parameters)
        else:
            self.model = self.model_builder(len(ts_train), **self.cfg.model_parameters)
        self.model.fit(series_train)

        # --- Generate prediction ---------------------------------------
        predicted = self.model.predict(forecast_len)

        # --- Post-process and evaluate results -------------------------
        predicted_vals = predicted.values().flatten().tolist()
        ts_test = ts_data[-forecast_len:]
        min_len = min(len(predicted_vals), len(ts_test))
        predicted_vals = predicted_vals[:min_len]
        ts_test = ts_test[:min_len]
        
        analysis_result: dict = {}
        for a in self.analyzers:
            if ts_seasonality:
                analysis_result[a.AnalyzerType] = a.Analyze(ts_test, predicted_vals, ts_train, ts_seasonality)
            else:
                analysis_result[a.AnalyzerType] = a.Analyze(ts_test, predicted_vals)

        # --- Return structured results ---------------------------------
        prediction_results = [{
            "inverse_success": True,
            "predicted": safe_to_list(predicted_vals),
            "metrics": analysis_result,
        }]

        return {
            "id": ts_name,
            "data": {
                "original": safe_to_list(ts_data),
                "train": safe_to_list(ts_train),
                "test": safe_to_list(ts_test),
            },
            "predictions": prediction_results
        }


class BaselineExperimentRunner:
    """Runs a set of baseline model predictions and records results."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.output_dir)
        self.jsonl_filename = cfg.extra.get("output_jsonl", "results.jsonl")
        self.recorder = ResultRecorder(self.out_dir, self.jsonl_filename)

    def run(self) -> List:
        """
        Load dataset, run model on each time series, record and plot results.
        """
        dataset = build(self.cfg.dataset_name, DATASET_REGISTRY, **self.cfg.dataset_params)
        series_list = list(dataset.load())

        processor = DartsSeriesProcessor(cfg=self.cfg)
        results = []

        for series in series_list:
            outcome = processor(series)

            # --- Plot predictions --------------------------------------
            predictions = [p["predicted"] for p in outcome["predictions"]]
            successes = [p["inverse_success"] for p in outcome["predictions"]]
            plot_path = plot_series(
                series,
                [],  # No reconstruction for baselines
                predictions,
                successes,
                str(self.out_dir),
                prediction_offset=len(outcome["data"]["train"]),
            )

            outcome["plot_path"] = plot_path
            self.recorder.record_jsonl(outcome)
            results.append(outcome)

        # --- Finalize -------------------------------------------------
        fix_output_ownership(self.out_dir)
        print(f"All baselines completed. Results saved in {self.out_dir}")
        return results


def run_multirun(cfg_path: str | Path):
    """
    Run Darts baseline experiments for each model in a list (multi-model run).
    Each model gets its own config and output directory.
    """
    base_cfg = load_config(cfg_path, build_experiment_name_flag=False)
    recorder = ResultRecorder(Path(base_cfg.output_dir), base_cfg.extra.get("output_jsonl", "results.jsonl"))

    for model_name in base_cfg.model_name:
        # --- Prepare config for each model ----------------------------
        sub_cfg = deepcopy(base_cfg)
        sub_cfg.model_name = model_name
        sub_cfg.build_experiment_name_flag = True
        sub_cfg.__post_init__()  # Refresh experiment name/output path
        print(f"Running model: {model_name} → Output: {sub_cfg.output_dir}")
        sub_cfg.save()

        # --- Run experiment and record results ------------------------
        runner = BaselineExperimentRunner(sub_cfg)
        recorder.record_results_to_table(runner.run(), sub_cfg)

        # Free memory
        import gc
        gc.collect()

    fix_output_ownership(Path(base_cfg.output_dir))


# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    # --- CLI interface to run single or multi-model experiments -------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--multirun", action="store_true", help="Enable separate runs for each model")
    args = parser.parse_args()

    if args.multirun:
        run_multirun(args.config)
    else:
        cfg = load_config(args.config)
        cfg.save()
        recorder = ResultRecorder(Path(cfg.output_dir), "")
        recorder.record_results_to_table(BaselineExperimentRunner(cfg).run())
