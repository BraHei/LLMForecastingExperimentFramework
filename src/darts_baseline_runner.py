from __future__ import annotations

import json
from pathlib import Path
from typing import List
from typing import Any
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
from darts.models.forecasting.baselines import NaiveMean, NaiveSeasonal, NaiveDrift, NaiveMovingAverage


def get_dynamic_baseline(model_name: str):
    if model_name == "NaiveSeasonal":
        return lambda series_length, **kwargs: NaiveSeasonal(**kwargs)
    elif model_name == "NaiveMovingAverage":
        return lambda series_length, **kwargs: NaiveMovingAverage(input_chunk_length=max(1, int(series_length * 0.2)), **kwargs)
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
        ts_name = series["metadata"]["dataset_name"]
        ts_data = series["series"]
        ts_seasonality = series["metadata"].get("seasonality", None)

        # --- split data ----------------------------------------------
        if (self.cfg.input_data_length is not None):
            ts_data_split = ts_data[:self.cfg.input_data_length]
        else:
            ts_data_split = split_data(ts_data, self.cfg.input_data_factor)
        
        # --- Preperation ---------------------------------------------
        forecast_len = len(ts_data) - len(ts_data_split)
        series_train = TimeSeries.from_values(np.array(ts_data_split))

        if self.model_name == "NaiveSeasonal":
            self.model = self.model_builder(len(ts_data_split), K=ts_seasonality, **self.cfg.model_parameters)
        else:
            self.model = self.model_builder(len(ts_data_split), **self.cfg.model_parameters)
        
        self.model.fit(series_train)

        # --- Prediction -----------------------------------------------
        prediction = self.model.predict(forecast_len)

        # --- Post Processing ------------------------------------------
        predicted_vals = prediction.values().flatten().tolist()
        true_vals = ts_data[-forecast_len:]
        min_len = min(len(predicted_vals), len(true_vals))
        predicted_vals = predicted_vals[:min_len]
        true_vals = true_vals[:min_len]
        
        analysis_result: dict = {}
        for a in self.analyzers:
            if ts_seasonality:
                analysis = a.Analyze(true_vals, predicted_vals, ts_data_split, ts_seasonality)
                analysis_result[a.AnalyzerType] = analysis
            else:
                analysis_result[a.AnalyzerType] = a.Analyze(true_vals, predicted_vals)

        return {
            "id": f"{ts_name}",
            "inverse_success": True,
            "data": {
                "original": safe_to_list(ts_data),
                "original_split": safe_to_list(ts_data_split),
                "reconstructed_split": safe_to_list(ts_data_split),
                "predicted": safe_to_list(predicted_vals),
            },
            "model": {
                "model_name": self.model_name,
            },
            "metrics": analysis_result,
        }

class BaselineExperimentRunner:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.output_dir)
        self.jsonl_filename = cfg.extra.get("output_jsonl", "results.jsonl")
        self.recorder = ResultRecorder(self.out_dir, self.jsonl_filename)

    def run(self) -> List:
        dataset = build(self.cfg.dataset_name, DATASET_REGISTRY, **self.cfg.dataset_params)
        series_list = list(dataset.load())

        processor = DartsSeriesProcessor(cfg=self.cfg)
        results = []

        for series in series_list:
            outcome = processor(series)

            plot_path = plot_series(
                series,
                [],
                outcome["data"]["predicted"],
                outcome["inverse_success"],
                str(self.out_dir),
                prediction_offset=len(outcome["data"]["original_split"]),
            )
            outcome["plot_path"] = plot_path
            self.recorder.record_jsonl(outcome)
            results.append(outcome)

        fix_output_ownership(self.out_dir)
        print(f"All baselines completed. Results saved in {self.out_dir}")
        return results

def run_multirun(cfg_path: str | Path):
    """Run Darts baseline experiments for each model individually, with separate output dirs."""
    base_cfg = load_config(cfg_path, build_experiment_name_flag=False)

    recorder = ResultRecorder(Path(base_cfg.output_dir), base_cfg.extra.get("output_jsonl", "results.jsonl"))

    for model_name in base_cfg.model_name:
        sub_cfg = deepcopy(base_cfg)
        sub_cfg.model_name = model_name  # Set single model
        sub_cfg.build_experiment_name_flag = True
        sub_cfg.__post_init__()  # Refresh the output_dir and experiment_name

        print(f"Running model: {model_name} → Output: {sub_cfg.output_dir}")
        sub_cfg.save()

        runner = BaselineExperimentRunner(sub_cfg)
        
        # We record a master file with relevant information for later extraction
        recorder.record_results_to_table(runner.run(), sub_cfg)
        import gc
        gc.collect()

    fix_output_ownership(Path(base_cfg.output_dir))

if __name__ == "__main__":
    import argparse

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


    