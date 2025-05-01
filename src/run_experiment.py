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
    inverse_transform_safe
)
from src.available_datasets import DATASET_REGISTRY
from src.pretokenizer import PRETOKENIZER_REGISTRY
from src.lmwrapper import MODEL_REGISTRY
from src.data_analyzers import DATA_ANALYZER_REGISTRY


# ---------------------------------------------------------------------
class SeriesProcessor:
    """Turns one time‑series into (recon, pred, metrics)."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.tokenizer = build(cfg.tokenizer_name, PRETOKENIZER_REGISTRY, **cfg.tokenizer_params)
        self.model = build(cfg.model_name, MODEL_REGISTRY, **cfg.model_parameters)
        self.analyzers = [build(name, DATA_ANALYZER_REGISTRY) for name in cfg.data_analyzers]

    # --------------------------------------------------------------
    def __call__(self, ts_name: str, ts_data: List[float]) -> dict:
        ts_data_split = ts_data[:cfg.input_data_length]
        # --- encode -------------------------------------------------
        data_string = self.tokenizer.encode(ts_data_split)

        # --- LLM interaction ---------------------------------------
        start = time.perf_counter()
        generated = self.model.generate_response(data_string)
        latency = time.perf_counter() - start

        # --- decode -------------------------------------------------
        reconstructed, _ = inverse_transform_safe(self.tokenizer, data_string, ts_data_split[-1])
        predicted, pred_success = inverse_transform_safe(self.tokenizer, generated, ts_data_split[-1])

        # --- metrics ----------------------------------------------
        analysis_result = {}
        if pred_success:
            true_segment = ts_data[len(ts_data_split) : len(ts_data_split) + len(predicted)]
            min_len = min(len(true_segment), len(predicted))
            true_segment = true_segment[:min_len]
            predicted = predicted[:min_len]
            for a in self.analyzers:
                analysis_result[a.AnalyzerType] = a.Analyze(true_segment, predicted)
        else:
            analysis_result["Malformed output"] = 0.0

        return {
            "id": ts_name,
            "inverse_success": pred_success,
            "data": {
                "original_split": safe_to_list(ts_data_split),
                "reconstructed": safe_to_list(reconstructed),
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
    """Handles IO concerns only."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "results.jsonl"

    def record_jsonl(self, results: Iterable[dict]) -> None:
        with open(self.jsonl_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        fix_output_ownership(self.out_dir)


# ---------------------------------------------------------------------
class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.name = cfg.experiment_name or cfg.build_experiment_name()
        self.out_dir = Path(cfg.output_dir) / self.name
        self.processor = SeriesProcessor(cfg)
        self.recorder = ResultRecorder(self.out_dir)

    # --------------------------------------------------------------
    def run(self) -> None:
        # --- dataset ------------------------------------------------
        dataset = build(self.cfg.dataset_name, DATASET_REGISTRY, **self.cfg.dataset_params)
        series_iter = list(dataset.load())

        results: List[dict] = []
        for series in series_iter:
            ts_name = series["metadata"]["dataset_name"]
            ts_data = series["series"]
            outcome = self.processor(ts_name, ts_data)

            # side‑effect: plot ------------------------------------
            ts_plot_path = plot_series(
                ts_name,
                ts_data,
                outcome["data"]["reconstructed"],
                outcome["data"]["predicted"],
                outcome["inverse_success"],
                str(self.out_dir),
                prediction_offset=len(outcome["data"]["original_split"]),
            )
            outcome["plot_path"] = ts_plot_path
            results.append(outcome)

        # --- persist all -------------------------------------------
        self.recorder.record_jsonl(results)
        print(f"Experiment '{self.name}' finished. Results in {self.out_dir}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a time‑series LLM experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ExperimentRunner(cfg).run()
