import yaml
import json
import time
import os
from pathlib import Path
from experiment_utils import *
from available_datasets import get_dataset
from pretokenizer import get_pretokenizer
from lmwrapper import get_model
from data_analyzers import get_data_analyzer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
        
def generate_experiment_name(config):
    name = f"PTOK-{config['tokenizer_name']}_LLM-{config['checkpoint_name']}_"
    name += f"NTOK{config['max_new_tokens']}_MKER{config['max_kernels']}_SLEN{config['sequence_length']}"
    return name
    
def run(config):
    experiment_name = generate_experiment_name(config)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    output_folder = f"results/{experiment_name}_{timestamp}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(config["dataset_name"], **config.get("dataset_params", {}))
    ts_list = dataset.load()

    tokenizer = get_pretokenizer(config["tokenizer_name"], **config.get("tokenizer_params", {}))
    model = get_model(config["checkpoint_name"], max_new_tokens=config["max_new_tokens"], temperature=1.0, top_p=0.9)

    analyzers = []
    for analyzer_name in config['data_analyzers']:
        analyzers.append(get_data_analyzer(analyzer_name))

    save_experiment_settings(output_folder, model, tokenizer, dataset, analyzers)
    results = []
    jsonl_path = os.path.join(output_folder, config["output_jsonl"])

    for idx, ts in enumerate(ts_list):
        ts_data = ts["target"]
        ts_data_split = split_data(ts_data, config["prompt_length_factor"])
        data_string = tokenizer.encode(ts_data_split)
        model_response = model.generate_response(data_string)

        reconstructed, _ = inverse_transform_safe(tokenizer, data_string, ts_data_split[0])
        predicted, pred_success = inverse_transform_safe(tokenizer, model_response, ts_data_split[-1])

        analysis_result = {}
        if (pred_success):
            true_values = ts_data[len(ts_data_split):len(ts_data_split) + len(predicted)]

            # Adjust in case predicted is longer than the available true data
            min_len = min(len(true_values), len(predicted))
            true_values = true_values[:min_len]
            predicted = predicted[:min_len]

            for analyzer in analyzers:
                analysis_result[analyzer.AnalyzerType] = analyzer.Analyze(true_values, predicted)
        else:
            analysis_result["Malformed output"] = 0

        plot_path = plot_series(idx, ts_data, reconstructed, predicted, pred_success, output_folder, prediction_offset = len(ts_data_split))

        result = {
            "id": idx,
            "inverse_success": pred_success,
            "plot_path": plot_path,
            "data": {
                "original": safe_to_list(ts_data),
                "original_split": safe_to_list(ts_data_split),
                "reconstructed": safe_to_list(reconstructed),
                "predicted": safe_to_list(predicted)
            },
            "model": {
                "original_string": data_string,
                "model_response": model_response,
            },
            "analysis": analysis_result
        }

        results.append(result)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    fix_output_ownership(Path(output_folder))
    print(f"Experiment {experiment_name} complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    run(config)

