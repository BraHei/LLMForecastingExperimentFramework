import yaml
import json
import time
import os
from pathlib import Path
from src.experiment_utils import *
from src.available_datasets import get_dataset
from src.pretokenizer import get_pretokenizer
from src.lmwrapper import get_model
from src.data_analyzers import get_data_analyzer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
        
def generate_experiment_name(config):
    base = f"PTOK-{config['tokenizer_name']}_LLM-{config['model_name']}_NTOK{config['model_parameters']['max_new_tokens']}"

    dataset_name = config["dataset_name"]
    dataset_params = config.get("dataset_params", {})

    # KernelSynth-specific naming
    if dataset_name == "kernelsynth":
        name = f"{base}_DS-kernelsynth_MKER{dataset_params.get('max_kernels', '?')}_SLEN{dataset_params.get('sequence_lenght', '?')}"
    # Darts-specific naming
    elif dataset_name == "darts":
        dataset_names = dataset_params.get("dataset_names", [])
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        datasets_str = "-".join(dataset_names)
        name = f"{base}_DS-darts"
    # Nixtla or other datasets
    else:
        name = f"{base}_DS-{dataset_name}"

    return name
    
def run(config):
    experiment_name = generate_experiment_name(config)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    output_folder = f"results/{experiment_name}_{timestamp}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(config["dataset_name"], **config.get("dataset_params", {}))
    ts_list = dataset.load()

    tokenizer = get_pretokenizer(config["tokenizer_name"], **config.get("tokenizer_params", {}))
    model = get_model(config["model_name"], **config.get("model_parameters", {}))

    analyzers = []
    for analyzer_name in config['data_analyzers']:
        analyzers.append(get_data_analyzer(analyzer_name))

    save_experiment_settings(output_folder, model, tokenizer, dataset, analyzers)
    results = []
    jsonl_path = os.path.join(output_folder, config["output_jsonl"])

    for ts in ts_list:
        ts_data = ts["series"]
        ts_name = ts["metadata"]["dataset_name"]
        ts_data_split = split_data(ts_data, config["prompt_length_factor"])
        data_string = tokenizer.encode(ts_data_split)
        model_response = model.generate_response(data_string)

        reconstructed, _ = inverse_transform_safe(tokenizer, data_string, ts_data_split[0])
        predicted, pred_success = inverse_transform_safe(tokenizer, model_response, ts_data_split[-1])

        analysis_result_recon = {}

        # Make a copy so we don't modify the original
        reconstructed_padded = reconstructed.copy()
        
        if len(ts_data_split) > len(reconstructed_padded):
            print("Reconstruction lost data, appending zeros to fill")
            padding_length = len(ts_data_split) - len(reconstructed_padded)
            reconstructed_padded += [0] * padding_length
        elif len(ts_data_split) < len(reconstructed_padded):
            print("Reconstruction has extra data, truncating")
            reconstructed_padded = reconstructed_padded[:len(ts_data_split)]

        for analyzer in analyzers:
            analysis_result_recon[analyzer.AnalyzerType] = analyzer.Analyze(ts_data_split, reconstructed_padded)


        analysis_result_pred = {}
        if (pred_success):
            true_values = ts_data[len(ts_data_split):len(ts_data_split) + len(predicted)]

            # Adjust in case predicted is longer than the available true data
            min_len = min(len(true_values), len(predicted))
            true_values = true_values[:min_len]
            predicted = predicted[:min_len]

            for analyzer in analyzers:
                analysis_result_pred[analyzer.AnalyzerType] = analyzer.Analyze(true_values, predicted)
        else:
            analysis_result_pred["Malformed output"] = 0

        plot_path = plot_series(ts_name, ts_data, reconstructed, predicted, pred_success, output_folder, prediction_offset = len(ts_data_split))

        predicted_lenght = len(reconstructed) + len(predicted) if (pred_success and predicted is not None) else 0
        ts_data_predict = ts_data[: predicted_lenght]

        result = {
            "id": ts_name,
            "inverse_success": pred_success,
            "plot_path": plot_path,
            "data": {
                "original_split": safe_to_list(ts_data_predict),
                "reconstructed": safe_to_list(reconstructed),
                "predicted": safe_to_list(predicted)
            },
            "model": {
                "original_string": data_string,
                "model_response": model_response,
            },
            "analysis_pred": analysis_result_pred,
            "analysis_result_recon": analysis_result_recon
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

