import os
import json
import yaml
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

compressed_instruction = {'None':'None', 'Forecast the next 25 steps: ' : 'F25S', 
                          'Predict the next 50 steps: ': 'P50S', 
                          'With the following Time Series, forecast the next 50 steps: ': 'WF50S', 
                          'Predict the next 50 steps given the previous steps: ' : 'PN50S'}

def extract_mae_from_results(results_path):
    maes = {}
    with open(results_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            dataset_id = entry['id']
            mae = entry['metrics'].get('MeanAbsoluteError')
            maes[dataset_id] = mae
    return maes

def extract_model_name(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config.get('model_name', 'unknown_model')

def extract_compressed_instruction(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        instruction = config.get('instruction_string', '-')
        if instruction in compressed_instruction:
            instruction = compressed_instruction[instruction]
        else:
            print("WARNING: Couldn't find instruction in lookup table, its uncompressed")
        return instruction 
    
def extract_compressed_seperator(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config.get('preprocessor_params', {}).get('time_sep', 'no_seperator')

def extract_result_file_name(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config.get('output_jsonl', 'unknown_file_name')

def build_metric_table(base_folder, instruction, seperator):
    base_path = Path(base_folder)
    records = []

    columns_to_use = ["Model", "Metric"]

    if (instruction):
        columns_to_use.append("Instruction")

    if (seperator):
        columns_to_use.append("Seperator")

    for subdir in base_path.iterdir():
        if subdir.is_dir():
            config_path = subdir / "experiment_config.yaml"
            results_path = subdir / "results.jsonl"
            if config_path.exists():
                output_filename = extract_result_file_name(config_path)
                results_path = subdir / output_filename

                if not results_path.exists():
                    print("WARNING: Couldnt find the output file named f{output_filename}, defaulting back to result.jsonl")
                    results_path = subdir / "results.jsonl"

            if config_path.exists() and results_path.exists():
                model_name = extract_model_name(config_path)
                used_instruction_compressed = extract_compressed_instruction(config_path)
                used_seperator = extract_compressed_seperator(config_path)

                with open(results_path, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        dataset = entry['id']
                        mae = entry['metrics'].get('MeanAbsoluteError')
                        # mse = entry['metrics'].get('MeanSquareError')

                        records.append({
                            "Model": model_name,
                            "Instruction": used_instruction_compressed,
                            "Seperator": used_seperator,
                            "Dataset": dataset,
                            "Metric": "MAE",
                            "Value": mae
                        })
                        # records.append({
                        #     "Model": model_name,
                        #     "Dataset": dataset,
                        #     "Metric": "MSE",
                        #     "Value": mse
                        # })

    df = pd.DataFrame(records)
    metric_table = df.pivot_table(index="Dataset", columns=columns_to_use, values="Value")
    metric_table.columns = pd.MultiIndex.from_tuples(metric_table.columns)
    return metric_table

def plot_predictions_across_models(base_folder, output_folder):
    base_path = Path(base_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_data = {}

    for subdir in base_path.iterdir():
        if subdir.is_dir():
            config_path = subdir / "experiment_config.yaml"
            results_path = subdir / "results.jsonl"

            if config_path.exists():
                output_filename = extract_result_file_name(config_path)
                results_path = subdir / output_filename

                if not results_path.exists():
                    print("WARNING: Couldnt find the output file named f{output_filename}, defaulting back to result.jsonl")
                    results_path = subdir / "results.jsonl"

            if config_path.exists() and results_path.exists():
                model_name = extract_model_name(config_path)

                with open(results_path, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        dataset_id = entry["id"]
                        original = entry["data"]["original"]
                        reconstructed = entry["data"]["reconstructed_split"]
                        predicted = entry["data"].get("predicted", None)

                        if dataset_id not in series_data:
                            series_data[dataset_id] = {
                                "original": original,
                                "reconstructed": reconstructed,
                                "models": {}
                            }

                        if predicted:
                            series_data[dataset_id]["models"][model_name] = predicted
                        else:
                            print(f"Warning: No 'predicted' values for model '{model_name}' on dataset '{dataset_id}'")

    for dataset, data in series_data.items():
        original = data["original"]
        reconstructed = data["reconstructed"]
        r_end = len(reconstructed)

        # Determine how far to plot based on longest prediction
        max_pred_len = max((len(p) for p in data["models"].values()), default=0)
        plot_end = min(len(original), r_end + max_pred_len)

        plt.figure(figsize=(12, 5))
        plt.plot(range(plot_end), original[:plot_end], label="Original", linewidth=2, color='black')
        plt.plot(range(r_end), reconstructed, label="Reconstructed", linestyle=":", color='gray')

        for model, prediction in data["models"].items():
            prediction = prediction[:plot_end - r_end]  # Trim if it goes beyond the original
            x = list(range(r_end, r_end + len(prediction)))
            plt.plot(x, prediction, label=f"Prediction - {model}", linestyle='--')

        plt.title(f"{dataset} - Predictions by Model")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset}_all_models.png")
        plt.close()

    print(f"Plots saved to: {output_dir.resolve()}")

def process_instruction_seperator(df, output_dir):
    normalized_df = (df - df.min(axis=1).values[:, None]) / (df.max(axis=1) - df.min(axis=1)).values[:, None]
    mean_scores = normalized_df.mean(axis=0)
    sorted_scores = mean_scores.sort_values()

    print("\n Normalized Summary Table:\n", sorted_scores)
    normalized_file = output_dir / "normalized_summary.tsv"
    sorted_scores.to_csv(normalized_file, sep="\t")
    print(f"\n Saved normalized summary to {normalized_file.resolve()}")

    sorted_scores.plot(kind='barh', figsize=(10, 6), title='Avg Normalized MAE per (Instruction + Separator)')
    plt.xlabel('Avg Normalized MAE (lower is better)')
    plt.tight_layout()
    plt.savefig(output_dir / "avg_normalized_mae.png", bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics and plot time series predictions.")
    parser.add_argument("--input_folder", type=str, help="Path to the root experiment folder (e.g., LLMABBA_LLMTimeComparison)")
    parser.add_argument("--output_folder", type=str, default=None, help="Directory where metric summary and plots will be saved")
    parser.add_argument("--instruction", type=bool, default=False, help="Path to the root experiment folder (e.g., LLMABBA_LLMTimeComparison)")
    parser.add_argument("--seperator", type=bool, default=False, help="Path to the root experiment folder (e.g., LLMABBA_LLMTimeComparison)")

    args = parser.parse_args()
    
    output_dir = Path(args.output_folder) if args.output_folder else Path(args.input_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    get_instruction = args.seperator
    get_seperator = args.instruction

    print(f" Scanning experiment results in: {args.input_folder}")
    metric_df = build_metric_table(args.input_folder, get_instruction, get_seperator)
    print("\n Metric Summary Table:\n", metric_df)

    # Save to tsv
    metric_file = output_dir / "metric_summary.tsv"
    metric_df.to_csv(metric_file, sep="\t")
    print(f"\n Saved metric summary to {metric_file.resolve()}")

    # Plot
    if (get_instruction or get_seperator):
        process_instruction_seperator(metric_df, output_dir)
    else:
        plot_predictions_across_models(args.input_folder, output_folder=output_dir)

if __name__ == "__main__":
    main()
