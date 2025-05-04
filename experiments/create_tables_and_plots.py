import os
import json
import yaml
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

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

def build_metric_table(base_folder):
    base_path = Path(base_folder)
    records = []

    for subdir in base_path.iterdir():
        if subdir.is_dir():
            config_path = subdir / "experiment_config.yaml"
            results_path = subdir / "results.jsonl"

            if config_path.exists() and results_path.exists():
                model_name = extract_model_name(config_path)

                with open(results_path, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        dataset = entry['id']
                        mae = entry['metrics'].get('MeanAbsoluteError')
                        mse = entry['metrics'].get('MeanSquareError')

                        records.append({
                            "Model": model_name,
                            "Dataset": dataset,
                            "Metric": "MAE",
                            "Value": mae
                        })
                        records.append({
                            "Model": model_name,
                            "Dataset": dataset,
                            "Metric": "MSE",
                            "Value": mse
                        })

    df = pd.DataFrame(records)
    metric_table = df.pivot_table(index="Dataset", columns=["Model", "Metric"], values="Value")
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

        plt.figure(figsize=(12, 5))
        plt.plot(range(len(original)), original, label="Original", linewidth=2, color='black')
        plt.plot(range(r_end), reconstructed, label="Reconstructed", linestyle=":", color='gray')

        for model, prediction in data["models"].items():
            p_end = r_end + len(prediction)
            if p_end > len(original):
                prediction = prediction[:len(original) - r_end]  # trim if it overshoots
                p_end = r_end + len(prediction)
            x = list(range(r_end, p_end))
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

def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics and plot time series predictions.")
    parser.add_argument("input_folder", type=str, help="Path to the root experiment folder (e.g., LLMABBA_LLMTimeComparison)")
    parser.add_argument("--output_folder", type=str, default=None, help="Directory where metric summary and plots will be saved")

    args = parser.parse_args()
    
    output_dir = Path(args.output_folder) if args.output_folder else Path(args.input_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f" Scanning experiment results in: {args.input_folder}")
    metric_df = build_metric_table(args.input_folder)
    print("\n Metric Summary Table:\n", metric_df)

    # Save to tsv
    metric_file = output_dir / "metric_summary.tsv"
    metric_df.to_csv(metric_file, sep="\t")
    print(f"\n Saved metric summary to {metric_file.resolve()}")

    # Plot
    plot_predictions_across_models(args.input_folder, output_folder=output_dir)

if __name__ == "__main__":
    main()
