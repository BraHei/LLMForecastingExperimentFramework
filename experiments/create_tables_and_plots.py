import os
import json
import yaml
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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

def extract_instruction_shorthand(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        instr = config.get('instruction_object')
        if isinstance(instr, dict):
            return instr.get('name', 'None')
        elif isinstance(instr, list) and instr:
            # if it’s a list, take the first one
            return instr[0].get('name', 'None')
        else:
            return 'None'

def extract_separator(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config.get('preprocessor_params', {}).get('time_sep', 'no_separator')

def extract_result_file_name(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config.get('output_jsonl', 'unknown_file_name')

def build_metric_table(base_folder, include_instruction, include_separator):
    base_path = Path(base_folder)
    records = []

    cols = ["Model", "Metric"]
    if include_instruction:
        cols.append("Instruction")
    if include_separator:
        cols.append("Separator")

    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue

        config_path = subdir / "experiment_config.yaml"
        if not config_path.exists():
            continue

        output_filename = extract_result_file_name(config_path)
        results_path = subdir / output_filename
        if not results_path.exists():
            print(f"WARNING: Couldn’t find the output file named {output_filename}, falling back to results.jsonl")
            results_path = subdir / "results.jsonl"
            if not results_path.exists():
                continue

        model_name = extract_model_name(config_path)
        instr_code = extract_instruction_shorthand(config_path)
        sep_code   = extract_separator(config_path)

        with open(results_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                dataset = entry['id']
                mae     = entry['metrics'].get('MeanAbsoluteError')

                rec = {
                    "Model": model_name,
                    "Metric": "MAE",
                    "Value": mae,
                }
                if include_instruction:
                    rec["Instruction"] = instr_code
                if include_separator:
                    rec["Separator"]   = sep_code
                rec["Dataset"] = dataset

                records.append(rec)

    df = pd.DataFrame(records)
    table = df.pivot_table(
        index="Dataset",
        columns=cols,
        values="Value"
    )
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table

def plot_predictions_across_models(base_folder, output_folder):
    base_path = Path(base_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_data = {}

    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue

        config_path = subdir / "experiment_config.yaml"
        if not config_path.exists():
            continue

        output_filename = extract_result_file_name(config_path)
        results_path = subdir / output_filename
        if not results_path.exists():
            print(f"WARNING: Couldn’t find the output file named {output_filename}, falling back to results.jsonl")
            results_path = subdir / "results.jsonl"
            if not results_path.exists():
                continue

        model_name = extract_model_name(config_path)

        with open(results_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                ds = entry["id"]
                orig = entry["data"]["original"]
                recn = entry["data"]["reconstructed_split"]
                pred = entry["data"].get("predicted")

                series_data.setdefault(ds, {
                    "original": orig,
                    "reconstructed": recn,
                    "models": {}
                })
                if pred:
                    series_data[ds]["models"][model_name] = pred
                else:
                    print(f"Warning: No 'predicted' values for model '{model_name}' on dataset '{ds}'")

    for ds, data in series_data.items():
        orig = data["original"]
        recn = data["reconstructed"]
        r_end = len(recn)
        max_pred = max((len(p) for p in data["models"].values()), default=0)
        plot_end = min(len(orig), r_end + max_pred)

        plt.figure(figsize=(12, 5))
        plt.plot(range(plot_end), orig[:plot_end], label="Original", linewidth=2)
        plt.plot(range(r_end), recn, linestyle=":", label="Reconstructed")

        for mdl, pred in data["models"].items():
            xs = range(r_end, r_end + min(len(pred), plot_end - r_end))
            plt.plot(xs, pred[:plot_end-r_end], linestyle="--", label=f"Prediction - {mdl}")

        plt.title(f"{ds} - Predictions by Model")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"{ds}_all_models.png")
        plt.close()

    print(f"Plots saved to: {output_dir.resolve()}")

def process_instruction_separator(df, output_dir):
    normalized = (df - df.min(axis=1).values[:, None]) / (df.max(axis=1) - df.min(axis=1)).values[:, None]
    mean_scores = normalized.mean(axis=0).sort_values()

    print("\nNormalized Summary:\n", mean_scores)
    normalized_file = output_dir / "normalized_summary.tsv"
    mean_scores.to_csv(normalized_file, sep="\t")
    print(f"Saved normalized summary to {normalized_file.resolve()}")

    ax = mean_scores.plot(kind='barh', figsize=(10, 6), title='Avg Normalized MAE per (Instr + Sep)')
    ax.set_xlabel('Avg Normalized MAE (lower is better)')
    plt.tight_layout()
    plt.savefig(output_dir / "avg_normalized_mae.png", bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate metrics and plot time series predictions."
    )
    parser.add_argument(
        "--input_folder", type=str, required=True,
        help="Root experiment folder (e.g., LLMABBA_LLMTimeComparison)"
    )
    parser.add_argument(
        "--output_folder", type=str, default=None,
        help="Directory where outputs will be saved"
    )
    parser.add_argument(
        "--instruction", action="store_true",
        help="Include the Instruction dimension in the summary"
    )
    parser.add_argument(
        "--separator", action="store_true",
        help="Include the Separator dimension in the summary"
    )

    args = parser.parse_args()

    out_dir = Path(args.output_folder) if args.output_folder else Path(args.input_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning experiment results in: {args.input_folder}")
    metric_df = build_metric_table(
        args.input_folder,
        include_instruction=args.instruction,
        include_separator=args.separator
    )
    print("\nMetric Summary Table:\n", metric_df)

    # Save summary
    metric_file = out_dir / "metric_summary.tsv"
    metric_df.to_csv(metric_file, sep="\t")
    print(f"Saved metric summary to {metric_file.resolve()}")

    # Either process instr+sep or plot
    if args.instruction or args.separator:
        process_instruction_separator(metric_df, out_dir)
    else:
        plot_predictions_across_models(args.input_folder, output_folder=out_dir)

if __name__ == "__main__":
    main()
