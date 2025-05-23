import argparse
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import yaml

def extract_model_name(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        return config.get("model_name", Path(config_path).parent.name)

def get_dynamic_group_columns(df, always_include=["model_name", "series_id"]):
    exclude = set([
        "experiment_name", "metric_name", "metric_value", "dataset_name", "instruction_text",
        "input_data_length", "input_data_factor", "device", "max_new_tokens",
        "temperature", "top_p", "do_sample"
    ])
    candidates = [col for col in df.columns if col not in exclude and col not in always_include]
    dynamic = [col for col in candidates if df[col].nunique() > 1]
    return always_include + dynamic

def find_tsv_in_folder(folder_path):
    folder = Path(folder_path)
    tsvs = list(folder.glob("master_results.tsv"))
    if not tsvs:
        raise FileNotFoundError(f"No master_results TSV files found in directory: {folder.resolve()}")
    if len(tsvs) > 1:
        raise RuntimeError(f"Multiple TSV files found in directory {folder.resolve()}: {tsvs}\n"
                           f"Please specify the file directly.")
    return tsvs[0]

def read_experiment_tsv(input_path):
    path = Path(input_path)
    if path.is_dir():
        tsv_path = find_tsv_in_folder(path)
    elif path.is_file() and path.suffix == ".tsv":
        tsv_path = path
    else:
        raise FileNotFoundError(f"Input '{input_path}' is not a .tsv file or directory containing a .tsv file.")
    print(f"Reading TSV: {tsv_path.resolve()}")
    return pd.read_csv(tsv_path, sep="\t")

def summarize_metrics_df(df, out_dir, suffix="", subfolder=None):
    """
    Perform summary analysis (win counts, mean metrics, pivot table) on a provided DataFrame,
    writing output files with an optional suffix (e.g., per-model).
    """
    if subfolder:
        out_dir = out_dir / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)

    group_cols = get_dynamic_group_columns(df)
    index = ["series_id"]
    col_index = [col for col in group_cols if col != "series_id"]
    pivot = df.pivot_table(index=index, columns=col_index, values="metric_value", aggfunc="median")
    
    winners = pivot.eq(pivot.min(axis=1), axis=0)
    win_counts = winners.sum(axis=0).sort_values(ascending=False)
    median_metric = pivot.median(axis=0).sort_values()
    median_metric = median_metric.astype("object").map(lambda x: f"{x:.2f}" if pd.notnull(x) and isinstance(x, (float, int)) else "")
    
    # Save win summary
    win_counts.to_csv(out_dir / f"win_summary{suffix}.tsv", sep="\t", header=["Wins"])
    print(f"Saved win summary to {out_dir / f'win_summary{suffix}.tsv'}")
    # Save win mask
    winners.astype(int).to_csv(out_dir / f"wins{suffix}.tsv", sep="\t")
    # Save mean summary
    median_metric.to_csv(out_dir / f"median_summary{suffix}.tsv", sep="\t")
    print(f"Saved mean summary to {out_dir / f'median_summary{suffix}.tsv'}")
    # Save pivot (with mean row for convenience)
    pivot_mean = pd.concat([pivot, pd.DataFrame([median_metric], index=["Median"])])
    pivot_formatted = pivot_mean.astype("object").map(lambda x: f"{x:.2f}" if pd.notnull(x) and isinstance(x, (float, int)) else "")
    pivot_formatted.to_csv(out_dir / f"metric_summary_wide{suffix}.tsv", sep="\t")
    print(f"Saved wide metric summary to {out_dir / f'metric_summary_wide{suffix}.tsv'}")

def summarize_metrics_table(tsv_path, out_dir, comparison_metric="seasonalMeanAbsoluteScaledError", model_specific=False):
    df = read_experiment_tsv(tsv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if comparison_metric not in df['metric_name'].unique():
        fallback = "MeanAbsoluteScaledError"
        if fallback in df['metric_name'].unique():
            print(f"WARNING: {comparison_metric} not found, using {fallback} instead.")
            comparison_metric = fallback
        else:
            raise ValueError(f"Neither '{comparison_metric}' nor '{fallback}' found in 'metric_name' column.")

    summary_df = df[df["metric_name"] == comparison_metric].copy()

    if model_specific:
        unique_models = summary_df["model_name"].unique()
        for model in unique_models:
            model_df = summary_df[summary_df["model_name"] == model]
            print(f"Analyzing model: {model}")
            summarize_metrics_df(model_df, out_dir, suffix="", subfolder=f"model_{model}")
    else:
        summarize_metrics_df(summary_df, out_dir, suffix="", subfolder="global")


def collect_and_merge_model_responses(parent_folder):
    parent_folder = Path(parent_folder)
    model_folders = [f for f in parent_folder.iterdir() if f.is_dir() and (f / "model_responses.jsonl").exists()]
    all_rows = []
    for folder in model_folders:
        # Try to get context from config
        config_path = folder / "experiment_config.yaml"
        config = {}
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            except Exception as ex:
                print(f"Could not load config: {ex}")

        responses_path = folder / "model_responses.jsonl"
        with open(responses_path, "r") as fin:
            for line in fin:
                row = json.loads(line)
                # Get model_name and dataset_name from config if not present
                row["model_name"] = config.get("model_name", folder.name)
                # Will try to get dataset_name from config or row
                if "dataset_name" not in row:
                    # From config (if dataset_names is a list, match id, else just use)
                    ds = config.get("dataset_name")
                    ds_list = config.get("dataset_params", {}).get("dataset_names")
                    row["dataset_name"] = ds if ds else (ds_list[0] if isinstance(ds_list, list) else ds_list)
                # For clarity, store which folder/run this was from
                row["_run_dir"] = folder.name
                all_rows.append(row)
    if not all_rows:
        raise ValueError("No model_responses.jsonl rows found in any subfolder!")
    df = pd.DataFrame(all_rows)
    return df


from pathlib import Path
import json
import matplotlib.pyplot as plt

def plot_predictions_across_models(base_folder, output_folder):
    base_path = Path(base_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_data = {}
    all_models = set()

    # First pass: gather all model names
    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue

        config_path = subdir / "experiment_config.yaml"
        results_path = subdir / "model_responses.jsonl"
        if not (config_path.exists() and results_path.exists()):
            continue

        model_name = extract_model_name(config_path)
        all_models.add(model_name)

        with open(results_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                ds = entry["id"]
                orig = entry["data"]["original"]
                recn = entry["data"]["reconstructed_split"]
                pred = entry["data"].get("predicted")

                if ds not in series_data:
                    series_data[ds] = {
                        "original": orig,
                        "reconstructed": recn,
                        "models": {}
                    }
                if pred:
                    series_data[ds]["models"][model_name] = pred

    # Assign each model a unique color
    all_models = sorted(all_models)
    pred_colors = plt.colormaps['tab10'].colors

    model_color_map = {
        mdl: pred_colors[i % len(pred_colors)] for i, mdl in enumerate(all_models)
    }

    for ds, data in series_data.items():
        orig = data["original"]
        recn = data["reconstructed"]
        r_end = len(recn)

        plt.figure(figsize=(12, 5))

        plt.plot(range(len(orig)), orig, label="Original", color="black", linewidth=1.3, zorder=1)
        plt.plot(range(r_end), recn, color="gray", linestyle=":", linewidth=1.0, alpha=0.9, label="Reconstructed", zorder=2)

        # Plot all model predictions with consistent colors
        for mdl, pred in data["models"].items():
            xs = range(r_end, r_end + len(pred))
            plt.plot(xs, pred, linestyle="--", linewidth=1.3,
                     color=model_color_map[mdl], alpha=0.95,
                     label=f"Prediction - {mdl}", zorder=2)

        plt.title(f"{ds} - Predictions by Model")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.18)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / f"{ds}_all_models.png")
        plt.close()

    print(f"Plots saved to: {output_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Summarize and plot metrics from experiment results or model response subfolders.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to experiment results top folder (or TSV)")
    parser.add_argument("--output_folder", type=str, default=None, help="Directory where outputs will be saved")
    parser.add_argument("--summarize", action="store_true", help="Summarize metrics and create normalized comparisons")
    parser.add_argument("--model_specific", action="store_true", help="Summarize metrics and create normalized comparisons")
    parser.add_argument("--plot-predictions", action="store_true", help="Plot prediction series from all model_responses in subfolders")
    parser.add_argument("--comparison-metric", type=str, default="seasonalMeanAbsoluteScaledError", help="Metric for comparison (default: seasonalMeanAbsoluteScaledError)")

    args = parser.parse_args()
    output_folder = Path(args.output_folder) if args.output_folder else Path(args.input_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summarize_metrics_table(args.input_folder, output_folder, args.comparison_metric, args.model_specific)
    if args.plot_predictions:
        plot_predictions_across_models(args.input_folder, output_folder)
    if not (args.summarize or args.plot_predictions):
        parser.print_help()

if __name__ == "__main__":
    main()
