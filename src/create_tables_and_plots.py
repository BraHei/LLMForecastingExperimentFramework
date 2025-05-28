import argparse
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import yaml

from src.experiment_utils import fix_output_ownership

def extract_model_name(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        return config.get("model_name", Path(config_path).parent.name)

def get_dynamic_group_columns(df, always_include=["experiment_name", "model_name"]):
    exclude = set([
        "experiment_name", "metric_name", "metric_value", "dataset_name", "prediction_index",
        "input_data_length", "input_data_factor", "device", "max_new_tokens", "top_p", "do_sample"
    ])
    candidates = [col for col in df.columns if col not in exclude and col not in always_include]
    dynamic = [col for col in candidates if df[col].nunique(dropna=False) > 1]
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
    return pd.read_csv(tsv_path, sep="\t", skipinitialspace=False)

def group_mean_na_if_any_na(df, group_cols):
    # Returns NA if any NA in the group, otherwise mean and spread (std)
    def mean_or_na(series):
        if series.isna().any():
            return pd.NA
        return series.mean()
    def spread(series):
        return series.dropna().std()
    result = (
        df.groupby(group_cols, dropna=False)["metric_value"]
          .agg(mean_value=mean_or_na, spread=spread)
          .reset_index()
    )
    return result

def group_median_na_if_any_na_with_spread(df, group_cols, value_col="mean_value"):
    # Returns NA for median if any NA in group, std as spread (ignoring NAs)
    def median_or_na(series):
        if series.isna().any():
            return pd.NA
        return series.median()
    def spread(series):
        return series.dropna().std()
    grouped = df.groupby(group_cols, dropna=False)[value_col]
    summary = grouped.agg(
        median=median_or_na,
        spread=spread
    ).reset_index()
    return summary

def summarize_metrics_df(df, out_dir, suffix="", subfolder=None):
    """
    Aggregates per measurement (series_id) with mean, then
    aggregates per config with median/spread (strict NA rules).
    Outputs both per-run and per-group summaries, plus wide table.
    """
    if subfolder:
        out_dir = out_dir / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-measurement mean (over predictions per series) ---
    # Grouping: config columns + series_id
    per_run_group_cols = get_dynamic_group_columns(df)
    per_run_summary = group_mean_na_if_any_na(df, per_run_group_cols)
    per_run_summary.to_csv(out_dir / f"metric_per_run{suffix}.tsv", sep="\t", index=False)
    print(f"Saved metric per run (with mean) to {out_dir / f'metric_per_run{suffix}.tsv'}")

    # --- Per-config median/spread (across series/runs) ---
    # Grouping: config columns (no series_id or prediction_index)
    per_group_cols = [c for c in per_run_group_cols if c not in ("series_id", "prediction_index")]
    per_group_summary = group_median_na_if_any_na_with_spread(
        per_run_summary, per_group_cols, value_col="mean_value"
    )
    per_group_summary.to_csv(out_dir / f"metric_per_config{suffix}.tsv", sep="\t", index=False)
    print(f"Saved metric per config (median/spread) to {out_dir / f'metric_per_config{suffix}.tsv'}")

    # --- Wide summary table (rows: series_id, cols: config columns), median and spread at the bottom ---
    index = ["series_id"]
    col_index = [c for c in per_run_group_cols if c not in index]
    pivot = per_run_summary.pivot_table(index=index, columns=col_index, values="mean_value")

    # Median row: strict NA logic (NA if any in col)
    median_row = [
        pd.NA if col_series.isna().any() else col_series.median()
        for _, col_series in pivot.items()
    ]
    # Spread row: std, skipna
    spread_row = [
        col_series.dropna().std()
        for _, col_series in pivot.items()
    ]
    pivot.loc["Median"] = median_row
    pivot.loc["Spread"] = spread_row

    # Pretty format numbers
    fmt = lambda x: f"{x:.4f}" if pd.notna(x) and isinstance(x, (float, int)) else x
    pivot = pivot.astype("object").map(fmt)
    pivot.to_csv(out_dir / f"metric_summary_wide{suffix}.tsv", sep="\t")
    print(f"Saved wide metric summary to {out_dir / f'metric_summary_wide{suffix}.tsv'}")

def summarize_metrics_table(tsv_path, out_dir, comparison_metric="seasonalMeanAbsoluteScaledError", model_specific=False):
    df = read_experiment_tsv(tsv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    malformed_mask = df["metric_name"].eq("Malformed output")
    df.loc[malformed_mask, "metric_name"] = comparison_metric
    df.loc[malformed_mask, "metric_value"] = pd.NA 

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
        fix_output_ownership(output_folder)
    if args.plot_predictions:
        plot_predictions_across_models(args.input_folder, output_folder)
        fix_output_ownership(output_folder)
    if not (args.summarize or args.plot_predictions):
        parser.print_help()


if __name__ == "__main__":
    main()
