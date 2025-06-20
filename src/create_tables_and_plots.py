import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for saving plots in environments without a display
import matplotlib.pyplot as plt
import json
import yaml
import math
from src.experiment_utils import fix_output_ownership

# Formatter for displaying float values nicely
fmt = lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, float) else x

# --- Utilities for data extraction and processing ---

def extract_model_name(config_path):
    """Extract model name from a YAML config file, or use folder name as fallback."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        return config.get("model_name", Path(config_path).parent.name)

def get_dynamic_group_columns(df, always_include=["experiment_name", "model_name"]):
    """
    Identify columns to group by:
    - Always include a few fixed identifiers.
    - Add only those columns with more than one unique value (excluding static ones).
    """
    exclude = set([
        "experiment_name", "metric_name", "metric_value", "dataset_name", "prediction_index",
        "input_data_length", "input_data_factor", "device", "max_new_tokens", "top_p", "do_sample"
    ])
    candidates = [col for col in df.columns if col not in exclude and col not in always_include]
    dynamic = [col for col in candidates if df[col].nunique(dropna=False) > 1]
    return always_include + dynamic

def find_tsv_in_folder(folder_path):
    """Locate a single TSV file named 'master_results.tsv' in the provided folder."""
    folder = Path(folder_path)
    tsvs = list(folder.glob("master_results.tsv"))
    if not tsvs:
        raise FileNotFoundError(f"No master_results TSV files found in directory: {folder.resolve()}")
    if len(tsvs) > 1:
        raise RuntimeError(f"Multiple TSV files found in directory {folder.resolve()}: {tsvs}\n"
                           f"Please specify the file directly.")
    return tsvs[0]

def read_experiment_tsv(input_path):
    """Read a TSV file from a file or folder containing a single TSV file."""
    path = Path(input_path)
    if path.is_dir():
        tsv_path = find_tsv_in_folder(path)
    elif path.is_file() and path.suffix == ".tsv":
        tsv_path = path
    else:
        raise FileNotFoundError(f"Input '{input_path}' is not a .tsv file or directory containing a .tsv file.")
    print(f"Reading TSV: {tsv_path.resolve()}")
    return pd.read_csv(tsv_path, sep="\t", skipinitialspace=False)

# --- Aggregation Functions ---

def group_mean_na_if_any_na(df, group_cols):
    """
    For each group:
    - Return mean of 'metric_value' if no NA exists, otherwise NA.
    - Also return spread (std) ignoring NAs.
    """
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
    """
    For each group:
    - Return median of 'value_col' if no NA exists, otherwise NA.
    - Compute spread as mean absolute deviation (MAD) from mean, ignoring NAs.
    """
    def median_or_na(series):
        if series.isna().any():
            return pd.NA
        return series.median()
    def spread(series):
        series = series.dropna()
        return (series - series.mean()).abs().mean()
    grouped = df.groupby(group_cols, dropna=False)[value_col]
    summary = grouped.agg(
        median=median_or_na,
        spread=spread
    ).reset_index()
    return summary

# --- Main summary logic ---

def summarize_metrics_df(df, out_dir, suffix="", subfolder=None):
    """
    Summarize metrics:
    1. Mean metric per series (per-run).
    2. Median and spread across series (per-config).
    3. Wide pivot table with median/spread rows.
    Saves all three summaries to disk.
    """
    if subfolder:
        out_dir = out_dir / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-run summary ---
    per_run_group_cols = get_dynamic_group_columns(df)
    per_run_summary = group_mean_na_if_any_na(df, per_run_group_cols)
    per_run_summary_formated = per_run_summary.map(fmt)
    per_run_summary_formated.to_csv(out_dir / f"metric_per_run{suffix}.tsv", sep="\t", index=False)
    print(f"Saved metric per run (with mean) to {out_dir / f'metric_per_run{suffix}.tsv'}")

    # --- Per-config summary ---
    per_group_cols = [c for c in per_run_group_cols if c not in ("series_id", "prediction_index")]
    per_group_summary = group_median_na_if_any_na_with_spread(per_run_summary, per_group_cols).map(fmt)
    per_group_summary_formated = per_group_summary.map(fmt)
    per_group_summary_formated.to_csv(out_dir / f"metric_per_config{suffix}.tsv", sep="\t", index=False)
    print(f"Saved metric per config (median/spread) to {out_dir / f'metric_per_config{suffix}.tsv'}")

    # --- Wide pivot table ---
    index = ["series_id"]
    col_index = [c for c in per_run_group_cols if c not in index]
    pivot = per_run_summary.pivot_table(index=index, columns=col_index, values="mean_value")

    # Add Median and Spread rows
    median_row = [pd.NA if col_series.isna().any() else col_series.median() for _, col_series in pivot.items()]
    spread_row = [(col_series.dropna() - col_series.dropna().mean()).abs().mean() for _, col_series in pivot.items()]
    pivot.loc["Median"] = median_row
    pivot.loc["Spread"] = spread_row

    pivot = pivot.astype("object").map(fmt)
    pivot.to_csv(out_dir / f"metric_summary_wide{suffix}.tsv", sep="\t")
    print(f"Saved wide metric summary to {out_dir / f'metric_summary_wide{suffix}.tsv'}")

def summarize_metrics_table(tsv_path, out_dir, comparison_metric="seasonalMeanAbsoluteScaledError", model_specific=False):
    """Wrapper that filters metric and delegates to summarization, either per-model or global."""
    df = read_experiment_tsv(tsv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Replace "Malformed output" with NA
    malformed_mask = df["metric_name"].eq("Malformed output")
    df.loc[malformed_mask, "metric_name"] = comparison_metric
    df.loc[malformed_mask, "metric_value"] = pd.NA 

    # Fallback if metric not found
    if comparison_metric not in df['metric_name'].unique():
        fallback = "MeanAbsoluteScaledError"
        if fallback in df['metric_name'].unique():
            print(f"WARNING: {comparison_metric} not found, using {fallback} instead.")
            comparison_metric = fallback
        else:
            raise ValueError(f"Neither '{comparison_metric}' nor '{fallback}' found in 'metric_name' column.")

    summary_df = df[df["metric_name"] == comparison_metric].copy()

    # Summarize per model if requested
    if model_specific:
        unique_models = summary_df["model_name"].unique()
        for model in unique_models:
            model_df = summary_df[summary_df["model_name"] == model]
            print(f"Analyzing model: {model}")
            summarize_metrics_df(model_df, out_dir, suffix="", subfolder=f"model_{model}")
    else:
        summarize_metrics_df(summary_df, out_dir, suffix="", subfolder="global")

# --- Plotting Functions ---

def plot_predictions_across_models(base_folder, output_folder):
    """
    Plot and compare predicted series from all models on the same chart,
    grouped by dataset ID, with consistent coloring for models.
    """
    base_path = Path(base_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_data = {}
    all_models = set()

    # Collect all predictions and metadata
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
                recn = entry["data"]["reconstructed_train"]
                pred = entry["data"].get("predicted")

                if ds not in series_data:
                    series_data[ds] = {
                        "original": orig,
                        "reconstructed": recn,
                        "models": {}
                    }
                if pred:
                    series_data[ds]["models"][model_name] = pred

    # Assign a unique color to each model
    all_models = sorted(all_models)
    pred_colors = plt.colormaps['tab10'].colors
    model_color_map = {mdl: pred_colors[i % len(pred_colors)] for i, mdl in enumerate(all_models)}

    # Plot each dataset with original, reconstructed and model predictions
    for ds, data in series_data.items():
        orig = data["original"]
        recn = data["reconstructed"]
        r_end = len(recn)

        plt.figure(figsize=(12, 5))
        plt.plot(range(len(orig)), orig, label="Original", color="black", linewidth=1.3, zorder=1)
        plt.plot(range(r_end), recn, color="gray", linestyle=":", linewidth=1.0, alpha=0.9, label="Reconstructed", zorder=2)

        for mdl, pred in data["models"].items():
            xs = range(r_end, r_end + len(pred))
            plt.plot(xs, pred, linestyle="--", linewidth=1.3, color=model_color_map[mdl], alpha=0.95,
                     label=f"Prediction - {mdl}", zorder=2)

        plt.title(f"{ds} - Predictions by Model")
        plt.xlabel("Time")
        plt.ylabel("Value Steps")
        plt.grid(True, alpha=0.18)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / f"{ds}_all_models.png")
        plt.close()

    print(f"Plots saved to: {output_dir.resolve()}")

def plot_single_series(folder_path, output_folder):
    """
    Plot individual prediction charts per dataset for one model.
    Includes spread area if multiple predictions exist.
    """
    folder_path = Path(folder_path)
    config_path = folder_path / "experiment_config.yaml"
    responses_path = folder_path / "model_responses.jsonl"

    if not config_path.exists():
        raise FileNotFoundError(f"No experiment_config.jsonl found in {folder_path}")

    model_name = extract_model_name(config_path)

    colour_ground = "#1a1a40"
    # colour_forecast = "#0055cc"
    # colour_spread = "#4da6ff"
    colour_forecast = "#df8600"
    colour_spread = "#ffb84d"


    if not responses_path.exists():
        raise FileNotFoundError(f"No model_responses.jsonl found in {folder_path}")

    with open(responses_path, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    n_entries = len(lines)
    n_cols = 3
    n_rows = math.ceil(n_entries / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), dpi=300)
    fig.suptitle(model_name, fontsize=15)
    axes = axes.flatten()

    for idx, (entry, ax) in enumerate(zip(lines, axes)):
        ds_id = entry["id"]
        data = entry["data"]
        original = np.array(data["original"])

        # Handle legacy key naming
        if "train" in data:
            train = np.array(data["train"])
        elif "original_split" in data:
            train = np.array(data["original_split"])

        train_len = len(train)
        orig_len = len(original)

        # Collect valid predictions
        preds = []
        if "predictions" in entry and isinstance(entry["predictions"], list):
            for pred_block in entry["predictions"]:
                if pred_block.get("inverse_success", True):
                    preds.append(np.array(pred_block["predicted"]))
        if not preds:
            ax.text(0.5, 0.5, 'No valid predictions', ha='center', va='center', fontsize=10)
            ax.set_title(ds_id, fontsize=13)
            continue

        # Trim predictions to equal length
        min_len = min(len(p) for p in preds)
        preds = [p[:min_len] for p in preds]

        label_name = "Mean Prediction"
        if len(preds) == 1:
            mean_pred = preds[0]
            show_spread = False
            label_name = 'Prediction'
        else:
            preds_arr = np.vstack(preds)
            mean_pred = np.mean(preds_arr, axis=0)
            min_pred = np.min(preds_arr, axis=0)
            max_pred = np.max(preds_arr, axis=0)
            show_spread = True

        max_pred_points = orig_len - train_len
        mean_pred = mean_pred[:max_pred_points]
        pred_x = np.arange(train_len, train_len + len(mean_pred))

        # Individual plot
        fig_single, ax_single = plt.subplots(figsize=(6, 4), dpi=300)
        ax_single.plot(np.arange(orig_len), original, label="Original", color=colour_ground, linewidth=0.9, zorder=2)
        ax_single.plot(pred_x, mean_pred, label=label_name, color=colour_forecast, linewidth=1.2, zorder=3)

        if show_spread:
            min_pred = min_pred[:max_pred_points]
            max_pred = max_pred[:max_pred_points]
            ax_single.fill_between(pred_x, min_pred, max_pred, color=colour_spread, alpha=0.4,
                                   label="Prediction spread (min/max)", zorder=1)

        ax_single.set_title(f"{model_name}: {ds_id}", fontsize=13)
        ax_single.set_xlabel("Time (-)", fontsize=11)
        ax_single.set_ylabel("Amplitude (-)", fontsize=11)
        ax_single.grid(alpha=0.18)
        ax_single.tick_params(axis='both', labelsize=9)
        ax_single.legend(fontsize=10, loc='upper left')

        single_save_path = folder_path / f"{ds_id}.pdf"
        fig_single.savefig(single_save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig_single)

        # Add to grid plot
        ax.plot(np.arange(orig_len), original, label="Original", color=colour_ground, linewidth=0.9, zorder=2)
        ax.plot(pred_x, mean_pred, label=label_name, color=colour_forecast, linewidth=0.80, zorder=3)
        if show_spread:
            ax.fill_between(pred_x, min_pred, max_pred, color=colour_spread, alpha=0.4, label="Prediction spread (min/max)", zorder=1)
        ax.set_title(ds_id, fontsize=13)
        ax.set_xlabel("Time (-)", fontsize=11)
        ax.set_ylabel("Amplitude (-)", fontsize=11)
        ax.grid(alpha=0.18)
        ax.tick_params(axis='both', labelsize=9)
        if idx == 0:
            ax.legend(fontsize=10, loc='upper left')

    for ax in axes[n_entries:]:
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.22)
    save_path = folder_path / f"{model_name}.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"[INFO] Saved PDF figure with {n_entries} plots to {save_path}")

# --- CLI Entrypoint ---

def main():
    parser = argparse.ArgumentParser(description="Summarize and plot metrics from experiment results or model response subfolders.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to experiment results top folder (or TSV)")
    parser.add_argument("--output_folder", type=str, default=None, help="Directory where outputs will be saved")
    parser.add_argument("--summarize", action="store_true", help="Summarize metrics and create normalized comparisons")
    parser.add_argument("--model_specific", action="store_true", help="Summarize metrics and create normalized comparisons")
    parser.add_argument("--plot_predictions", action="store_true", help="Plot prediction series from all model_responses in subfolders")
    parser.add_argument("--plot_predictions_single", type=str, default=None, help="Plot prediction series from a single model with spread")
    parser.add_argument("--comparison_metric", type=str, default="seasonalMeanAbsoluteScaledError", help="Metric for comparison (default: seasonalMeanAbsoluteScaledError)")

    args = parser.parse_args()
    output_folder = Path(args.output_folder) if args.output_folder else Path(args.input_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summarize_metrics_table(args.input_folder, output_folder, args.comparison_metric, args.model_specific)
        fix_output_ownership(output_folder)
    if args.plot_predictions:
        plot_predictions_across_models(args.input_folder, output_folder)
        fix_output_ownership(output_folder)
    if args.plot_predictions_single:
        # Find a single folder matching the search string inside input_folder
        base_path = Path(args.input_folder)
        matches = [f for f in base_path.iterdir() if f.is_dir() and args.plot_predictions_single in f.name]
        if not matches:
            raise RuntimeError(f"No folder matching '{args.plot_predictions_single}' found in {base_path}")
        for match in matches:
            plot_single_series(match, output_folder)
            fix_output_ownership(output_folder)

    if not (args.summarize or args.plot_predictions or args.plot_predictions_single):
        parser.print_help()


if __name__ == "__main__":
    main()
