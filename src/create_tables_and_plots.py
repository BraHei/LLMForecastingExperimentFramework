import argparse
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_dynamic_group_columns(df, always_include=["model_name", "dataset_name"]):
    # Get all possible columns that are not "metric_name", "metric_value", "series_id", etc.
    exclude = set(["experiment_name", "metric_name", "metric_value", "series_id", "instruction_text", "input_data_length", "input_data_factor", "device", "max_new_tokens", "temperature", "top_p", "do_sample"])
    candidates = [col for col in df.columns if col not in exclude and col not in always_include]
    # Only keep those with >1 unique value (i.e., they meaningfully differentiate runs)
    dynamic = [col for col in candidates if df[col].nunique() > 1]
    # Return always-included + dynamic
    return always_include + dynamic

def find_tsv_in_folder(folder_path):
    folder = Path(folder_path)
    tsvs = list(folder.glob("*.tsv"))
    if not tsvs:
        raise FileNotFoundError(f"No TSV files found in directory: {folder.resolve()}")
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

def summarize_metrics_table(
    tsv_path, 
    out_dir, 
    comparison_metric="seasonalMeanAbsoluteScaledError"
):
    df = read_experiment_tsv(tsv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if comparison_metric not in df['metric_name'].unique():
        # Fallback to MeanAbsoluteScaledError if present
        fallback = "MeanAbsoluteScaledError"
        if fallback in df['metric_name'].unique():
            print(f"WARNING: {comparison_metric} not found, using {fallback} instead.")
            comparison_metric = fallback
        else:
            raise ValueError(f"Neither '{comparison_metric}' nor '{fallback}' found in 'metric_name' column.")

    # Filter for just the metric we want to compare on
    summary_df = df[df["metric_name"] == comparison_metric].copy()

    # Dynamic grouping columns
    group_cols = get_dynamic_group_columns(summary_df)
    print(f"Comparison axes: {group_cols}")

    # Pivot so columns = all combos of axes except dataset_name, rows = dataset_name, values = metric_value
    index = ["dataset_name"]
    col_index = [col for col in group_cols if col != "dataset_name"]
    pivot = summary_df.pivot_table(index=index, columns=col_index, values="metric_value", aggfunc="mean")

    # Save wide metric summary (rows: dataset, columns: configurations)
    pivot.to_csv(out_dir / "metric_summary_wide.tsv", sep="\t")

    # Compute mean sMASE across datasets for each configuration
    mean_sMASE = pivot.mean(axis=0).sort_values()
    print("\nMean sMASE Summary (lower is better):\n", mean_sMASE)

    mean_sMASE.to_csv(out_dir / "mean_summary.tsv", sep="\t")
    print(f"Saved mean summary to {out_dir / 'mean_summary.tsv'}")

    # Plot the means for visual inspection
    ax = mean_sMASE.plot(kind='barh', figsize=(10, 6), title=f'Mean {comparison_metric} per Setting')
    ax.set_xlabel(f'Mean {comparison_metric} (lower is better)')
    plt.tight_layout()
    plt.savefig(out_dir / f"mean_{comparison_metric}.png", bbox_inches='tight')
    plt.close()

def plot_predictions(tsv_path, out_dir):
    # Only if the CSV contains prediction values!
    df = pd.read_csv(tsv_path, sep=None, engine="python")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not set(["series_id", "model_name", "dataset_name"]).issubset(df.columns):
        print("TSV does not contain necessary columns for plotting predictions.")
        return

    # Example: if your TSV includes a column for the predicted series values as a list or stringified list, adapt here
    pred_col = None
    for c in df.columns:
        if "predict" in c.lower() or "forecast" in c.lower():
            pred_col = c
            break
    if pred_col is None:
        print("No column containing predictions found.")
        return

    for (ds, mdl), subdf in df.groupby(["dataset_name", "model_name"]):
        for idx, row in subdf.iterrows():
            y_pred = row[pred_col]
            # This assumes y_pred is a stringified list, e.g., "[1.0,2.1,3.4,...]"
            if isinstance(y_pred, str):
                try:
                    import ast
                    y_pred = ast.literal_eval(y_pred)
                except Exception as e:
                    print(f"Could not parse prediction for {ds}/{mdl}: {e}")
                    continue
            # Could also plot true values if available; adjust as needed
            plt.figure(figsize=(10, 4))
            plt.plot(y_pred, label="Prediction")
            plt.title(f"{ds} - {mdl}")
            plt.legend()
            plt.tight_layout()
            fname = f"{ds}_{mdl}_prediction.png"
            plt.savefig(out_dir / fname)
            plt.close()
    print(f"Prediction plots saved to {out_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Summarize and plot metrics from experiment TSV.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to experiment results TSV file")
    parser.add_argument("--output_folder", type=str, default=None, help="Directory where outputs will be saved")
    parser.add_argument("--summarize", action="store_true", help="Summarize metrics and create normalized comparisons")
    parser.add_argument("--plot-predictions", action="store_true", help="Plot prediction series (if available in TSV)")
    parser.add_argument("--comparison-metric", type=str, default="seasonalMeanAbsoluteScaledError", help="Metric for comparison (default: seasonalMeanAbsoluteScaledError)")

    args = parser.parse_args()

    output_folder = Path(args.output_folder) if args.output_folder else Path(args.input_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summarize_metrics_table(args.input_folder, output_folder, args.comparison_metric)
    if args.plot_predictions:
        plot_predictions(args.input_folder, output_folder)
    if not (args.summarize or args.plot_predictions):
        parser.print_help()

if __name__ == "__main__":
    main()
