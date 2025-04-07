import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib.pyplot as plt


def load_arrow_data(file_path):
    with file_path.open("rb") as f:
        reader = ipc.RecordBatchFileReader(f)
        table = reader.read_all()
    return table


def plot_series(start_times, series_list, num_series=5):
    plt.figure(figsize=(12, 6))
    for idx, (start, target) in enumerate(zip(start_times, series_list[:num_series])):
        x = range(len(target))
        plt.plot(x, target, label=f"Series {idx} | Start: {start}")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.title(f"First {min(num_series, len(series_list))} Time Series from Arrow File")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    default_path = Path(__file__).parent / "kernelsynth-data.arrow"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arrow_file",
        nargs="?",
        default=default_path,
        type=Path,
        help=f"Path to the .arrow file (default: {default_path})"
    )
    parser.add_argument(
        "-n", "--num-series",
        type=int,
        default=5,
        help="Number of time series to visualize (default: 5)"
    )
    args = parser.parse_args()

    table = load_arrow_data(args.arrow_file)

    # Extract the "start" and "target" columns
    if {"start", "target"}.issubset(set(table.column_names)):
        start_times = table.column("start").to_pylist()
        target_series = table.column("target").to_pylist()
    else:
        raise ValueError("Arrow file must contain 'start' and 'target' fields.")

    plot_series(start_times, target_series, num_series=args.num_series)


if __name__ == "__main__":
    main()
