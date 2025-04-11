import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.ipc as ipc
from fABBA import fABBA

def read_all_timeseries(file_path):
    """
    Reads all time series from the 'target' column in an Arrow file.

    Parameters
    ----------
    file_path : str or Path
        Path to the .arrow file.

    Returns
    -------
    list of np.ndarray
        List of time series.
    """
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        reader = ipc.RecordBatchFileReader(f)
        table = reader.read_all()

    if "target" in table.column_names:
        target_column = table.column("target")
        timeseries_list = [np.array(row.as_py()) for row in target_column]
        return timeseries_list
    else:
        raise ValueError("Arrow file must contain 'target' field.")


def save_encoded_dataset(encoded_list, output_path="encoded_dataset.csv"):
    """
    Saves the encoded fABBA strings to a CSV file.

    Parameters
    ----------
    encoded_list : list of str
        The encoded symbolic strings.
    output_path : str
        Path to the output CSV file.
    """
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "encoded_string"])
        for idx, code in enumerate(encoded_list):
            writer.writerow([idx, code])
    print(f"Encoded dataset saved to: {output_path}")


def encode_all_timeseries(ts_list, tol=0.1, alpha=0.1):
    """
    Encodes each time series using fABBA and returns a list of encoded strings.

    Parameters
    ----------
    ts_list : list of np.ndarray
        List of time series.
    tol : float
        fABBA tolerance parameter.
    alpha : float
        fABBA alpha parameter.

    Returns
    -------
    list of str
        Encoded symbolic strings.
    """
    fabba = fABBA(tol=tol, alpha=alpha, sorting='2-norm', scl=1, verbose=0)
    return [fabba.fit_transform(ts) for ts in ts_list]


def main():
    file_path = "kernelsynth-data_n200_k5.arrow"  # replace with actual path
    ts_list = read_all_timeseries(file_path)
    encoded_dataset = encode_all_timeseries(ts_list)

    save_encoded_dataset(encoded_dataset)  # <-- new line here

    for i, code in enumerate(encoded_dataset[:5]):  # just show a sample
        print(f"Time Series {i}: {code}")

    # Optional: visualize one reconstruction
    fabba = fABBA(tol=0.1, alpha=0.1, sorting='2-norm', scl=1, verbose=0)
    sample_index = 0
    fabba.fit_transform(ts_list[sample_index])
    reconstructed = fabba.inverse_transform(encoded_dataset[sample_index], ts_list[sample_index][0])

    plt.plot(ts_list[sample_index], label='original')
    plt.plot(reconstructed, label='reconstruction')
    plt.legend()
    plt.grid(True, axis='y')
    plt.show()
    plt.savefig("reconstruction.png")



if __name__ == "__main__":
    main()

