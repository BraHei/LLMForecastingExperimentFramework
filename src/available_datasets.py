from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from darts.datasets import (
    AirPassengersDataset, AusBeerDataset, EnergyDataset, HeartRateDataset, IceCreamHeaterDataset,
    MonthlyMilkDataset, SunspotsDataset, TemperatureDataset, AustralianTourismDataset,
    WeatherDataset, WoolyDataset, GasRateCO2Dataset, ETTh1Dataset, ETTh2Dataset,
    ETTm1Dataset, ETTm2Dataset
)
import pyarrow as pa

from src.datasets_assets.kernelsynth import generate_time_series

# --- Abstract base class for datasets ---

class BaseDataset(ABC):
    @abstractmethod
    def load(self) -> Any:
        """Load the dataset (returns list of dicts or similar)."""
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the dataset (name, type, etc.)."""
        pass

# --- Darts Dataset Configuration ---

# Mapping of dataset names to their respective Darts classes
DARTS_DATASET_CLASSES = {
    "AirPassengers": AirPassengersDataset,
    "AusBeer": AusBeerDataset,
    "Energy": EnergyDataset,
    "HeartRate": HeartRateDataset,
    "IceCreamHeater": IceCreamHeaterDataset,
    "MonthlyMilk": MonthlyMilkDataset,
    "Sunspots": SunspotsDataset,
    "Temperature": TemperatureDataset,
    "AustralianTourism": AustralianTourismDataset,
    "Weather": WeatherDataset,
    "Wooly": WoolyDataset,
    "GasRateCO2": GasRateCO2Dataset,
    "ETTh1": ETTh1Dataset,
    "ETTh2": ETTh2Dataset,
    "ETTm1": ETTm1Dataset,
    "ETTm2": ETTm2Dataset,
}

# Default seasonalities for known Darts datasets
DARTS_DATASET_SEASONALITY = {
    "AirPassengers": 12,
    "AusBeer": 4,
    "Energy": 24,
    "HeartRate": 1,
    "IceCreamHeater": 12,
    "MonthlyMilk": 12,
    "Sunspots": 132,  # 11-year solar cycle, monthly data
    "Temperature": 12,
    "AustralianTourism": 12,
    "Weather": 24,  # 10 minutes to day = 144
    "Wooly": 12,
    "GasRateCO2": 1, #unknown
    "ETTh1": 24,
    "ETTh2": 24,
    "ETTm1": 96,
    "ETTm2": 96,
}

# --- Darts Dataset Wrapper ---

class DartsDataset(BaseDataset):
    def __init__(self, dataset_names: List[str]):
        # Accept string or list of dataset names
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        self.dataset_names = dataset_names

        # Validate names
        invalid = [ds for ds in dataset_names if ds not in DARTS_DATASET_CLASSES]
        if invalid:
            raise ValueError(
                f"Invalid Darts dataset(s): {invalid}. "
                f"Available: {list(DARTS_DATASET_CLASSES.keys())}"
            )

    def load(self):
        """
        Load and return a list of standardized time series dictionaries.
        Handles both univariate and multivariate datasets.
        """
        series_dicts = []

        for name in self.dataset_names:
            dataset_cls = DARTS_DATASET_CLASSES[name]
            dataset = dataset_cls()
            data = dataset.load()

            # Handle multivariate series by splitting into components
            if data.n_components and data.n_components > 1:
                for i, component in enumerate(data.components):
                    univariate_ts = data[component]
                    series_dicts.append({
                        "series": univariate_ts.values().squeeze().tolist(),
                        "known_covariates": None,
                        "metadata": {
                            "source": "darts",
                            "dataset_name": f"{name}_{i}",
                            "component": component,
                            "start": str(univariate_ts.start_time()) if univariate_ts.has_datetime_index else None,
                            "freq": univariate_ts.freq_str if univariate_ts.has_datetime_index else None,
                            "length": len(univariate_ts),
                            "seasonality": DARTS_DATASET_SEASONALITY.get(name, 1)
                        }
                    })
            else:
                # Handle univariate dataset
                series_dicts.append({
                    "series": data.values().squeeze().tolist(),
                    "known_covariates": None,
                    "metadata": {
                        "source": "darts",
                        "dataset_name": name,
                        "component": data.components[0],
                        "start": str(data.start_time()) if data.has_datetime_index else None,
                        "freq": data.freq_str if data.has_datetime_index else None,
                        "length": len(data),
                        "seasonality": DARTS_DATASET_SEASONALITY.get(name, 1)
                    }
                })
        return series_dicts

    def metadata(self):
        """Return general metadata for the dataset collection."""
        return {
            "name": ",".join(self.dataset_names),
            "type": "forecasting",
            "source": "darts",
        }

# --- KernelSynth Synthetic Dataset ---

class KernelSynthDataset(BaseDataset):
    def __init__(
        self,
        num_series: int = 100,
        max_kernels: int = 5,
        sequence_lenght: int = 1024,
        n_jobs: int = -1,
        save: bool = True
    ):
        self.num_series = num_series
        self.max_kernels = max_kernels
        self.sequence_lenght = sequence_lenght
        self.n_jobs = n_jobs
        self.save = save

        # Setup file path for optional caching
        base_dir = Path(__file__).parent / "synthetic"
        base_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = base_dir / f"kernel_synth_{num_series}_{max_kernels}_{sequence_lenght}.arrow"

    def load(self):
        """
        Load synthetic dataset from disk if available, else generate and optionally save.
        Uses joblib for parallel generation of multiple series.
        """
        if self.file_path.exists():
            print(f"Loading pre-generated dataset from {self.file_path}")
            raw_data = pa.ipc.open_file(self.file_path).read_all().to_pylist()
        else:
            print("Generating synthetic dataset...")
            raw_data = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(generate_time_series)(
                    max_kernels=self.max_kernels,
                    sequence_lenght=self.sequence_lenght
                )
                for _ in tqdm(range(self.num_series), desc="KernelSynth", leave=False)
            )

            # Save to disk in Arrow format
            if self.save:
                print(f"Saving dataset to {self.file_path}")
                table = pa.Table.from_pylist(raw_data)
                with self.file_path.open("wb") as f:
                    with pa.ipc.new_file(f, table.schema) as writer:
                        writer.write_table(table)

        # Standardized format for returned series
        return [
            {
                "series": item["target"],
                "known_covariates": None,
                "metadata": {
                    "source": "kernelsynth",
                    "dataset_name": idx,
                    "generator": {
                        "amplitudes": item.get("amplitudes"),
                        "frequencies": item.get("frequencies"),
                        "phases": item.get("phases"),
                    },
                    "params": {
                        "num_series": self.num_series,
                        "max_kernels": self.max_kernels,
                        "sequence_length": self.sequence_lenght,
                    },
                    "seasonality": 1
                }
            }
            for idx, item in enumerate(raw_data)
        ]

    def metadata(self):
        """Return metadata including generation parameters for the synthetic dataset."""
        return {
            "name": "kernel-synth",
            "type": "synthetic",
            "task": "forecasting",
            "params": {
                "num_series": self.num_series,
                "max_kernels": self.max_kernels,
                "sequence_lenght": self.sequence_lenght,
                "n_jobs": self.n_jobs,
                "save": self.save
            },
        }

# --- Dataset registry for dispatch ---
DATASET_REGISTRY = {
    "kernelsynth": KernelSynthDataset,
    "darts": DartsDataset,
}
