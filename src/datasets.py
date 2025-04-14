from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import sys

import pyarrow as pa
import numpy as np

from src.datasets_assets.kernelsynth import generate_time_series

class BaseDataset(ABC):
    @abstractmethod
    def load(self) -> Any:
        """Load the dataset (e.g., DataFrame, Arrow table, list of dicts)."""
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the dataset."""
        pass

class NixtlaDataset(BaseDataset):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.path = Path(f"./datasets/{dataset_name}.arrow")

    def load(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        return pa.ipc.open_file(self.path).read_all()

    def metadata(self):
        return {
            "name": self.dataset_name,
            "type": "forecasting",
            "source": "nixtla",
        }

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

        # Determine base directory and file path
        base_dir = Path(__file__).parent / "synthetic"
        base_dir.mkdir(parents=True, exist_ok=True)

        self.file_path = (
            base_dir /
            f"kernel_synth_{num_series}_{max_kernels}_{sequence_lenght}.arrow"
        )

    def load(self):
        # Try loading if file exists
        if self.file_path.exists():
            print(f"Loading pre-generated dataset from {self.file_path}")
            return pa.ipc.open_file(self.file_path).read_all().to_pylist()

        # Otherwise, generate the dataset
        print("Generating synthetic dataset...")
        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(generate_time_series)(
                max_kernels=self.max_kernels,
                sequence_lenght=self.sequence_lenght
            )
            for _ in tqdm(range(self.num_series), desc="KernelSynth", leave=False)
        )

        # Save to file if enabled
        if self.save:
            print(f"Saving dataset to {self.file_path}")
            table = pa.Table.from_pylist(results)
            with self.file_path.open("wb") as f:
                with pa.ipc.new_file(f, table.schema) as writer:
                    writer.write_table(table)

        return results

    def metadata(self):
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

DATASET_REGISTRY = {
    "nixtla": NixtlaDataset,
    "kernelsynth": KernelSynthDataset,
}

def get_dataset(name: str, **kwargs) -> BaseDataset:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](**kwargs)
