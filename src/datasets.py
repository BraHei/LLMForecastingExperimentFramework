from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import pyarrow as pa
import numpy as np

from datasets.kernelsynth import generate_time_series

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
    def __init__(self, num_series: int = 100, max_kernels: int = 5, sequence_length: int = 1024, n_jobs: int = -1):
        self.num_series = num_series
        self.max_kernels = max_kernels
        self.sequence_lenght = sequence_length
        self.n_jobs = n_jobs

    def load(self) -> List[Dict]:
        return Parallel(n_jobs=self.n_jobs)(
            delayed(generate_time_series)(max_kernels=self.max_kernels, sequence_length=self.sequence_lenght)
            for _ in tqdm(range(self.num_series), desc="KernelSynth")
        )

    def metadata(self):
        return {
            "name": "kernel-synth",
            "type": "synthetic",
            "task": "forecasting",
            "params": {
                "num_series": self.num_series,
                "max_kernels": self.max_kernels,
                "sequence_lenght": self.sequence_lenght,
                "n_jobs": self.n_jobs
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
