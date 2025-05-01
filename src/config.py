from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

@dataclass(slots=True)
class ExperimentConfig:
    """Typed representation of the YAML configuration.

    New fields can be added painlessly; unknown YAML keys are kept in
    ``extra`` so nothing is lost while you migrate.
    """

    # --- mandatory ------------------------------------------------------
    tokenizer_name: str
    model_name: str
    dataset_name: str
    input_data_length: int

    # --- optional / nested dicts ---------------------------------------
    tokenizer_params: Dict[str, Any] = field(default_factory=dict)
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    dataset_params: Dict[str, Any] = field(default_factory=dict)

    # --- misc -----------------------------------------------------------
    data_analyzers: List[str] = field(default_factory=lambda: ["basic"])
    experiment_name: Optional[str] = None
    output_dir: str = "results"
    seed: Optional[int] = None

    # catch-all for forward compatibility
    extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load *and* validate a YAML file into a config object."""
        with open(path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in raw.items():
            (kwargs if k in known_keys else extra)[k] = v

        cfg: ExperimentConfig = cls(**kwargs)  # type: ignore[arg-type]
        cfg.extra = extra
        cfg._post_init_validation(path)
        return cfg
    

    def __post_init__(self):

        if self.experiment_name == None:
            self.experiment_name = self.build_experiment_name()

        if self.output_dir == "results":
            self.output_dir = str(Path("results") / self.experiment_name)

    def _post_init_validation(self, source: str | Path) -> None:
        """Centralised sanity-checks so they’re not scattered across the code base."""
        
        if self.model_parameters.get("max_new_tokens", 0) <= 0:
            raise ValueError(
                f"{source}: 'model_parameters.max_new_tokens' must be positive"
            )
        if self.dataset_name == "kernelsynth":
            if "max_kernels" not in self.dataset_params:
                raise ValueError(
                    f"{source}: kernelsynth requires 'dataset_params.max_kernels'"
                )
            if "num_series" not in self.dataset_params:
                raise ValueError(
                    f"{source}: kernelsynth requires 'dataset_params.num_series'"
                )

    def build_experiment_name(self) -> str:
        """Deterministic but human-readable identifier that encodes the core setup."""
        base = (
            f"PTOK-{self.tokenizer_name}_"
            f"LLM-{self.model_name}_"
            f"NTOK{self.model_parameters.get('max_new_tokens', '?')}"
        )

        timestamp = time.strftime('%Y%m%d-%H%M%S')

        if self.dataset_name == "kernelsynth":
            return (
                f"{base}_DS-kernelsynth_"
                f"NSER{self.dataset_params.get('num_series', '?')}"
                f"MKER{self.dataset_params.get('max_kernels', '?')}_"
                f"SLEN{self.dataset_params.get('sequence_lenght', '?')}"
                f"-{timestamp}"
            )
        if self.dataset_name == "darts":
            datasets = ",".join(self.dataset_params.get("dataset_names", []))
            return f"{base}_DS-darts_{datasets}-{timestamp}"


        return f"{base}_DS-{self.dataset_name}-{timestamp}"

    def save(self, filename: str = "experiment_config.yaml") -> None:
        """Save the current configuration (including extra fields) to a YAML file."""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / filename

        # Combine known fields and extra fields
        full_config = {
            **{f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values() if f.name != "extra"},
            **self.extra,
        }

        with open(save_path, "w") as f:
            yaml.safe_dump(full_config, f, sort_keys=False)

# Helper that older code can import instead of touching YAML directly
load_config = ExperimentConfig.from_yaml
