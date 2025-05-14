from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

@dataclass(slots=True)
class ExperimentConfig:
    """Typed representation of the YAML configuration.

    Unknown YAML keys are kept in `extra` for forward compatibility.
    """

    # --- mandatory ------------------------------------------------------
    preprocessor_name: str
    model_name: Any
    dataset_name: str

    # --- optional / nested dicts ---------------------------------------
    preprocessor_params: Dict[str, Any] = field(default_factory=dict)
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    dataset_params: Dict[str, Any] = field(default_factory=dict)
    input_data_length: Optional[Any] = None
    input_data_factor: Optional[Any] = None
    # instruction_object is optional; can be a single dict or list of dicts
    instruction_object: Optional[List[Dict[str, Any]]] = None

    # --- misc -----------------------------------------------------------
    data_analyzers: List[str] = field(default_factory=lambda: ["basic"])
    experiment_name: Optional[str] = ""
    output_dir: str = "results"
    seed: Optional[int] = None
    build_experiment_name_flag: bool = True

    # catch-all for forward compatibility
    extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        build_experiment_name_flag: bool = True
    ) -> ExperimentConfig:
        """Load and validate a YAML file into a config object."""
        with open(path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        known = set(cls.__dataclass_fields__.keys())
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}

        for k, v in raw.items():
            if k == 'instruction_object':
                # normalize to list if dict provided
                if isinstance(v, dict):
                    kwargs[k] = [v]
                elif isinstance(v, list):
                    kwargs[k] = v
                else:
                    # invalid type, ignore
                    kwargs[k] = None
            elif k in known:
                kwargs[k] = v
            else:
                extra[k] = v

        kwargs['build_experiment_name_flag'] = build_experiment_name_flag
        cfg = cls(**kwargs)  # type: ignore[arg-type]
        cfg.extra = extra
        cfg._post_init_validation(path)
        return cfg

    def __post_init__(self):
        # exactly one of length or factor must be set
        if (self.input_data_length is None) == (self.input_data_factor is None):
            raise ValueError(
                "Exactly one of 'input_data_length' or 'input_data_factor' must be set."
            )
        if self.build_experiment_name_flag:
            self.experiment_name = self.build_experiment_name()
        self.output_dir = str(Path(self.output_dir) / self.experiment_name)

    def _post_init_validation(self, source: Union[str, Path]) -> None:
        if self.model_parameters.get("max_new_tokens", 1) <= 0:
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
        """Deterministic, human-readable identifier encoding core setup."""
        parts = [f"PR-{self.preprocessor_name}", f"M-{self.model_name}"]

        # Instruction shorthand if present
        inst_code: Optional[str] = None
        if self.instruction_object:
            first = self.instruction_object[0]
            if isinstance(first, dict) and 'name' in first and first['name']:
                inst_code = first['name']
        if inst_code:
            parts.append(f"I-{inst_code}")

        # Separator code if present
        sep = self.preprocessor_params.get('time_sep')
        if sep:
            sep_code = str(sep).replace(' ', '').replace('/', 'SL').replace('\\', 'BSL')
            parts.append(f"S-{sep_code}")

        # Dataset code
        if self.dataset_name == "kernelsynth":
            ds_code = f"N{self.dataset_params.get('num_series', '?')}M{self.dataset_params.get('max_kernels', '?')}L{self.dataset_params.get('sequence_lenght', '?')}"
        elif self.dataset_name == 'darts':
            ds_names = self.dataset_params.get('dataset_names', [])
            ds_code = ''.join(n[:2] for n in ds_names)
        else:
            ds_code = self.dataset_name[:4]
        parts.append(f"{ds_code}-{time.strftime('%Y%m%d-%H%M%S')}")

        return '_'.join(parts)

    def save(self, filename: str = "experiment_config.yaml") -> None:
        """Save the current configuration (including extra fields) to a YAML file."""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / filename

        full: Dict[str, Any] = {}
        for field_name in self.__dataclass_fields__:
            if field_name == 'extra':
                continue
            value = getattr(self, field_name)
            if value is not None:
                full[field_name] = value
        full.update(self.extra)

        with open(save_path, 'w') as f:
            yaml.safe_dump(full, f, sort_keys=False)

# Helper for backward compatibility
load_config = ExperimentConfig.from_yaml
