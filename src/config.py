from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

@dataclass(slots=True)
class ExperimentConfig:
    """
    Typed representation of the YAML configuration.

    This class handles known configuration fields explicitly, while preserving unknown fields
    in the `extra` dictionary for forward compatibility. Includes validation logic and
    experiment name generation based on configuration values.
    """

    # --- Basic configuration fields (single value) ---
    preprocessor_name: Optional[str] = None
    model_name: Optional[Any] = None
    dataset_name: Optional[str] = None

    # --- Optional nested parameter dictionaries ---
    preprocessor_params: Dict[str, Any] = field(default_factory=dict)
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    dataset_params: Dict[str, Any] = field(default_factory=dict)
    input_data_length: Optional[Any] = None
    input_data_factor: Optional[Any] = None
    instruction_object: Optional[List[Dict[str, Any]]] = None  # Accepts list or single dict (normalized)

    # --- Grid configuration fields (for sweeps) ---
    model_name_grid: Optional[List[Any]] = None
    preprocessor_params_grid: Optional[Dict[str, List[Any]]] = None
    model_parameters_grid: Optional[List[Dict[str, Any]]] = None
    instruction_object_grid: Optional[List[Any]] = None
    input_data_factor_grid: Optional[List[Any]] = None

    # --- Miscellaneous experiment control fields ---
    data_analyzers: List[str] = field(default_factory=lambda: ["basic"])
    experiment_name: Optional[str] = ""
    output_dir: str = "results"
    build_experiment_name_flag: bool = True

    # --- Catch-all for unknown YAML fields ---
    extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        build_experiment_name_flag: bool = True
    ) -> ExperimentConfig:
        """
        Load and parse a YAML configuration file into an ExperimentConfig instance.

        Unknown keys are stored in `extra`. Handles backward compatibility and
        normalization of certain fields like `instruction_object`.
        """
        with open(path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        known = set(cls.__dataclass_fields__.keys())
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}

        for k, v in raw.items():
            if k == 'instruction_object':
                # Normalize instruction object to a list if needed
                if isinstance(v, dict):
                    kwargs[k] = [v]
                elif isinstance(v, list):
                    kwargs[k] = v
                else:
                    kwargs[k] = None  # Invalid type
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
        """
        Post-initialization validation and auto-generation of experiment name.
        Ensures required combinations of fields and validates single/grid exclusivity.
        """
        input_length_ok = (self.input_data_length is not None) or hasattr(self, 'input_data_length_grid')
        input_factor_ok = (self.input_data_factor is not None) or hasattr(self, 'input_data_factor_grid')
        if input_length_ok == input_factor_ok:
            raise ValueError(
                "Exactly one of 'input_data_length' or 'input_data_factor' (or their _grid variants) must be set."
            )

        # Ensure required fields are present, either as value or grid
        mandatory = [
            ("preprocessor_name", "preprocessor_name_grid"),
            ("model_name", "model_name_grid"),
            ("dataset_name", "dataset_name_grid"),
        ]
        missing = []
        for single, grid in mandatory:
            if getattr(self, single, None) is None and getattr(self, grid, None) is None:
                missing.append(single)
        if missing:
            raise ValueError(
                f"Missing required field(s): {', '.join(missing)} (need either the field or its _grid variant)"
            )

        if self.build_experiment_name_flag:
            self.experiment_name = self.build_experiment_name()

        self.output_dir = str(Path(self.output_dir) / self.experiment_name)

    def _post_init_validation(self, source: Union[str, Path]) -> None:
        """
        Additional validation for model and dataset parameters.
        E.g., ensures positive token limits and kernel-specific requirements.
        """
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
        """
        Construct a deterministic, human-readable experiment name from config fields.
        Includes model, preprocessor, optional instruction/dataset codes and a timestamp.
        """
        parts = [f"PR-{self.preprocessor_name}", f"M-{self.model_name}"]

        # Optional instruction name if present
        inst_code: Optional[str] = None
        if self.instruction_object:
            if isinstance(self.instruction_object, dict) and 'name' in self.instruction_object and self.instruction_object['name']:
                inst_code = self.instruction_object['name']
        if inst_code:
            parts.append(f"I-{inst_code}")

        # Separator abbreviation
        sep = self.preprocessor_params.get('time_sep')
        if sep:
            sep_code = str(sep).replace(' ', '').replace('/', 'SL').replace('\\', 'BSL')
            parts.append(f"S-{sep_code}")

        # Dataset ID code
        if self.dataset_name == "kernelsynth":
            ds_code = f"N{self.dataset_params.get('num_series', '?')}M{self.dataset_params.get('max_kernels', '?')}L{self.dataset_params.get('sequence_lenght', '?')}"
        elif self.dataset_name == 'darts':
            ds_names = self.dataset_params.get('dataset_names', [])
            ds_code = ''.join(n[:2] for n in ds_names)
        else:
            ds_code = self.dataset_name[:4]

        # Timestamp
        parts.append(f"{ds_code}-{time.strftime('%Y%m%d-%H%M%S')}")
        return '_'.join(parts)

    def save(self, filename: str = "experiment_config.yaml") -> None:
        """
        Serialize the current configuration (including any extra fields) to a YAML file.
        Output is placed inside the output_dir, which is created if necessary.
        """
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

# Convenience alias for loading configs
load_config = ExperimentConfig.from_yaml
