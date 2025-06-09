# LLM Forecasting Experiment Framework
This framework is build to find preprocessors strategies on existing LLMs. Through configurations files (see experiments folder) either a single experiment can be executed or a multirun can be started with experiment parameters acting as a grid. Afterwards the data can be analyzed to see what how the model performed. More information is shown in the section Project Overview.

# Install requirements
The included dockerfile uses ROCm Pytorch as base image. Feel free to add a NVIDIA Cuda dockerfile.

## 1 · Helper Script — Fastest

```bash
./start_development_env.sh
```

- This script writes your UID/GID to `.env`
- Automatically launches the correct CPU or GPU Docker profile based on your hardware

---

## 2 · VS Code Dev-Container

1. Install the **Dev Containers** extension in VS Code.
2. Open the repository → **Reopen in Container**.
3. VS Code spins up the `dev-gpu` service as defined in `devcontainer.json` and runs `.devcontainer/setup.sh` for first-time setup.

---

## 3 · Docker Compose (Manual)

**CPU:**

```bash
docker compose --profile cpu run --rm dev-cpu
```

**GPU** *(requires `nvidia-docker` or ROCm)*:

```bash
docker compose --profile gpu run --rm dev-gpu
```

- Both profiles mount your working directory.
- They cache model/torch data in named volumes for faster re-runs.

---

## 4 · Pure-Python Install

```bash
python -m venv .venv && source .venv/bin/activate
# Install the exact Torch wheel you need (not included):
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

> **Note:** `requirements.txt` deliberately excludes PyTorch so you can choose the correct CUDA/CPU build.

## Used Sources and Inspirations

### Preprocessing

This project incorporates methods and code for data serialization and time series preprocessing from the following works:

#### LLMTime
- **Repository:** [https://github.com/ngruver/llmtime](https://github.com/ngruver/llmtime)
- **Code used:** `serialize.py` and associated serialization logic, adapted under the [MIT License](https://github.com/ngruver/llmtime/blob/main/LICENSE).
- **Paper:**  
  Gruver et al. (2023).  
  *Large Language Models Are Zero Shot Time Series Forecasters*.  
  [arXiv:2310.07820](https://arxiv.org/abs/2310.07820)
  
#### LLM-ABBA
- **Repository:** [https://github.com/inEXASCALE/llm-abba](https://github.com/inEXASCALE/llm-abba)
- **Paper**: 
  *LLM-ABBA: Understanding time series via symbolic approximation* (2024).  
  [arXiv:2411.18506v2](https://arxiv.org/pdf/2411.18506v2)

---

### Datasets

This project uses publicly available datasets for training and evaluation:

#### Kernel-Synth from Chronos Forecasting
- Code adapted from [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
- Licensed under the Apache License 2.0
- Paper: *Chronos: Learning the Language of Time Series*  
  [arXiv:2306.03893](https://arxiv.org/abs/2306.03893)

#### Nixtla Long Horizon Forecasting Dataset
- **Source:** [https://github.com/Nixtla/long-horizon-datasets](https://github.com/Nixtla/long-horizon-datasets)  
  Used for evaluating long-term forecasting performance across diverse domains.

#### UCR Anomaly Dataset
- **Source:** [https://www.cs.ucr.edu/~eamonn/discords/](https://www.cs.ucr.edu/~eamonn/discords/)  
  Employed for anomaly detection benchmarking in univariate time series.
  
  
# Project Overview

This project is a **time‑series forecasting benchmarking framework** built around modular factories.  Depending on whether you want to try one configuration or sweep many, you will interact with a single‑run script or the multirun driver. Results are collected in a uniform folder structure and can be post‑processed into tables and figures.

---

## 1 · Running Experiments

| Mode                | Entry point              | Typical call                                               | What happens                                                                                                                              |
| ------------------- | ------------------------ | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Single run**      | `run_experiment.py`      | `python -m src.run_experiment --config my_experiment.yaml` | Loads one YAML file, executes **dataset → preprocessor → model → analyzers**, and stores everything in `results/<experiment_name>/`.      |
| **Grid / multirun** | `multirun_experiment.py` | `python -m src.multirun_experiment --config sweep.yaml`    | Treats any `*_grid` fields as a Cartesian sweep and launches one run per combo.  Re‑uses model weights where possible to save GPU memory. |

Both scripts automatically timestamp the folder, record every run via a shared `ResultRecorder`, and emit a `results.jsonl` plus plots of predictions.

---

## 2 · Configuration (`config.py`)

A **single YAML file** drives either mode.  For each required axis you may supply a direct value *or* a grid variant (`*_grid`)—never both.

### Minimal single‑run example

```yaml
preprocessor_name: LLMTime                # encoder
model_name: llama3.1-8b                  # LLM alias
dataset_name: kernelsynth

input_data_length: 256                   # choose one of these
# input_data_factor: 0.8

dataset_params:
  num_series: 32
  max_kernels: 4

output_dir: results                      # optional overrides
experiment_name: "paper‑fig‑3"
model_parameters:
  max_new_tokens: 128
preprocessor_params:
  time_sep: ","
```

### Converting the same file into a sweep

```yaml
model_name_grid: ["llama3.1-8b", "smollm2-1.7b"]
input_data_factor_grid: [0.5, 0.8]
model_parameters_grid:
  - {max_new_tokens: 64}
  - {max_new_tokens: 128}
```

Validation rules enforced by `Config` (dataclass):

* **Exactly one** of `input_data_length{,_grid}` *or* `input_data_factor{,_grid}`
* All required axes present as value or grid
* `kernelsynth` requires `num_series` & `max_kernels`

---

## 3 · Datasets (`available_datasets.py`)

```python
dataset = build(cfg.dataset_name, DATASET_REGISTRY, **cfg.dataset_params)
series_iter = dataset.load()  # → list(dict(series=…, metadata=…))
```

| Key           | Loader               | Purpose                                                           |
| ------------- | -------------------- | ----------------------------------------------------------------- |
| `nixtla`      | `NixtlaDataset`      | Load Arrow files exported from Nixtla DB                          |
| `kernelsynth` | `KernelSynthDataset` | Synthetic kernel‑mix sequences (on‑the‑fly, cached)               |
| `darts`       | `DartsDataset`       | Wraps >16 classical sets from *darts* (AirPassengers, AusBeer, …) |

---

## 4 · Pre‑processing (`preprocessor.py`)

Choose an encoder from `PREPROCESSOR_REGISTRY` and call:

```python
encoded = preprocessor.encode(ts)
recovered = preprocessor.decode(encoded)
```

| Key        | Class                 | Notes                                                        |
| ---------- | --------------------- | ------------------------------------------------------------ |
| `LLM-ABBA` | `LLMABBAPreprocessor` | Symbolic ABBA encoding, tunable alphabet & separator         |
| `LLMTime`  | `LLMTimePreprocessor` | Numeric base‑*b* run‑length encoding from the LLMTimes paper |

---

## 5 · LLM Wrapper (`lmwrapper.py`)

`LMWrapper` is a thin helper around HuggingFace models:

* Downloads the checkpoint (or loads from cache)
* Applies a chat template if the model is *‑instruct*
* Restricts generation to numeric tokens to avoid textual hallucinations
* Provides `generate_response(prompt) → list[str]`

Extend via `MODEL_REGISTRY` by mapping a human‑friendly alias to a factory function that returns an `LMWrapper`.

---

## 6 · Metric Analyzers (`data_analyzers.py`)

Each analyzer implements `analyze(y_true, y_pred, training=None, seasonality=None)` and is called per batch.

| Key    | Class                         | Description                          |
| ------ | ----------------------------- | ------------------------------------ |
| `MAE`  | `MeanAbsoluteErrorAnalyzer`   | Mean absolute error                  |
| `MSE`  | `MeanSquareErrorAnalyzer`     | Mean squared error                   |
| `RMSE` | `RootMeanSquareErrorAnalyzer` | Square root of MSE                   |
| `MASE` | `MeanAbsoluteScaledError`     | Seasonal aware, needs training split |

Multiple analyzers can be requested at once via the `data_analyzers` field in the YAML.

---

## 7 · Creating Tables & Plots (`create_tables_and_plots.py`)

```bash
python -m src.create_tables_and_plots \
       --input_folder results/ \          # root experiment dir
       --summarize \                      # TSV → medians & spreads
       --plot_predictions \               # PNG overlay per series
       --plot_predictions_single llama3   # pretty grid for one model
```

Features

* Accepts a single `master_results.tsv` *or* a tree of run folders.
* Computes **per‑run means → per‑config medians (+MAD)** and writes three TSVs.
* Colours models via `COLOUR_ALIAS` and saves ready‑to‑embed PNG/PDF figures.

---

## 8 · End‑to‑End Workflow

1. **Write** a YAML config (single or with grids).
2. **Launch**

   * Single experiment: `python -m src.run_experiment …`
   * Sweep: `python -m src.multirun_experiment …`
3. **Post‑process** the `results/` directory with `create_tables_and_plots.py`.

