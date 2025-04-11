# MasterThesis
Information, source code and more for the Master Thesis


# Install requirements

The requirements do not contain Pytorch. This has to be installed separately. The included dockerfile uses ROCm Pytorch as base image, change accordingly.

To run use the following commands:
```
docker compose run --build --rm dev
```

## Used Sources and Inspirations

### Preprocessing & Tokenization

This project incorporates methods and code for data serialization and time series preprocessing from the following works:

#### LLMTime
- **Repository:** [https://github.com/ngruver/llmtime](https://github.com/ngruver/llmtime)
- **Code used:** `serialize.py` and associated serialization logic, adapted under the [MIT License](https://github.com/ngruver/llmtime/blob/main/LICENSE).
- **Paper:**  
  Gruver et al. (2023).  
  *LLMTime: Benchmarking Language Models on Temporal Understanding*.  
  [arXiv:2310.07820](https://arxiv.org/abs/2310.07820)

#### fABBA
- **Repository:** [https://github.com/nla-group/fABBA](https://github.com/nla-group/fABBA)
- **Code used:** fABBA segmentation method for symbolic representation.
- **Paper:**  
  Lemire et al. (2020).  
  *fABBA: Fast Adaptive Bivariate Piecewise Approximation of Time Series*.  
  [arXiv:2003.12469](https://arxiv.org/pdf/2003.12469)

#### Temporal Reasoning Inspiration
- **Conceptual framework** influenced by:  
  *Language Models are Temporal Reasoners* (2024).  
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
