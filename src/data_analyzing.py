import numpy as np
from abc import ABC, abstractmethod


class BaseDataAnalyzer(ABC):
    def __init__(self):
        self.AnalyzerType = "BaseClass"

    @abstractmethod
    def Analyze(self, true, predict):
        pass

class MeanAbsoluteErrorAnalyzer(BaseDataAnalyzer):
    def __init__(self):
        super().__init__()
        self.AnalyzerType = "MeanAbsoluteErrorAnalyzer"

    def Analyze(self,  true, predict):
        true = np.array(true)
        predict = np.array(predict)
        return np.mean(np.abs(true - predict))


class MeanSquareErrorAnalyzer(BaseDataAnalyzer):
    def __init__(self):
        super().__init__()
        self.AnalyzerType = "MeanSquareErrorAnalyzer"

    def Analyze(self,  true, predict):
        true = np.array(true)
        predict = np.array(predict)
        return np.mean((true - predict) ** 2)

class RootMeanSquareErrorAnalyzer(BaseDataAnalyzer):
    def __init__(self):
        super().__init__()
        self.AnalyzerType = "RootMeanSquareErrorAnalyzer"

    def Analyze(self, time_series):
        true = np.array(true)
        predict = np.array(predict)
        return np.sqrt(np.mean((true - predict) ** 2))

DATA_ANALYZER_REGISTRY = {
    "MAE": MeanAbsoluteErrorAnalyzer,
    "MSE": MeanSquareErrorAnalyzer,
    "RMSE": RootMeanSquareErrorAnalyzer,
}

def get_data_analyzer(name: str, **kwargs) -> BaseDataAnalyzer:
    if name not in DATA_ANALYZER_REGISTRY:
        raise ValueError(f"Unknown data_analyzer: {name}. Available: {list(DATA_ANALYZER_REGISTRY.keys())}")
    return DATA_ANALYZER_REGISTRY[name](**kwargs)
