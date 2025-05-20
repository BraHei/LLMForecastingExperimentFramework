import numpy as np
from abc import ABC, abstractmethod


class BaseDataAnalyzer(ABC):
    def __init__(self):
        self.AnalyzerType = "BaseClass"

    @abstractmethod
    def Analyze(self, true, predict, training = None, seasonality = None):
        pass

class MeanAbsoluteErrorAnalyzer(BaseDataAnalyzer):
    def __init__(self):
        super().__init__()
        self.AnalyzerType = "MeanAbsoluteError"

    def Analyze(self, true, predict, training = None, seasonality = None):
        true = np.array(true)
        predict = np.array(predict)
        return np.mean(np.abs(true - predict))

class MeanSquareErrorAnalyzer(BaseDataAnalyzer):
    def __init__(self):
        super().__init__()
        self.AnalyzerType = "MeanSquareError"

    def Analyze(self, true, predict, training = None, seasonality = None):
        true = np.array(true)
        predict = np.array(predict)
        return np.mean((true - predict) ** 2)

class RootMeanSquareErrorAnalyzer(BaseDataAnalyzer):
    def __init__(self):
        super().__init__()
        self.AnalyzerType = "RootMeanSquareError"

    def Analyze(self, true, predict, training = None, seasonality = None):
        true = np.array(true)
        predict = np.array(predict)
        return np.sqrt(np.mean((true - predict) ** 2))

class MeanAbsoluteScaledError(BaseDataAnalyzer):
    def __init__(self):
        super().__init__()
        self.AnalyzerType = "MeanAbsoluteScaledError"

    #https://github.com/ServiceNow/N-BEATS/blob/c746a4f13ffc957487e0c3279b182c3030836053/common/metrics.py#L24
    def Analyze(self, true, predict, training=None, seasonality=1):
        if (seasonality != 1):
            self.AnalyzerType = "seasonalMeanAbsoluteScaledError"

        if training is None:
            print("WARNING: inserted data does not contain trianing data required for MASE, skipping.")
            return float("nan")
        
        # Convert inputs to numpy arrays
        true = np.array(true)
        predict = np.array(predict)
        training = np.array(training)

        # Forecast error
        forecast_error = np.mean(np.abs(predict - true))

        # Naive forecast error (scaling term)
        if seasonality >= len(training):
            raise ValueError("Seasonality must be less than the length of the training data.")
        
        naive_errors = np.abs(training[seasonality:] - training[:-seasonality])
        scale = np.mean(naive_errors)

        return forecast_error / scale if scale != 0 else np.inf

DATA_ANALYZER_REGISTRY = {
    "MAE": MeanAbsoluteErrorAnalyzer,
    "MSE": MeanSquareErrorAnalyzer,
    "RMSE": RootMeanSquareErrorAnalyzer,
    "MASE": MeanAbsoluteScaledError,
}
