import numpy as np
from abc import ABC, abstractmethod

# --- Abstract Base Class for Analyzers ---

class BaseDataAnalyzer(ABC):
    def __init__(self):
        self.AnalyzerType = "BaseClass"

    @abstractmethod
    def Analyze(self, true, predict, training=None, seasonality=None):
        """Compute error metric given true and predicted values.
        
        Args:
            true: Ground truth series.
            predict: Forecasted series.
            training: Optional historical training data (for scale-based metrics).
            seasonality: Optional seasonal period (for MASE).
        """
        pass

# --- Concrete Implementations ---

class MeanAbsoluteErrorAnalyzer(BaseDataAnalyzer):
    def __init__(self):
        super().__init__()
        self.AnalyzerType = "MeanAbsoluteError"

    def Analyze(self, true, predict, training=None, seasonality=None):
        """Mean Absolute Error (MAE)."""
        true = np.array(true)
        predict = np.array(predict)
        return np.mean(np.abs(true - predict))

class MeanSquareErrorAnalyzer(BaseDataAnalyzer):
    def __init__(self):
        super().__init__()
        self.AnalyzerType = "MeanSquareError"

    def Analyze(self, true, predict, training=None, seasonality=None):
        """Mean Squared Error (MSE)."""
        true = np.array(true)
        predict = np.array(predict)
        return np.mean((true - predict) ** 2)

class RootMeanSquareErrorAnalyzer(BaseDataAnalyzer):
    def __init__(self):
        super().__init__()
        self.AnalyzerType = "RootMeanSquareError"

    def Analyze(self, true, predict, training=None, seasonality=None):
        """Root Mean Squared Error (RMSE)."""
        true = np.array(true)
        predict = np.array(predict)
        return np.sqrt(np.mean((true - predict) ** 2))

class MeanAbsoluteScaledError(BaseDataAnalyzer):
    def __init__(self):
        super().__init__()
        self.AnalyzerType = "MeanAbsoluteScaledError"

    def Analyze(self, true, predict, training=None, seasonality=1):
        """
        Mean Absolute Scaled Error (MASE). 
        Compares forecast to seasonal naive baseline.

        Requires:
            - training data (historical series)
            - seasonality period (integer > 0)
        """
        if seasonality != 1:
            self.AnalyzerType = "seasonalMeanAbsoluteScaledError"

        if training is None:
            print("WARNING: inserted data does not contain training data required for MASE, skipping.")
            return float("nan")

        true = np.array(true)
        predict = np.array(predict)
        training = np.array(training)

        # Forecast absolute error
        forecast_error = np.mean(np.abs(predict - true))

        # Naive seasonal forecast error (scaling term)
        if seasonality >= len(training):
            raise ValueError("Seasonality must be less than the length of the training data.")

        naive_errors = np.abs(training[:-seasonality] - training[seasonality:])
        scale = np.mean(naive_errors)

        return forecast_error / scale if scale != 0 else np.inf

# --- Registry of Available Analyzers ---

DATA_ANALYZER_REGISTRY = {
    "MAE": MeanAbsoluteErrorAnalyzer,
    "MSE": MeanSquareErrorAnalyzer,
    "RMSE": RootMeanSquareErrorAnalyzer,
    "MASE": MeanAbsoluteScaledError,
}
