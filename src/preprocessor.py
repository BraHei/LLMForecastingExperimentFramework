import csv
import pickle
from abc import ABC, abstractmethod
import numpy as np
from fABBA import fABBA
from llmabba import ABBA as LLMABBA
from src.pretokenizer_assets.llmtime import serialize_arr, deserialize_str, SerializerSettings

class BaseTimeSeriesPreprocessor(ABC):
    def __init__(self):
        self.preprocessor = "BaseClass"
        self.encoder = None
        self.time_seperator = ""

    @abstractmethod
    def encode(self, time_series):
        pass

    @abstractmethod
    def decode(self, encoded_string):
        pass

    def save_encoded(self, encoded_list, output_path=None):
        if output_path is None:
            output_path = f"{self.preprocessor}_encoded_dataset.csv"
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "encoded_string"])
            for idx, code in enumerate(encoded_list):
                writer.writerow([idx, code])
        print(f"Encoded dataset saved to: {output_path}")

    def save_model(self, filepath=None):
        if self.encoder is None:
            raise ValueError("Encoder must be fitted before saving.")
        if filepath is None:
            filepath = f"{self.preprocessor}_encoder.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self.encoder, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Encoder saved to {filepath}")

    def load_model(self, filepath=None):
        if filepath is None:
            filepath = f"{self.preprocessor}_encoder.pkl"
        with open(filepath, "rb") as f:
            encoder = pickle.load(f)
        self.encoder = encoder
        print(f"Encoder loaded from {filepath}")

class LLMABBAPreprocessor(BaseTimeSeriesPreprocessor):
    def __init__(self, **encoder_params):
        super().__init__()
        self.preprocessor_type = "LLM-ABBA"
        self.encoder_params = encoder_params

    def filter_symbols(self, encoded_string):
        allowed_set = set(self.encoder.string_[0])
        symbol_list = list(encoded_string)
        return [char for char in symbol_list if char in allowed_set]

    def encode(self, time_series):
        time_series = np.asarray(time_series)
        self.encoder = LLMABBA(**self.encoder_params)
        encoded_array = self.encoder.encode([time_series.tolist()])[0]
        return ''.join(encoded_array)

    def decode(self, encoded_string):
        if self.encoder is not None:
            filtered_list = self.filter_symbols(encoded_string)
            if (len(list(encoded_string)) != len(filtered_list)):
                print("WARNING: encoded string has reduced symbols")
            responds = self.encoder.decode([filtered_list])
            return responds[0]
        else:
            raise ValueError("Decoder requires a previously fitted encoder.")

# Can be improved later on for parallel processing
class LLMABBAEncoderSpaced(BaseTimeSeriesPreprocessor):
    def __init__(self, **encoder_params):
        super().__init__()
        self.preprocessor = "LLM-ABBA"
        self.encoder_params = encoder_params

    def encode(self, time_series):
        time_series = np.asarray(time_series)

        self.encoder = LLMABBA(**self.encoder_params)
        encoded_array = self.encoder.encode([time_series.tolist()])[0]
        return ' '.join(encoded_array)

    def decode(self, encoded_string):
        if self.encoder is not None:
            symbol_list = list(encoded_string.replace(" ", ""))
            responds = self.encoder.decode([symbol_list])
            return responds[0]
        else:
            raise ValueError("Decoder requires a previously fitted encoder.")

# Defaul settings for LLaMa2 models as per paper. Basic is also default False, use these for current results
# Reference https://github.com/ngruver/llmtime/blob/f74234c43e06de78774d94c0974371a87b1c6971/models/llmtime.py#L23
# https://github.com/ngruver/llmtime/blob/main/experiments/run_darts.py
class LLMTimePreprocessor(BaseTimeSeriesPreprocessor):
    def __init__(self, base = 10, prec = 3, bit_sep = '', time_sep = ', ', alpha = 0.99, beta = 0.3, basic = False, signed=True):
        super().__init__()
        self.preprocessor_type = "LLMTime"
        self.time_seperator = time_sep
        self.settings = SerializerSettings()
        self.settings.base = base
        self.settings.prec = prec   
        self.settings.bit_sep = bit_sep
        self.settings.time_sep = time_sep
        
        self.scalar_alpha = alpha
        self.scalar_beta = beta
        self.scalar_basic = False
        self.scalar_q = None
        self.scalar_min_ = None

    def calculate_scaling_parameters(self, time_series):
        time_series = np.array(time_series)
        time_series = time_series[~np.isnan(time_series)]
        if self.scalar_basic:
            self.scalar_q = np.maximum(np.quantile(np.abs(time_series), self.scalar_alpha),.01)
        else:
            self.scalar_min_ = np.min(time_series) - self.scalar_beta*(np.max(time_series)-np.min(time_series))
            self.scalar_q = np.quantile(time_series-self.scalar_min_, self.scalar_alpha)
            if self.scalar_q == 0:
                self.scalar_q = 1

    def encode(self, time_series):
        self.calculate_scaling_parameters(time_series)
        time_series = time_series / self.scalar_q if self.scalar_basic else (time_series - self.scalar_min_) / self.scalar_q
        return serialize_arr(np.array(time_series), self.settings)

    def decode(self, encoded_string):
        time_series = deserialize_str(encoded_string, self.settings, ignore_last = True)
        return time_series * self.scalar_q if self.scalar_basic else (time_series * self.scalar_q) + self.scalar_min_


PREPROCESSOR_REGISTRY = {
    "LLM-ABBA": LLMABBAPreprocessor,
    "LLM-ABBA_SPACED": LLMABBAEncoderSpaced,
    "LLMTime": LLMTimePreprocessor,
}
