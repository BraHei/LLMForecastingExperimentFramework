import csv
import pickle
from abc import ABC, abstractmethod
import numpy as np
from fABBA import fABBA
from llmabba import ABBA as LLMABBA
from src.pretokenizer_assets.llmtime import serialize_arr, deserialize_str, SerializerSettings

class BaseTimeSeriesPreprocessor(ABC):
    def __init__(self):
        self.tokenizer_type = "BaseClass"
        self.encoder = None

    @abstractmethod
    def encode(self, time_series):
        pass

    @abstractmethod
    def decode(self, encoded_string):
        pass

    def save_encoded(self, encoded_list, output_path=None):
        if output_path is None:
            output_path = f"{self.tokenizer_type}_encoded_dataset.csv"
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
            filepath = f"{self.tokenizer_type}_encoder.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self.encoder, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Encoder saved to {filepath}")

    def load_model(self, filepath=None):
        if filepath is None:
            filepath = f"{self.tokenizer_type}_encoder.pkl"
        with open(filepath, "rb") as f:
            encoder = pickle.load(f)
        self.encoder = encoder
        print(f"Encoder loaded from {filepath}")


# class FABBAPreprocessor(BaseTimeSeriesPreprocessor):
#     def __init__(self, **encoder_params):
#         super().__init__()
#         self.tokenizer_type = "fABBA"
#         self.encoder_params = encoder_params
#         self.model_params = None

#     def encode(self, time_series):
#         self.encoder = fABBA(**self.encoder_params)
#         return self.encoder.fit_transform(time_series)

#     def decode(self, encoded_string, reference_point):
#         if self.encoder is not None:
#             return self.encoder.inverse_transform(encoded_string, start = reference_point)
#         else:
#             raise ValueError("Decoder requires a previously fitted encoder.")


# Can be improved later on for parallel processing
class LLMABBAPreprocessor(BaseTimeSeriesPreprocessor):
    def __init__(self, **encoder_params):
        super().__init__()
        self.tokenizer_type = "LLM-ABBA"
        self.encoder_params = encoder_params

    def encode(self, time_series):
        time_series = np.asarray(time_series)

        self.encoder = LLMABBA(**self.encoder_params)
        encoded_array = self.encoder.encode([time_series.tolist()])[0]
        return ''.join(encoded_array)

    def decode(self, encoded_string):
        if self.encoder is not None:
            symbol_list = list(encoded_string)
            responds = self.encoder.decode([symbol_list])
            return responds[0]
        else:
            raise ValueError("Decoder requires a previously fitted encoder.")

# Defaul settings for LLaMa2 models as per paper. Basic is also default False, use these for current results
# Reference https://github.com/ngruver/llmtime/blob/f74234c43e06de78774d94c0974371a87b1c6971/models/llmtime.py#L23
class LLMTimePreprocessor(BaseTimeSeriesPreprocessor):
    def __init__(self, settings=None, alpha = 0.99, beta = 0.3, basic = False):
        super().__init__()
        self.tokenizer_type = "LLMTime"
        self.settings = None
        if settings is not None:
            self.settings = settings
        else: 
            self.settings = SerializerSettings()
            self.settings.bit_sep = ''
            self.settings.time_sep = ','
        
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
        time_series = deserialize_str(encoded_string, self.settings, True)
        return time_series * self.scalar_q if self.scalar_basic else (time_series * self.scalar_q) + self.scalar_min_


PRETOKENIZER_REGISTRY = {
    "LLM-ABBA": LLMABBAPreprocessor,
    "LLMTime": LLMTimePreprocessor,
}
