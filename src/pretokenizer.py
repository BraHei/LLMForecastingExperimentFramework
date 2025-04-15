import csv
import pickle
from abc import ABC, abstractmethod
import numpy as np
from fABBA import fABBA
from llmabba import ABBA as LLMABBA
from src.pretokenizer_assets.llmtime import serialize_arr, deserialize_str, SerializerSettings

class BaseTimeSeriesPreTokenizer(ABC):
    def __init__(self):
        self.tokenizer_type = "BaseClass"
        self.encoder = None

    @abstractmethod
    def encode(self, time_series):
        pass

    @abstractmethod
    def decode(self, encoded_string, reference_point):
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


class FABBAEncoder(BaseTimeSeriesPreTokenizer):
    def __init__(self, **encoder_params):
        super().__init__()
        self.tokenizer_type = "fABBA"
        self.encoder_params = encoder_params
        self.model_params = None

    def encode(self, time_series):
        self.encoder = fABBA(**self.encoder_params)
        return self.encoder.fit_transform(time_series)

    def decode(self, encoded_string, reference_point):
        if self.encoder is not None:
            return self.encoder.inverse_transform(encoded_string, start = reference_point)
        else:
            raise ValueError("Decoder requires a previously fitted encoder.")


# Can be improved later on for parallel processing
class LLMABBAEncoder(BaseTimeSeriesPreTokenizer):
    def __init__(self, **encoder_params):
        super().__init__()
        self.tokenizer_type = "LLM-ABBA"
        self.encoder_params = encoder_params

    def encode(self, time_series):
        time_series = np.asarray(time_series)

        self.encoder = LLMABBA(**self.encoder_params)
        encoded_array = self.encoder.encode([time_series.tolist()])[0]
        return ''.join(encoded_array)

    def decode(self, encoded_string, reference_point=None):
        if self.encoder is not None:
            symbol_list = list(encoded_string)
            responds = self.encoder.decode([symbol_list])
            return responds[0]
        else:
            raise ValueError("Decoder requires a previously fitted encoder.")



class LLMTimeEncoder(BaseTimeSeriesPreTokenizer):
    def __init__(self, settings=None):
        super().__init__()
        self.tokenizer_type = "LLMTime"
        self.settings = settings if settings is not None else SerializerSettings()

    def encode(self, time_series):
        return serialize_arr(np.array(time_series), self.settings)

    def decode(self, encoded_string, reference_point=None):
        return deserialize_str(encoded_string, self.settings)


PRETOKENIZER_REGISTRY = {
    "fABBA": FABBAEncoder,
    "LLM-ABBA": LLMABBAEncoder,
    "LLMTime": LLMTimeEncoder,
}

def get_pretokenizer(name: str, **kwargs) -> BaseTimeSeriesPreTokenizer:
    if name not in PRETOKENIZER_REGISTRY:
        raise ValueError(f"Unknown pretokenizer: {name}. Available: {list(PRETOKENIZER_REGISTRY.keys())}")
    return PRETOKENIZER_REGISTRY[name](**kwargs)

