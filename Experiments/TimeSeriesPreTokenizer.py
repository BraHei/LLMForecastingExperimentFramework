import csv
import pickle
import numpy as np
from fABBA import fABBA

class BaseTimeSeriesPreTokenizer:
    def __init__(self):
        self.tokenizer_type = "BaseClass"
        self.encoder = None

    def encode(self, time_series):
        raise NotImplementedError("Subclasses must implement 'encode' method.")

    def decode(self, encoded_string, reference_point):
        raise NotImplementedError("Subclasses must implement 'decode' method.")

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

    def encode(self, time_series):
        self.encoder = fABBA(**self.encoder_params)
        return self.encoder.fit_transform(time_series)

    def decode(self, encoded_string, reference_point):
        if self.encoder is not None:
            return self.encoder.inverse_transform(encoded_string, reference_point)
        else:
            raise ValueError("Decoder requires a previously fitted encoder.")

# Example usage:
# encoder = FABBAEncoder(tol=0.1, alpha=0.1, sorting='2-norm', scl=1, verbose=0)
# encoded = encoder.encode(some_time_series)
# encoder.save_model()
# encoder.load_model()
# decoded = encoder.decode(encoded_string, reference_point=some_time_series[0])