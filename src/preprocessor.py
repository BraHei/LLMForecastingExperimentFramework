import csv
import pickle
from abc import ABC, abstractmethod
import numpy as np
from fABBA import fABBA
from llmabba import ABBA as LLMABBA
from src.pretokenizer_assets.llmtime import serialize_arr, deserialize_str, SerializerSettings

# --- Abstract Base Class for Time Series Preprocessors ---

class BaseTimeSeriesPreprocessor(ABC):
    """
    Abstract base for time series tokenization/encoding.
    Handles basic I/O, saving, and interface enforcement.
    """

    def __init__(self):
        self.preprocessor = "BaseClass"
        self.encoder = None
        self.time_separator = ""

    @abstractmethod
    def encode(self, time_series):
        """Encode a time series into a string or symbol sequence."""
        pass

    @abstractmethod
    def decode(self, encoded_string):
        """Decode a string back into a numerical time series."""
        pass

    def save_encoded(self, encoded_list, output_path=None):
        """
        Save list of encoded series to a CSV file.
        Each row: [id, encoded_string]
        """
        if output_path is None:
            output_path = f"{self.preprocessor}_encoded_dataset.csv"
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "encoded_string"])
            for idx, code in enumerate(encoded_list):
                writer.writerow([idx, code])
        print(f"Encoded dataset saved to: {output_path}")

    def save_model(self, filepath=None):
        """
        Save the fitted encoder object using pickle.
        """
        if self.encoder is None:
            raise ValueError("Encoder must be fitted before saving.")
        if filepath is None:
            filepath = f"{self.preprocessor}_encoder.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self.encoder, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Encoder saved to {filepath}")

    def load_model(self, filepath=None):
        """
        Load an encoder object from a pickle file.
        """
        if filepath is None:
            filepath = f"{self.preprocessor}_encoder.pkl"
        with open(filepath, "rb") as f:
            encoder = pickle.load(f)
        self.encoder = encoder
        print(f"Encoder loaded from {filepath}")

# --- Symbol Sets Used for ABBA Encoding ---

ABBA_SYMBOL_LISTS= {
    "AlphabetAa": ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e',
                     'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j',
                     'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o',
                     'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't',
                     'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z'],
    "Alphabetab": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                     'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                     'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                     'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                     'W', 'X', 'Y', 'Z'],
    "AlphabetAB": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                     'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                     'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                     'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                     'w', 'x', 'y', 'z'],
    "Numbers": [str(i) for i in range(1001)],
    "Specials": [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>',
        '?', '@', 'A', 'B', 'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I' ,'J' ,'K' ,'L' ,'M' ,'N' ,'O' ,'P' ,'Q' ,'R' ,'S' ,'T' ,'U' ,'V' ,'W' ,'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b',
        'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '', ' ', '¢', '£', '§', '©',
        '«', '¬', '®', '°', '±', '²', '³', '´', 'µ', '·', '¹', 'º', '»', '¼', '½', '¾', '¿', 'Â', 'Ã', 'É', '×', 'ß', 'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì',
        'í', 'î', 'ï', 'ð', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'ā', 'ă', 'ć', 'č', 'ē', 'ę', 'ğ', 'ī', 'ı', 'ł', 'ń', 'ō', 'œ', 'ś', 'ş', 'š', 'ū', 'ž', 'ə', 'ɪ', 
        'ʻ', 'ʿ', 'ˈ', 'ː', 'Δ', 'ά', 'έ', 'ή', 'ί', 'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ο', 'π', 'ρ', 'ς', 'σ', 'τ', 'υ', 'φ', 'χ', 'ω', 'ό', 'ύ', 'ώ', 'В', 
        'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ы', 'ь', 'ю', 'я', 'і', 'ְ', 'ִ', 'ֶ', 'ַ', 'ָ', 'ֹ', 'ּ', 
        'א', 'ב', 'ד', 'ה', 'ו', 'י', 'ל', 'מ', 'נ', 'ר', 'ש', 'ת']
}

# --- LLM-ABBA Preprocessor ---

class LLMABBAPreprocessor(BaseTimeSeriesPreprocessor):
    """
    ABBA-style symbolic encoding using predefined alphabets.
    Suitable for symbolic regression or LLM input.
    """

    def __init__(self, separator=',', symbol_set="AlphabetAa", **encoder_params):
        super().__init__()
        self.preprocessor_type = "LLM-ABBA"
        self.seperator = separator
        self.encoder_params = encoder_params

        # Ensure separator is not in symbol list
        updated_set = ABBA_SYMBOL_LISTS[symbol_set]
        if separator in updated_set:
            updated_set.remove(separator)
        self.encoder_params["alphabet_set"] = updated_set

    def encode(self, time_series):
        """Convert time series into symbol sequence (comma-separated)."""
        time_series = np.asarray(time_series)
        self.encoder = LLMABBA(**self.encoder_params)
        encoded_array = self.encoder.encode([time_series.tolist()])[0]
        return self.seperator.join(encoded_array)

    def decode(self, encoded_string):
        """Decode comma-separated symbol string back to time series."""
        if self.encoder is not None:
            symbol_list = list(encoded_string.replace(self.seperator, ""))
            responds = self.encoder.decode([symbol_list])
            return responds[0]
        else:
            raise ValueError("Decoder requires a previously fitted encoder.")

# Defaul settings for LLaMa2 models as per paper. Basic is also default False, use these for current results
# Reference https://github.com/ngruver/llmtime/blob/f74234c43e06de78774d94c0974371a87b1c6971/models/llmtime.py#L23
# https://github.com/ngruver/llmtime/blob/main/experiments/run_darts.py
class LLMTimePreprocessor(BaseTimeSeriesPreprocessor):
    """
    Tokenizes time series using numerical base encoding (LLMTime).
    Supports quantization and serialization for LLM-friendly formatting.
    """

    def __init__(self, base=10, prec=3, bit_sep='', time_sep=',',
                 alpha=0.99, beta=0.3, basic=False, signed=True):
        super().__init__()
        self.preprocessor_type = "LLMTime"
        self.time_separator = time_sep
        self.settings = SerializerSettings()
        self.settings.base = base
        self.settings.prec = prec
        self.settings.bit_sep = bit_sep
        self.settings.time_sep = time_sep
        self.settings.signed = signed

        self.scalar_alpha = alpha
        self.scalar_beta = beta
        self.scalar_basic = basic
        self.scalar_q = None
        self.scalar_min_ = None

    def calculate_scaling_parameters(self, time_series):
        """
        Fit normalization parameters (q and optionally min) for quantization.
        """
        time_series = np.array(time_series)
        time_series = time_series[~np.isnan(time_series)]
        if self.scalar_basic:
            self.scalar_q = np.maximum(np.quantile(np.abs(time_series), self.scalar_alpha),.01)
        else:
            self.scalar_min_ = np.min(time_series) - self.scalar_beta * (np.max(time_series) - np.min(time_series))
            self.scalar_q = np.quantile(time_series - self.scalar_min_, self.scalar_alpha)
            if self.scalar_q == 0:
                self.scalar_q = 1

    def encode(self, time_series):
        """
        Normalize and encode time series into a structured string using serialization.
        """
        self.calculate_scaling_parameters(time_series)
        time_series = (time_series / self.scalar_q if self.scalar_basic
                       else (time_series - self.scalar_min_) / self.scalar_q)
        return serialize_arr(np.array(time_series), self.settings)

    def decode(self, encoded_string):
        """
        Decode structured string back into a numerical time series using inverse transform.
        """
        time_series = deserialize_str(encoded_string, self.settings, ignore_last=True)
        return (time_series * self.scalar_q if self.scalar_basic
                else (time_series * self.scalar_q) + self.scalar_min_)

# --- Preprocessor Registry ---

PREPROCESSOR_REGISTRY = {
    "LLM-ABBA": LLMABBAPreprocessor,
    "LLMTime": LLMTimePreprocessor,
}
