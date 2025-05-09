from __future__ import annotations

from typing import Dict, List, Tuple

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = [
    "LMWrapper",
    "MODEL_REGISTRY",
    "get_model",
]


class LMWrapper:
    def __init__(self, 
                 checkpoint: str,
                 device: str = None,
                 max_new_tokens: int = 250,
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 use_auth_token: bool = False,
                 access_token: str = None,
                 truncate_if_exceeds: bool = True,
                 do_sample: bool = False,
                 repetition_penalty: float = 1.0,
                 use_bnb_int8: bool = False):
        
        if use_auth_token:
            access_token = access_token or os.getenv("HF_ACCESS_TOKEN", None)
        else:
            access_token = None

        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.truncate_if_exceeds = truncate_if_exceeds
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            use_auth_token=access_token if use_auth_token else None
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        quant_cfg = None
        # if use_bnb_int8: Disable untill bitsandbytes is supported for ROCm 6.4.0
        #     quant_cfg = BitsAndBytesConfig(
        #         load_in_8bit=True,
        #         llm_int8_threshold=6.0,
        #     )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            quantization_config=quant_cfg,
            device_map="auto",
            use_auth_token=access_token if use_auth_token else None,
            torch_dtype=torch.float16 if quant_cfg is None else None
        )

        self.model.eval()

        self.context_window = getattr(self.model.config, "max_position_embeddings", None)
        if self.context_window is None:
            print("Warning: Model context window not found in config.")

        self.is_instruct = (("-instruct" in checkpoint.lower()) and hasattr(self.tokenizer, "apply_chat_template")
        )

        self.precision = str(next(self.model.parameters()).dtype)

    def get_context_window(self) -> int:
        return self.context_window

    @torch.no_grad()
    def generate_response(self, prompt: str) -> str:
        if self.is_instruct:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        if self.context_window:
            max_length = self.context_window - self.max_new_tokens
            max_length = max_length - (max_length % 8) # for 8bit should be flexible later
        else:
            max_length = None

        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=self.truncate_if_exceeds,
            max_length=max_length,
            padding="longest", 
            pad_to_multiple_of=8 # for 8bit should be flexible later
        ).to(self.device)

        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            repetition_penalty = self.repetition_penalty
            )


        predictions = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return predictions[0]


MODEL_REGISTRY = {
    "distilgpt2": lambda **kwargs: LMWrapper(checkpoint="distilbert/distilgpt2", **kwargs),
    "llama3-8b-instruct": lambda **kwargs: LMWrapper(checkpoint="meta-llama/Meta-Llama-3.1-8B-Instruct", use_bnb_int8=False, **kwargs),
    "llama3-8b": lambda **kwargs: LMWrapper(checkpoint="meta-llama/Meta-Llama-3.1-8B", use_bnb_int8=False, **kwargs),
    "smollm2-1.7b-instruct": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct", **kwargs),
    "smollm2-1.7b": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-1.7B", **kwargs),
    "smollm2-135m-instruct": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-135M-Instruct", **kwargs),
    "smollm2-135m": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-135M", **kwargs),
}
