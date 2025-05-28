from __future__ import annotations

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


__all__ = [
    "LMWrapper",
    "MODEL_REGISTRY",
]

class LMWrapper:
    def __init__(self, 
                 checkpoint: str,
                 device: str = None,
                 max_new_tokens: int = 250,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 use_auth_token: bool = False,
                 access_token: str = None,
                 truncate_if_exceeds: bool = True,
                 do_sample: bool = False,
                 repetition_penalty: float = 1.0):
        
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

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, 
            torch_dtype=torch.bfloat16,
            use_auth_token=access_token if use_auth_token else None,
            device_map="auto",
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
    def generate_response(self, prompt: str, separator: str) -> str:
        try:
            if self.is_instruct:
                messages = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=self.truncate_if_exceeds,
                max_length=self.context_window - self.max_new_tokens if self.context_window else None,
            ).to(self.model.device)

            # Define allowed tokens: digits, separator, space, dash, and any known numeric tokens up to 1000
            allowed_tokens = set("0123456789" + separator + " -" + prompt)

            # Add number strings that are single tokens (like "259", "1000") — only if tokenizer treats them as one token
            for i in range(10000):
                s = str(i)
                token_ids = self.tokenizer(s, add_special_tokens=False)["input_ids"]
                if len(token_ids) == 1:
                    allowed_tokens.add(s)

            # Convert allowed tokens into token IDs
            allowed_token_ids = set()
            for token in allowed_tokens:
                token_ids = self.tokenizer(token, add_special_tokens=False)["input_ids"]
                # Skip unknown tokens
                if self.tokenizer.unk_token_id in token_ids and len(token_ids) == 1:
                    continue
                allowed_token_ids.update(token_ids)

            # Build bad token list
            vocab_size = len(self.tokenizer)
            bad_token_ids = [[i] for i in range(vocab_size) if i not in allowed_token_ids]

            torch.cuda.empty_cache()
            with torch.inference_mode():
                outputs = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    repetition_penalty=self.repetition_penalty,
                    bad_words_ids=bad_token_ids,
                    renormalize_logits=True
                )

            predictions = self.tokenizer.batch_decode(
                outputs[:, enc["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            return predictions[0]

        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if "out of memory" not in str(err).lower():
                raise
            print(f"WARNING: error while generating response: {err}")
            return ""

MODEL_REGISTRY = {
    "distilgpt2-88m": lambda **kwargs: LMWrapper(checkpoint="distilbert/distilgpt2", **kwargs),
    "llama3.1-8b-instruct": lambda **kwargs: LMWrapper(checkpoint="meta-llama/Meta-Llama-3.1-8B-Instruct", **kwargs),
    "llama3.1-8b": lambda **kwargs: LMWrapper(checkpoint="meta-llama/Meta-Llama-3.1-8B", **kwargs),
    "smollm2-1.7b-instruct": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct", **kwargs),
    "smollm2-1.7b": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-1.7B", **kwargs),
    "smollm2-360m-instruct": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct", **kwargs),
    "smollm2-360m": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-360M", **kwargs),
    "smollm2-135m-instruct": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-135M-Instruct", **kwargs),
    "smollm2-135m": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-135M", **kwargs),

    "smollm-1.7b-instruct": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM-1.7B-Instruct", **kwargs),
    "smollm-1.7b": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM-1.7B", **kwargs),
    "smollm-360m-instruct": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM-360M-Instruct", **kwargs),
    "smollm-360m": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM-360M", **kwargs),
    "smollm-135m-instruct": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM-135M-Instruct", **kwargs),
    "smollm-135m": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM-135M", **kwargs),
}
