from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LMWrapper:
    def __init__(self, 
                 checkpoint: str,
                 device: str = None,
                 max_new_tokens: int = 250,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 use_auth_token: bool = False,
                 access_token: str = None,
                 truncate_if_exceeds: bool = True):
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.truncate_if_exceeds = truncate_if_exceeds

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            use_auth_token=access_token if use_auth_token else None
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            use_auth_token=access_token if use_auth_token else None
        ).to(self.device)

        self.context_window = getattr(self.model.config, "max_position_embeddings", None)
        if self.context_window is None:
            print("Warning: Model context window not found in config.")

        self.is_instruct = hasattr(self.tokenizer, "apply_chat_template")

    def get_context_window(self) -> int:
        return self.context_window

    def generate_response(self, prompt: str) -> str:
        if self.is_instruct:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        total_tokens = input_ids.shape[-1] + self.max_new_tokens
        if self.context_window and total_tokens > self.context_window:
            over_by = total_tokens - self.context_window
            if self.truncate_if_exceeds:
                print(f"Warning: Prompt exceeds context window by {over_by} tokens. Truncating input.")
                input_ids = input_ids[:, -(self.context_window - self.max_new_tokens):]
            else:
                return "[ERROR] Input exceeds model's context window."

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)

MODEL_REGISTRY = {
    "distilgpt2": "distilbert/distilgpt2",
    "llama3-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3": "meta-llama/Meta-Llama-3.1-8B",
    "smollm2-1.7b-instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
    "smollm2-135m-instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
}

def get_model(name: str, **kwargs) -> CausalLMWrapper:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return LMWrapper(checkpoint=MODEL_REGISTRY[name], **kwargs)
