from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
                 repetition_penalty: float = 1.0):
        
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
            torch_dtype="auto",
            use_auth_token=access_token if use_auth_token else None
        ).to(self.device)

        self.model.eval()

        self.context_window = getattr(self.model.config, "max_position_embeddings", None)
        if self.context_window is None:
            print("Warning: Model context window not found in config.")

        self.is_instruct = (("-instruct" in checkpoint.lower()) and hasattr(self.tokenizer, "apply_chat_template")
        )

        self.precision = str(next(self.model.parameters()).dtype)

    def get_context_window(self) -> int:
        return self.context_window

    def generate_response(self, prompt: str) -> str:
        if self.is_instruct:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=self.truncate_if_exceeds,
            max_length=self.context_window - self.max_new_tokens if self.context_window else None
        ).input_ids.to(self.device)

        attention_mask = torch.ones_like(input_ids)

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Use no_grad to reduce memory footprint
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                low_memory = True,
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
    "distilgpt2": "distilbert/distilgpt2",
    "llama3-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3": "meta-llama/Meta-Llama-3.1-8B",
    "smollm2-1.7b-instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
    "smollm2-135m-instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
}

def get_model(name: str, **kwargs) -> LMWrapper:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return LMWrapper(checkpoint=MODEL_REGISTRY[name], **kwargs)
