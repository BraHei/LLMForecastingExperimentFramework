from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CausalLMWrapper:
    def __init__(self, 
                 checkpoint: str,
                 device: str = None,
                 max_new_tokens: int = 250,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 use_auth_token: bool = False,
                 access_token: str = None,
                 truncate_if_exceeds: bool = True):
        """
        Initialize the CausalLMWrapper.
        """
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
            print("Warning: Model context window (max_position_embeddings) not found in config.")

        self.is_instruct = "-instruct" in checkpoint and hasattr(self.tokenizer, "apply_chat_template")

    def get_context_window(self) -> int:
        """
        Return the model's context window size.
        """
        return self.context_window

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using a chat template if available and applicable.
        """
        if self.is_instruct:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt"
            ).to(self.device)
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        total_tokens = input_ids.shape[-1] + self.max_new_tokens
        if self.context_window and total_tokens > self.context_window:
            over_by = total_tokens - self.context_window
            if self.truncate_if_exceeds:
                print(f"Warning: Prompt and generation exceeds context window by {over_by} tokens. Truncating input.")
                input_ids = input_ids[:, - (self.context_window - self.max_new_tokens):]
            else:
                print(f"Error: Prompt and {self.max_new_tokens} tokens exceeds context window of {self.context_window}.")
                return "[ERROR] Input exceeds model's context window."

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)


def read_multiline_input(prompt: str = "Enter your prompt (end with empty line):") -> str:
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run a causal LM model interactively.")
    parser.add_argument("--checkpoint", type=str, required=True, help="HuggingFace model checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=250, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling threshold")
    parser.add_argument("--access_token", type=str, default=None, help="Hugging Face access token if required")
    parser.add_argument("--no_truncate", action="store_true", help="Disable truncating if prompt exceeds context window")

    args = parser.parse_args()

    model = CausalLMWrapper(
        checkpoint=args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_auth_token=bool(args.access_token),
        access_token=args.access_token,
        truncate_if_exceeds=not args.no_truncate
    )

    print(f"\nModel '{args.checkpoint}' loaded.")
    print(f"Context window: {model.get_context_window()} tokens")
    print("Use ctrl+c to quit.\n")

    while True:
        prompt = read_multiline_input()
        response = model.generate_response(prompt)
        print("\n##################################")
        print("Model Response:")
        print("##################################\n")
        print(response + "\n")


if __name__ == "__main__":
    main()

