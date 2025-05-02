from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = [
    "LMWrapper",
    "MODEL_REGISTRY",
    "get_model",
]

def _softmax(logits: torch.Tensor) -> torch.Tensor:
    """ Numerically-stable softmax """
    return torch.log_softmax(logits, dim=-1).exp()


def _single_token_ids(tokenizer, chars: str) -> Dict[str, int]:
    """Return mapping for *single* character tokens (digits, dot, comma).

    We assume a **character-level tokenizer** (as used in the paper). If the tokenizer
    merges digits (e.g. "23" –> a single token) the discrete-to-continuous method doesn’t
    apply directly and the caller should fall back to sampling-based uncertainty. An
    exception is raised so the experiment pipeline can catch & handle it.
    """
    mapping = {}
    for c in chars:
        token_ids = tokenizer.encode(c, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Tokenizer does not treat '{c}' as a single token. "
                "Use a character-level tokenizer or adapt mapping logic."
            )
        mapping[c] = token_ids[0]
    return mapping


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

    @torch.no_grad()
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
    

    # @torch.no_grad()
    # def next_token_distribution(
    #     self, prompt: str, restrict_to_chars: str | None = None
    # ) -> Dict[str, float]:
    #     """Return probability distribution over next **single** token.

    #     If *restrict_to_chars* is provided we only keep probabilities for that
    #     subset (e.g., digits, dot, comma) and renormalise.
    #     """
    #     if self.is_instruct:
    #         messages = [{"role": "user", "content": prompt}]
    #         prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

    #     input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
    #     logits = self.model(input_ids).logits[0, -1]  # (vocab,)
    #     probs = _softmax(logits).cpu()

    #     if restrict_to_chars is not None:
    #         keep_ids = [_single_token_ids(self.tokenizer, restrict_to_chars)[c] for c in restrict_to_chars]
    #         probs_restricted = probs[keep_ids]
    #         probs_restricted /= probs_restricted.sum()
    #         tokens = restrict_to_chars
    #         return {t: p.item() for t, p in zip(tokens, probs_restricted)}
    #     else:
    #         topk_probs, topk_idx = torch.topk(probs, k=100)
    #         return {
    #             self.tokenizer.decode([idx]): p.item() for idx, p in zip(topk_idx, topk_probs)
    #         }

    # def numeric_density(
    #     self,
    #     prompt: str,
    #     precision: int = 1,
    #     delimiter: str = ",",
    #     ) -> List[Tuple[Tuple[float, float], float]]:
    #     """Compute piece-wise uniform density over R for a **single-step forecast**.

    #     *precision* is number of decimal places – following the paper we assume
    #     fixed-width bins of 10^{-precision}.

    #     Returns a list of tuples: [ ((low, high), probability), ... ]
    #     """
    #     # Sanity: ensure delimiter & digits are single-token
    #     _single_token_ids(self.tokenizer, delimiter)
    #     digits = "0123456789"
    #     digit_ids = _single_token_ids(self.tokenizer, digits)
    #     dot_id = _single_token_ids(self.tokenizer, ".")["."]
    #     delim_id = _single_token_ids(self.tokenizer, delimiter)[delimiter]

    #     # encode prompt once
    #     if self.is_instruct:
    #         messages = [{"role": "user", "content": prompt}]
    #         prompt_enc = self.tokenizer.apply_chat_template(messages, tokenize=False)
    #     else:
    #         prompt_enc = prompt
    #     base_input = self.tokenizer(prompt_enc, return_tensors="pt").input_ids.to(self.device)

    #     # BFS enumeration limited to 1 integer + dot + precision digits (e.g., "23.4")
    #     # ------------------------------------------------------------------
    #     sequences: List[Tuple[List[int], str, float]] = [([], "", 1.0)]  # (token_ids, str, prob)
    #     final: List[Tuple[str, float]] = []

    #     for step in range(1 + 1 + precision):  # integer, dot, decimals
    #         new_sequences = []
    #         for token_prefix, str_prefix, prob_prefix in sequences:
    #             # context ids = base_input + token_prefix
    #             inp = torch.cat([base_input, torch.tensor([token_prefix], device=self.device, dtype=torch.long)], dim=-1) if token_prefix else base_input
    #             logits = self.model(inp).logits[0, -1]
    #             probs = _softmax(logits)

    #             # branch out valid next tokens
    #             candidate_ids = list(digit_ids.values())
    #             if step == 1:  # after integer place we allow dot
    #                 candidate_ids.append(dot_id)
    #             for cid in candidate_ids:
    #                 p = probs[cid].item()
    #                 if p == 0:
    #                     continue
    #                 char = self.tokenizer.decode([cid])
    #                 new_sequences.append((token_prefix + [cid], str_prefix + char, prob_prefix * p))
    #         sequences = new_sequences

    #     # append delimiter token to terminate and collect probability mass
    #     for token_prefix, num_str, prob in sequences:
    #         inp = torch.cat([base_input, torch.tensor([token_prefix], device=self.device, dtype=torch.long)], dim=-1)
    #         logits = self.model(inp).logits[0, -1]
    #         p_end = _softmax(logits)[delim_id].item()
    #         if p_end > 0:
    #             final.append((num_str, prob * p_end))

    #     # normalise (safety)
    #     total_prob = sum(p for _, p in final)
    #     if total_prob == 0:
    #         raise RuntimeError("No probability mass captured – check tokenizer / prompt")
    #     final = [(s, p / total_prob) for s, p in final]

    #     # map to bins
    #     bin_width = 10 ** (-precision)
    #     density: List[Tuple[Tuple[float, float], float]] = []
    #     for s, p in final:
    #         value = float(s)
    #         low = value - 0.5 * bin_width
    #         high = value + 0.5 * bin_width
    #         density.append(((low, high), p / bin_width))  # uniform density in bin
    #     return density
    
    # def symbol_density(
    #     self,
    #     prompt: str,
    #     centres: dict[str, float],
    #     widths: dict[str, float] | None = None,
    #     ):
    #     """
    #     Continuous density for an ABBA symbol forecast.
    #     centres : symbol -> centre value
    #     widths  : symbol -> half-width (optional).  If None, uses half the
    #               nearest-cluster distance.
    #     Returns: [ ((low, high), density) , … ]
    #     """
    #     if widths is None:
    #         # build default widths = ½ min distance to any other centre
    #         widths = {}
    #         for s, c in centres.items():
    #             d = min(abs(c - c2) for t, c2 in centres.items() if t != s)
    #             widths[s] = 0.5 * d

    #     # get P(symbol | prompt)
    #     probs = self.next_token_distribution(prompt, restrict_to_chars="".join(centres))
    #     density = []
    #     for s, p_s in probs.items():
    #         w = widths[s]
    #         low, high = centres[s] - w, centres[s] + w
    #         density.append(((low, high), p_s / (2 * w)))  # uniform in bin
    #     return density

MODEL_REGISTRY = {
    "distilgpt2": lambda **kwargs: LMWrapper(checkpoint="distilbert/distilgpt2", **kwargs),
    "llama3-instruct": lambda **kwargs: LMWrapper(checkpoint="meta-llama/Meta-Llama-3.1-8B-Instruct", **kwargs),
    "llama3": lambda **kwargs: LMWrapper(checkpoint="meta-llama/Meta-Llama-3.1-8B", **kwargs),
    "smollm2-1.7b-instruct": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct", **kwargs),
    "smollm2-1.7b": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-1.7B", **kwargs),
    "smollm2-135m-instruct": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-135M-Instruct", **kwargs),
    "smollm2-135m": lambda **kwargs: LMWrapper(checkpoint="HuggingFaceTB/SmolLM2-135M", **kwargs),
}
