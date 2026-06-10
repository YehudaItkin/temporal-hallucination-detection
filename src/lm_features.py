"""LM-based features via TinyLlama.

Computes per-subword log-probability, entropy, and rank from a causal LM,
then aggregates them to word-level tokens.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LMFeatureExtractor:
    """Extract token-level LM features (log-prob, entropy, rank)."""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading LM {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(self.device)
        self.model.requires_grad_(False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("LM loaded.")

    @torch.no_grad()
    def extract_token_logprobs(
        self,
        context: str,
        output: str,
        max_ctx_tokens: int = 512,
    ) -> dict:
        """Return per-subword log-probs, entropies, ranks, and offsets."""
        prompt = f"Context: {context[:2000]}\n\nOutput: "
        prompt_char_len = len(prompt)
        full_text = prompt + output
        encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_ctx_tokens + 256,
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        offsets = encoding["offset_mapping"][0].tolist()
        logits = self.model(input_ids).logits[0]

        prompt_token_len = len(offsets)
        for i, (char_start, _char_end) in enumerate(offsets):
            if char_start >= prompt_char_len:
                prompt_token_len = i
                break

        output_logits = logits[prompt_token_len - 1 : -1]
        output_ids = input_ids[0, prompt_token_len:]
        output_offsets = [
            (max(0, s - prompt_char_len), max(0, e - prompt_char_len))
            for s, e in offsets[prompt_token_len:]
        ]

        if len(output_logits) == 0:
            return {
                "logprobs": [],
                "entropies": [],
                "ranks": [],
                "tokens": [],
                "offsets": [],
            }

        log_probs_dist = torch.log_softmax(output_logits.float(), dim=-1)
        probs_dist = torch.softmax(output_logits.float(), dim=-1)
        token_logprobs = (
            log_probs_dist[torch.arange(len(output_ids)), output_ids.cpu()]
            .cpu()
            .numpy()
        )
        entropies = (
            -(probs_dist * log_probs_dist).sum(dim=-1).cpu().numpy()
        )

        sorted_indices = torch.argsort(
            output_logits, dim=-1, descending=True
        )
        ranks: list[int] = []
        for i, tid in enumerate(output_ids.cpu()):
            rank = (sorted_indices[i] == tid).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)

        tokens = [self.tokenizer.decode(tid) for tid in output_ids.cpu()]
        return {
            "logprobs": token_logprobs.tolist(),
            "entropies": entropies.tolist(),
            "ranks": ranks,
            "tokens": tokens,
            "offsets": output_offsets,
        }

    def map_to_words(
        self,
        lm_result: dict,
        word_tokens: list[dict],
    ) -> np.ndarray:
        """Aggregate subword LM features to word-level (T, 4) array."""
        if not lm_result["logprobs"] or not word_tokens:
            return np.zeros((len(word_tokens), 4), dtype=np.float32)
        features = np.zeros((len(word_tokens), 4), dtype=np.float32)
        for w_idx, wtok in enumerate(word_tokens):
            w_start, w_end = wtok["char_start"], wtok["char_end"]
            matched_lp: list[float] = []
            matched_ent: list[float] = []
            matched_rnk: list[int] = []
            for offset, lp, ent, rank in zip(
                lm_result["offsets"],
                lm_result["logprobs"],
                lm_result["entropies"],
                lm_result["ranks"],
            ):
                s_start, s_end = offset
                if s_end <= w_start or s_start >= w_end:
                    continue
                matched_lp.append(lp)
                matched_ent.append(ent)
                matched_rnk.append(rank)
            if matched_lp:
                features[w_idx] = [
                    np.mean(matched_lp),
                    np.mean(matched_ent),
                    np.mean(matched_rnk),
                    np.max(matched_rnk),
                ]
            else:
                features[w_idx] = [-10.0, 5.0, 1000, 1000]
        return features


def extract_and_save(split: str, extractor: LMFeatureExtractor) -> None:
    """Run LM extraction for *split* and persist to disk."""
    data_dir = Path(".") / "prepared_data"
    out_dir = Path(".") / "lm_features"
    out_dir.mkdir(exist_ok=True)
    with open(data_dir / f"{split}.json") as fh:
        examples = json.load(fh)
    all_features: list[list[list[float]]] = []
    for i, ex in enumerate(examples):
        lm_result = extractor.extract_token_logprobs(ex["context"], ex["output"])
        word_feats = extractor.map_to_words(lm_result, ex["token_labels"])
        all_features.append(word_feats.tolist())
        if (i + 1) % 200 == 0:
            print(f"  {split}: {i + 1}/{len(examples)}", flush=True)
    with open(out_dir / f"{split}_lm.json", "w") as fh:
        json.dump(all_features, fh)
    print(
        f"  Saved {len(all_features)} examples to "
        f"{out_dir / f'{split}_lm.json'}"
    )


if __name__ == "__main__":
    extractor = LMFeatureExtractor()
    for split in ("train", "test"):
        print(f"Extracting LM features for {split}...")
        extract_and_save(split, extractor)
