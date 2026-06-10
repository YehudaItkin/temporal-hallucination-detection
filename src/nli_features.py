"""NLI-based features via DeBERTa cross-encoder.

Computes per-sentence entailment / contradiction / neutral probabilities
and maps them back to word-level tokens.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def split_into_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split *text* into ``(sentence, char_start, char_end)`` triples."""
    sentences: list[tuple[str, int, int]] = []
    pattern = re.compile(
        r"(?<!\b[A-Z])(?<!\b\d)(?<!\b[A-Z][a-z])[.!?]\s+"
        r"|(?<!\b[A-Z])(?<!\b\d)[.!?]$|\n+"
    )
    last = 0
    for m in pattern.finditer(text):
        end = m.end()
        chunk = text[last:end].strip()
        if chunk:
            start_actual = text.index(chunk, last)
            sentences.append((chunk, start_actual, start_actual + len(chunk)))
        last = end
    if last < len(text):
        chunk = text[last:].strip()
        if chunk:
            start_actual = text.index(chunk, last)
            sentences.append((chunk, start_actual, start_actual + len(chunk)))
    return sentences


class NLIFeatureExtractor:
    """Extract NLI scores for each token using a DeBERTa cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading NLI model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.requires_grad_(False)
        print("NLI model loaded.")

    def score_sentence(
        self, premise: str, hypothesis: str
    ) -> dict[str, float]:
        """Return contradiction / entailment / neutral probabilities."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        return {
            "contradiction": float(probs[0]),
            "entailment": float(probs[1]),
            "neutral": float(probs[2]),
        }

    def extract_per_token_nli(self, example: dict) -> np.ndarray:
        """Return an ``(T, 3)`` array of NLI features per token."""
        context = example["context"]
        output = example["output"]
        token_labels = example["token_labels"]
        if not token_labels:
            return np.zeros((0, 3), dtype=np.float32)

        ctx_truncated = " ".join(context.split()[:400])
        sentences = split_into_sentences(output)
        if not sentences:
            return np.full((len(token_labels), 3), 1 / 3, dtype=np.float32)

        sentence_scores: list[tuple[int, int, dict[str, float]]] = []
        for sent_text, sent_start, sent_end in sentences:
            if len(sent_text.split()) < 3:
                scores = {
                    "contradiction": 0.0,
                    "neutral": 1.0,
                    "entailment": 0.0,
                }
            else:
                scores = self.score_sentence(ctx_truncated, sent_text)
            sentence_scores.append((sent_start, sent_end, scores))

        nli_features = np.full(
            (len(token_labels), 3), 1 / 3, dtype=np.float32
        )
        for i, tok in enumerate(token_labels):
            tok_mid = (tok["char_start"] + tok["char_end"]) / 2
            for sent_start, sent_end, scores in sentence_scores:
                if sent_start <= tok_mid < sent_end:
                    nli_features[i] = [
                        scores["contradiction"],
                        scores["entailment"],
                        scores["neutral"],
                    ]
                    break
        return nli_features


def extract_and_save(split: str, extractor: NLIFeatureExtractor) -> None:
    """Run NLI extraction for *split* and persist to disk."""
    data_dir = Path(".") / "prepared_data"
    out_dir = Path(".") / "nli_features"
    out_dir.mkdir(exist_ok=True)
    with open(data_dir / f"{split}.json") as fh:
        examples = json.load(fh)
    all_features: list[list[list[float]]] = []
    for i, ex in enumerate(examples):
        feats = extractor.extract_per_token_nli(ex)
        all_features.append(feats.tolist())
        if (i + 1) % 500 == 0:
            print(f"  {split}: {i + 1}/{len(examples)}")
    with open(out_dir / f"{split}_nli.json", "w") as fh:
        json.dump(all_features, fh)
    print(
        f"  Saved {len(all_features)} examples to "
        f"{out_dir / f'{split}_nli.json'}"
    )


if __name__ == "__main__":
    extractor = NLIFeatureExtractor()
    for split in ("train", "test"):
        print(f"Extracting NLI features for {split}...")
        extract_and_save(split, extractor)
