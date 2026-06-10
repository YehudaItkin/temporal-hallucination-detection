#!/usr/bin/env python3
"""Download RAGTruth from HuggingFace and prepare word-level annotations.

Converts span-level hallucination annotations to word-level binary labels
and saves the result as JSON files (one per split).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from datasets import load_dataset


@dataclass
class TokenLabel:
    """Per-word annotation."""

    word: str
    char_start: int
    char_end: int
    is_hallucination: bool
    hallucination_type: str | None = None


@dataclass
class Example:
    """Processed example with word-level labels."""

    id: str
    query: str
    context: str
    output: str
    task_type: str
    model: str
    has_hallucination: bool
    token_labels: list[TokenLabel] = field(default_factory=list)


def parse_hallucination_spans(labels_str: str) -> list[dict]:
    """Parse the JSON string of hallucination span annotations."""
    if not labels_str or labels_str == "[]":
        return []
    try:
        return json.loads(labels_str)
    except json.JSONDecodeError:
        return []


def word_tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Whitespace tokenizer that tracks character offsets."""
    tokens: list[tuple[str, int, int]] = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        tokens.append((text[i:j], i, j))
        i = j
    return tokens


def assign_token_labels(
    output_text: str,
    hallucination_spans: list[dict],
) -> list[TokenLabel]:
    """Map span-level annotations to word-level binary labels."""
    words = word_tokenize_with_offsets(output_text)
    labels: list[TokenLabel] = []
    for word, start, end in words:
        is_hallu = False
        hallu_type = None
        for span in hallucination_spans:
            span_start = span["start"]
            span_end = span["end"]
            overlap = max(0, min(end, span_end) - max(start, span_start))
            word_len = end - start
            if overlap > word_len * 0.5:
                is_hallu = True
                label_type = span.get("label_type", "").lower()
                if "conflict" in label_type:
                    hallu_type = "evident_conflict"
                else:
                    hallu_type = "baseless_info"
                break
        labels.append(TokenLabel(
            word=word,
            char_start=start,
            char_end=end,
            is_hallucination=is_hallu,
            hallucination_type=hallu_type,
        ))
    return labels


def prepare_dataset(split: str = "train") -> list[Example]:
    """Download and process one split of RAGTruth."""
    ds = load_dataset("wandb/RAGTruth-processed", split=split)
    examples: list[Example] = []
    for row in ds:
        spans = parse_hallucination_spans(row["hallucination_labels"])
        proc = row["hallucination_labels_processed"]
        has_hallu = (proc["evident_conflict"] > 0) or (proc["baseless_info"] > 0)
        token_labels = assign_token_labels(row["output"], spans)
        examples.append(Example(
            id=row["id"],
            query=row["query"],
            context=row["context"],
            output=row["output"],
            task_type=row["task_type"],
            model=row["model"],
            has_hallucination=has_hallu,
            token_labels=token_labels,
        ))
    return examples


def save_examples(examples: list[Example], path: Path) -> None:
    """Serialize examples to JSON."""
    data = [asdict(ex) for ex in examples]
    with open(path, "w") as f:
        json.dump(data, f)


def main() -> None:
    out_dir = Path("prepared_data")
    out_dir.mkdir(exist_ok=True)
    for split in ["train", "test"]:
        print(f"Preparing {split}...")
        examples = prepare_dataset(split)
        save_examples(examples, out_dir / f"{split}.json")
        n_hallu = sum(1 for ex in examples if ex.has_hallucination)
        total_tokens = sum(len(ex.token_labels) for ex in examples)
        hallu_tokens = sum(
            sum(1 for t in ex.token_labels if t.is_hallucination)
            for ex in examples
        )
        print(f"  {len(examples)} examples ({n_hallu} with hallucinations)")
        print(
            f"  {total_tokens} total tokens, {hallu_tokens} hallucinated "
            f"({hallu_tokens / total_tokens * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
