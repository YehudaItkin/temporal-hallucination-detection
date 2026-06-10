"""Token-level feature extraction (20 features per token).

Computes surface-level, n-gram overlap, positional, and windowed novelty
features for each word token in the generated output, relative to the
source context.
"""

import re

import numpy as np


def is_in_context(word: str, context_words: set[str]) -> float:
    """Return 1.0 if the word (lowered, stripped) appears in the context set."""
    return 1.0 if word.lower().strip(".,!?;:'\"()") in context_words else 0.0


def is_number(word: str) -> float:
    """Return 1.0 if the word looks like a number / percentage / currency."""
    return 1.0 if re.match(r'^[\d,./%$€£]+$', word) else 0.0


def is_capitalized(word: str) -> float:
    """Return 1.0 if the word starts with an upper-case letter."""
    return 1.0 if word and word[0].isupper() else 0.0


def word_length(word: str) -> int:
    """Return the character length of *word*."""
    return len(word)


def _build_ngram_set(words: list[str], n: int) -> set[tuple[str, ...]]:
    """Build a set of n-gram tuples from a word list."""
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _detect_sentence_boundaries(token_labels: list[dict]) -> list[int]:
    """Return token indices that start a new sentence."""
    boundaries = [0]
    for i, tok in enumerate(token_labels):
        if tok["word"].rstrip() in (".", "!", "?") and i + 1 < len(token_labels):
            boundaries.append(i + 1)
    return boundaries


def extract_token_features(example: dict) -> np.ndarray:
    """Extract a (T, 20) feature matrix for one example.

    Args:
        example: Dictionary with keys ``context`` and ``token_labels``.

    Returns:
        NumPy array of shape ``(T, 20)`` where *T* is the number of tokens.
    """
    ctx = example["context"]
    token_labels = example["token_labels"]
    if not token_labels:
        return np.zeros((0, 20))

    context_words = set(ctx.lower().split())
    context_words_stripped = {w.strip(".,!?;:'\"()") for w in context_words}
    context_numbers = set(re.findall(r"\b\d+\.?\d*\b", ctx))
    output_len = len(token_labels)

    ctx_words_list = [w.strip(".,!?;:'\"()") for w in ctx.lower().split()]
    ctx_bigrams = _build_ngram_set(ctx_words_list, 2)
    ctx_trigrams = _build_ngram_set(ctx_words_list, 3)

    sent_bounds = _detect_sentence_boundaries(token_labels)
    n_sentences = len(sent_bounds)

    features = []
    running_in_ctx = 0.0
    running_novel = 0.0
    consecutive_novel = 0

    for i, tok in enumerate(token_labels):
        word = tok["word"]
        word_lower = word.lower().strip(".,!?;:'\"()")

        in_ctx = 1.0 if word_lower in context_words_stripped else 0.0
        running_in_ctx += in_ctx
        running_novel += 1.0 - in_ctx

        if in_ctx < 0.5:
            consecutive_novel += 1
        else:
            consecutive_novel = 0

        position = i / max(output_len - 1, 1)
        running_ctx_ratio = running_in_ctx / (i + 1)
        running_novel_ratio = running_novel / (i + 1)

        is_num = is_number(word)
        num_in_ctx = 0.0
        if is_num:
            nums = re.findall(r"\d+\.?\d*", word)
            num_in_ctx = 1.0 if any(n in context_numbers for n in nums) else 0.0

        prev_word = (
            token_labels[i - 1]["word"].lower().strip(".,!?;:'\"()") if i > 0 else ""
        )
        bigram_in_ctx = 0.0
        if i > 0 and (prev_word, word_lower) in ctx_bigrams:
            bigram_in_ctx = 1.0
        trigram_in_ctx = 0.0
        if i > 1:
            pprev = token_labels[i - 2]["word"].lower().strip(".,!?;:'\"()")
            if (pprev, prev_word, word_lower) in ctx_trigrams:
                trigram_in_ctx = 1.0

        is_sent_start = 1.0 if i in sent_bounds else 0.0
        is_entity = (
            1.0
            if (is_capitalized(word) > 0.5 and not is_sent_start and len(word) > 1)
            else 0.0
        )

        sent_idx = sum(1 for b in sent_bounds if b <= i) - 1
        sent_position = sent_idx / max(n_sentences - 1, 1)

        feat = [
            in_ctx,
            is_num,
            is_capitalized(word),
            word_length(word) / 20.0,
            position,
            running_ctx_ratio,
            running_novel_ratio,
            1.0 - in_ctx,
            num_in_ctx if is_num else 0.0,
            is_num * (1.0 - num_in_ctx),
            bigram_in_ctx,
            trigram_in_ctx,
            is_entity,
            min(consecutive_novel / 10.0, 1.0),
            sent_position,
        ]
        features.append(feat)

    features = np.array(features, dtype=np.float32)

    # Windowed novelty (5, 10, 20)
    novel_idx = 7
    for window_size in [5, 10, 20]:
        window_feat = np.zeros(len(token_labels), dtype=np.float32)
        novel_col = features[:, novel_idx]
        for i in range(len(token_labels)):
            start = max(0, i - window_size + 1)
            window_feat[i] = novel_col[start : i + 1].mean()
        features = np.column_stack([features, window_feat])

    # Delta and second-delta of running_novel_ratio
    if len(features) > 1:
        novel_ratio = features[:, 6]
        delta = np.zeros_like(novel_ratio)
        delta[1:] = novel_ratio[1:] - novel_ratio[:-1]
        features = np.column_stack([features, delta])
        delta2 = np.zeros_like(delta)
        delta2[2:] = delta[2:] - delta[1:-1]
        features = np.column_stack([features, delta2])
    else:
        features = np.column_stack([features, np.zeros((len(features), 2))])

    return features


FEATURE_NAMES = [
    "in_context",
    "is_number",
    "is_capitalized",
    "word_length",
    "position",
    "running_ctx_ratio",
    "running_novel_ratio",
    "is_novel",
    "num_in_context",
    "novel_number",
    "bigram_in_context",
    "trigram_in_context",
    "is_entity",
    "consecutive_novel",
    "sentence_position",
    "window_5_novelty",
    "window_10_novelty",
    "window_20_novelty",
    "delta_novel_ratio",
    "delta2_novel_ratio",
]


def extract_token_labels(example: dict) -> np.ndarray:
    """Extract binary hallucination labels for each token.

    Returns:
        1-D float32 array of length *T*.
    """
    return np.array(
        [int(t["is_hallucination"]) for t in example["token_labels"]],
        dtype=np.float32,
    )
