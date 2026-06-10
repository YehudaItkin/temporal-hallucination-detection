"""Hallucination onset detection metrics.

Evaluates how accurately a model detects the *beginning* of hallucination
spans rather than just individual token labels.
"""

import numpy as np


def find_span_onsets(token_labels: list[dict]) -> list[int]:
    """Return indices where a new hallucination span starts."""
    onsets: list[int] = []
    in_span = False
    for i, tok in enumerate(token_labels):
        if tok["is_hallucination"] and not in_span:
            onsets.append(i)
            in_span = True
        elif not tok["is_hallucination"]:
            in_span = False
    return onsets


def find_predicted_onsets(
    probs: np.ndarray, threshold: float = 0.5
) -> list[int]:
    """Return indices where a new predicted span starts."""
    binary = (probs >= threshold).astype(int)
    onsets: list[int] = []
    for i in range(len(binary)):
        if binary[i] == 1 and (i == 0 or binary[i - 1] == 0):
            onsets.append(i)
    return onsets


def onset_precision_recall_f1(
    true_onsets: list[int],
    pred_onsets: list[int],
    tolerance_k: int = 3,
) -> dict[str, float]:
    """Compute onset-level precision, recall, and F1 with tolerance *k*."""
    if not pred_onsets and not true_onsets:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_onsets:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not true_onsets:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp_precision = sum(
        1
        for p in pred_onsets
        if any(abs(p - t) <= tolerance_k for t in true_onsets)
    )
    precision = tp_precision / len(pred_onsets)

    tp_recall = sum(
        1
        for t in true_onsets
        if any(abs(p - t) <= tolerance_k for p in pred_onsets)
    )
    recall = tp_recall / len(true_onsets)

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def early_detection_rate(
    true_onsets: list[int],
    pred_onsets: list[int],
    tolerance_k: int = 3,
) -> float:
    """Fraction of detected onsets caught at or before the true position."""
    if not true_onsets or not pred_onsets:
        return 0.0
    early_count = 0
    detected_count = 0
    for t in true_onsets:
        matching_preds = [
            p for p in pred_onsets if abs(p - t) <= tolerance_k
        ]
        if matching_preds:
            detected_count += 1
            if min(matching_preds) <= t:
                early_count += 1
    return early_count / detected_count if detected_count > 0 else 0.0


def compute_onset_metrics(
    examples: list[dict],
    all_probs: list[np.ndarray],
    threshold: float = 0.5,
    tolerance_values: list[int] | None = None,
) -> dict[str, float]:
    """Aggregate onset metrics across examples for multiple tolerance values."""
    if tolerance_values is None:
        tolerance_values = [1, 3, 5]
    results: dict[str, float] = {}
    for k in tolerance_values:
        all_prec: list[float] = []
        all_rec: list[float] = []
        all_f1: list[float] = []
        all_early: list[float] = []
        for ex, probs in zip(examples, all_probs):
            true_onsets = find_span_onsets(ex["token_labels"])
            if not true_onsets:
                continue
            pred_onsets = find_predicted_onsets(probs, threshold)
            metrics = onset_precision_recall_f1(true_onsets, pred_onsets, k)
            all_prec.append(metrics["precision"])
            all_rec.append(metrics["recall"])
            all_f1.append(metrics["f1"])
            all_early.append(
                early_detection_rate(true_onsets, pred_onsets, k)
            )
        results[f"onset_precision@{k}"] = (
            float(np.mean(all_prec)) if all_prec else 0.0
        )
        results[f"onset_recall@{k}"] = (
            float(np.mean(all_rec)) if all_rec else 0.0
        )
        results[f"onset_f1@{k}"] = (
            float(np.mean(all_f1)) if all_f1 else 0.0
        )
        results[f"early_detection@{k}"] = (
            float(np.mean(all_early)) if all_early else 0.0
        )
    return results
