"""Data loading, feature assembly, and batching utilities.

Provides helpers to load pre-extracted features from disk, post-process
NLI and LM features, and create PyTorch ``DataLoader`` instances for
sequence-labeling training.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.features import extract_token_features, extract_token_labels

# Sentinel values used when the LM could not align a token.
LM_FALLBACK = (-10.0, 5.0, 1000.0, 1000.0)


# ------------------------------------------------------------------
# Raw feature loading
# ------------------------------------------------------------------


def load_base_features(
    split: str,
    data_dir: str | Path = "prepared_data",
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """Load examples from disk and compute surface features + labels.

    Returns:
        ``(features_list, labels_list, examples)`` where each element of
        *features_list* has shape ``(T_i, 20)`` and each element of
        *labels_list* has shape ``(T_i,)``.
    """
    data_dir = Path(data_dir)
    with open(data_dir / f"{split}.json") as fh:
        examples = json.load(fh)
    all_feats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for ex in examples:
        all_feats.append(extract_token_features(ex))
        all_labels.append(extract_token_labels(ex))
    return all_feats, all_labels, examples


def load_extra_features(
    split: str,
    kind: str,
    data_dir: str | Path = ".",
) -> list[np.ndarray] | None:
    """Load pre-computed NLI or LM features from a JSON file."""
    path = Path(data_dir) / f"{kind}_features" / f"{split}_{kind}.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return [np.array(x, dtype=np.float32) for x in json.load(fh)]


# ------------------------------------------------------------------
# Feature post-processing
# ------------------------------------------------------------------


def compute_lm_medians(
    lm_features_list: list[np.ndarray],
) -> list[float]:
    """Compute per-column medians over non-fallback LM rows."""
    all_real: list[list[float]] = [[] for _ in range(4)]
    for ex_feats in lm_features_list:
        if len(ex_feats) == 0:
            continue
        for row in ex_feats:
            if not (row[0] == LM_FALLBACK[0] and row[1] == LM_FALLBACK[1]):
                for col in range(4):
                    all_real[col].append(row[col])
    return [float(np.median(vals)) if vals else 0.0 for vals in all_real]


def postprocess_lm(
    lm_features_list: list[np.ndarray],
    medians: list[float] | None = None,
) -> list[np.ndarray]:
    """Post-process raw LM features: impute fallback rows, log-transform ranks.

    Output dimensionality is 6: logprob, entropy, log1p(mean_rank),
    log1p(max_rank), has_alignment, logprob*has_alignment.
    """
    if medians is None:
        medians = compute_lm_medians(lm_features_list)
    result: list[np.ndarray] = []
    for ex_feats in lm_features_list:
        if len(ex_feats) == 0:
            result.append(np.zeros((0, 6), dtype=np.float32))
            continue
        n = len(ex_feats)
        enhanced = np.zeros((n, 6), dtype=np.float32)
        for j in range(n):
            row = ex_feats[j]
            is_fallback = (
                row[0] == LM_FALLBACK[0] and row[1] == LM_FALLBACK[1]
            )
            if is_fallback:
                enhanced[j, 0] = medians[0]
                enhanced[j, 1] = medians[1]
                enhanced[j, 2] = np.log1p(medians[2])
                enhanced[j, 3] = np.log1p(medians[3])
            else:
                enhanced[j, 0] = row[0]
                enhanced[j, 1] = row[1]
                enhanced[j, 2] = np.log1p(row[2])
                enhanced[j, 3] = np.log1p(row[3])
            enhanced[j, 4] = 0.0 if is_fallback else 1.0
            enhanced[j, 5] = enhanced[j, 0] * enhanced[j, 4]
        result.append(enhanced)
    return result


def derive_nli_features(
    nli_features_list: list[np.ndarray],
) -> list[np.ndarray]:
    """Derive enhanced NLI features (7-dim) from raw 3-dim NLI scores."""
    result: list[np.ndarray] = []
    for ex_feats in nli_features_list:
        if len(ex_feats) == 0:
            result.append(np.zeros((0, 7), dtype=np.float32))
            continue
        n = len(ex_feats)
        enhanced = np.zeros((n, 7), dtype=np.float32)
        enhanced[:, :3] = ex_feats[:, :3]
        running_contra = 0.0
        for j in range(n):
            running_contra += ex_feats[j, 0]
            enhanced[j, 3] = running_contra / (j + 1)
            if j > 0:
                enhanced[j, 4] = ex_feats[j, 0] - ex_feats[j - 1, 0]
        contra_col = ex_feats[:, 0]
        for j in range(n):
            start = max(0, j - 9)
            enhanced[j, 5] = contra_col[start : j + 1].max()
        enhanced[:, 6] = 1.0 - ex_feats[:, 1]
        result.append(enhanced)
    return result


def compute_pos_weight(labels_list: list[np.ndarray]) -> float:
    """Compute positive class weight for imbalanced binary labels."""
    all_labs = np.concatenate([lab for lab in labels_list if len(lab) > 0])
    n_neg = (all_labs == 0).sum()
    n_pos = (all_labs == 1).sum()
    return float(n_neg / max(n_pos, 1))


def assemble_features(
    base: list[np.ndarray],
    nli: list[np.ndarray] | None,
    lm: list[np.ndarray] | None,
    use_nli: bool,
    use_lm: bool,
) -> list[np.ndarray]:
    """Horizontally stack feature groups into a single matrix per example."""
    result: list[np.ndarray] = []
    for i in range(len(base)):
        parts = [base[i]]
        if use_nli and nli and i < len(nli) and len(nli[i]) == len(base[i]):
            parts.append(nli[i])
        if use_lm and lm and i < len(lm) and len(lm[i]) == len(base[i]):
            parts.append(lm[i])
        result.append(np.hstack(parts) if len(parts) > 1 else parts[0])
    return result


# ------------------------------------------------------------------
# PyTorch Dataset and collate
# ------------------------------------------------------------------


class SeqDataset(Dataset):
    """Variable-length sequence dataset for token-level labeling."""

    def __init__(
        self,
        features: list[np.ndarray],
        labels: list[np.ndarray],
        max_len: int = 512,
    ) -> None:
        self.data: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        for feat, lab in zip(features, labels):
            if len(feat) == 0:
                continue
            if len(feat) > max_len:
                feat = feat[:max_len]
                lab = lab[:max_len]
            self.data.append(
                (
                    torch.tensor(feat, dtype=torch.float32),
                    torch.tensor(lab, dtype=torch.float32),
                    len(feat),
                )
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.data[idx]


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length sequences."""
    feats, labels, lengths = zip(*batch)
    return (
        pad_sequence(feats, batch_first=True),
        pad_sequence(labels, batch_first=True, padding_value=-1),
        torch.tensor(lengths, dtype=torch.long),
    )
