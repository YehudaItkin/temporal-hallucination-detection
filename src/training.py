"""Training loops and evaluation for token-level hallucination detection.

Provides training routines for neural sequence labelers (with and without
CRF heads), logistic regression baselines, and per-example scoring utilities.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data import SeqDataset, collate, compute_pos_weight
from src.onset_metrics import compute_onset_metrics


# ------------------------------------------------------------------
# Scoring helpers
# ------------------------------------------------------------------


def score_model(
    model: nn.Module,
    test_loader: DataLoader,
    test_examples: list[dict],
    device: torch.device,
) -> tuple[dict, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Evaluate a model on a test loader and return aggregate metrics.

    Returns:
        Tuple of ``(metrics_dict, per_example_probs, per_example_preds,
        per_example_labels)``.
    """
    model.eval()
    per_example_probs: list[np.ndarray] = []
    per_example_preds: list[np.ndarray] = []
    per_example_labels: list[np.ndarray] = []
    is_crf = getattr(model, "has_crf", False)

    with torch.no_grad():
        for feats, labels, lengths in test_loader:
            feats, labels = feats.to(device), labels.to(device)
            mask = labels >= 0
            if is_crf:
                emissions = model(feats)
                decoded = model.crf.decode(emissions, mask)
                p = torch.softmax(emissions, dim=-1)[:, :, 1]
            else:
                p = torch.sigmoid(model(feats))
                decoded = None
            for i in range(len(lengths)):
                m = mask[i]
                ex_probs = p[i][m].cpu().numpy()
                ex_labs = labels[i][m].cpu().numpy()
                per_example_probs.append(ex_probs)
                per_example_labels.append(ex_labs)
                if is_crf:
                    per_example_preds.append(decoded[i][m].cpu().numpy())
                else:
                    per_example_preds.append((ex_probs > 0.5).astype(int))

    all_labs = np.concatenate(per_example_labels)
    all_probs = np.concatenate(per_example_probs)
    all_preds = np.concatenate(per_example_preds)
    token_f1 = f1_score(all_labs, all_preds, zero_division=0)
    token_auc = (
        roc_auc_score(all_labs, all_probs)
        if len(np.unique(all_labs)) > 1
        else 0.0
    )

    hallu_idx = [
        i for i, ex in enumerate(test_examples) if ex["has_hallucination"]
    ]
    hallu_examples = [test_examples[i] for i in hallu_idx]
    hallu_probs = [
        per_example_probs[i] for i in hallu_idx if i < len(per_example_probs)
    ]
    onset = compute_onset_metrics(hallu_examples, hallu_probs)

    metrics = {"token_f1": token_f1, "token_auc": token_auc, **onset}
    return metrics, per_example_probs, per_example_preds, per_example_labels


def score_model_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Score a model without onset metrics (used during validation).

    Returns:
        ``(per_example_probs, per_example_labels)``
    """
    model.eval()
    per_example_probs: list[np.ndarray] = []
    per_example_labels: list[np.ndarray] = []
    with torch.no_grad():
        for feats, labels, lengths in loader:
            feats, labels = feats.to(device), labels.to(device)
            mask = labels >= 0
            p = torch.sigmoid(model(feats))
            for i in range(len(lengths)):
                m = mask[i]
                per_example_probs.append(p[i][m].cpu().numpy())
                per_example_labels.append(labels[i][m].cpu().numpy())
    return per_example_probs, per_example_labels


def score_per_task(
    per_example_probs: list[np.ndarray],
    per_example_preds: list[np.ndarray],
    per_example_labels: list[np.ndarray],
    test_examples: list[dict],
) -> dict[str, dict]:
    """Break down metrics by ``task_type`` field in examples."""
    groups: dict[str, list[int]] = {}
    for i, ex in enumerate(test_examples):
        tt = ex.get("task_type", "unknown")
        groups.setdefault(tt, []).append(i)
    task_results: dict[str, dict] = {}
    for tt, indices in sorted(groups.items()):
        labs = np.concatenate([per_example_labels[i] for i in indices])
        probs = np.concatenate([per_example_probs[i] for i in indices])
        preds = np.concatenate([per_example_preds[i] for i in indices])
        f1 = f1_score(labs, preds, zero_division=0)
        auc = (
            roc_auc_score(labs, probs)
            if len(np.unique(labs)) > 1
            else 0.0
        )
        hallu_idx = [
            i for i in indices if test_examples[i]["has_hallucination"]
        ]
        hallu_ex = [test_examples[i] for i in hallu_idx]
        hallu_pr = [per_example_probs[i] for i in hallu_idx]
        onset = compute_onset_metrics(hallu_ex, hallu_pr) if hallu_ex else {}
        task_results[tt] = {
            "token_f1": f1,
            "token_auc": auc,
            "n_examples": len(indices),
            "n_hallu": len(hallu_idx),
            **onset,
        }
    return task_results


# ------------------------------------------------------------------
# Neural network training
# ------------------------------------------------------------------


def train_nn(
    model: nn.Module,
    train_feats: list[np.ndarray],
    train_labels: list[np.ndarray],
    val_feats: list[np.ndarray],
    val_labels: list[np.ndarray],
    val_examples: list[dict],
    test_feats: list[np.ndarray],
    test_labels: list[np.ndarray],
    test_examples: list[dict],
    device: torch.device,
    epochs: int = 15,
) -> tuple[dict, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Train a neural model with early stopping on validation F1."""
    train_ds = SeqDataset(train_feats, train_labels)
    val_ds = SeqDataset(val_feats, val_labels)
    test_ds = SeqDataset(test_feats, test_labels)
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, collate_fn=collate
    )

    is_crf = getattr(model, "has_crf", False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    if not is_crf:
        pw = compute_pos_weight(train_labels)
        pos_weight = torch.tensor([pw], device=device)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction="none"
        )
    best_f1, best_state = 0.0, None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for feats, labels, _lengths in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            mask = labels >= 0
            if is_crf:
                loss = model.crf(
                    model(feats), labels.long().clamp(min=0), mask
                )
            else:
                logits = model(feats)
                loss = (criterion(logits, labels) * mask).sum() / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_metrics, _, _, _ = score_model(
            model, val_loader, val_examples, device
        )
        avg_loss = epoch_loss / max(n_batches, 1)
        print(
            f"  Epoch {epoch + 1}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"val_f1={val_metrics['token_f1']:.4f}  "
            f"val_auc={val_metrics['token_auc']:.4f}",
            flush=True,
        )
        if val_metrics["token_f1"] > best_f1:
            best_f1 = val_metrics["token_f1"]
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state:
        model.load_state_dict(best_state)
    return score_model(model, test_loader, test_examples, device)


def train_with_recipe(
    model: nn.Module,
    tr_feats: list[np.ndarray],
    tr_labs: list[np.ndarray],
    val_feats: list[np.ndarray],
    val_labs: list[np.ndarray],
    te_feats: list[np.ndarray],
    te_labs: list[np.ndarray],
    device: torch.device,
    recipe: dict,
) -> dict:
    """Train with an architecture-specific recipe (lr, beta2, warmup, etc.).

    Returns:
        Dictionary of test metrics including training history.
    """
    lr = recipe["lr"]
    betas = (0.9, recipe.get("beta2", 0.999))
    wd = recipe.get("weight_decay", 1e-4)
    epochs = recipe.get("epochs", 15)
    warmup_frac = recipe.get("warmup_frac", 0.0)

    tr_ds = SeqDataset(tr_feats, tr_labs)
    val_ds = SeqDataset(val_feats, val_labs)
    te_ds = SeqDataset(te_feats, te_labs)
    tr_loader = DataLoader(
        tr_ds, batch_size=32, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, collate_fn=collate
    )
    te_loader = DataLoader(
        te_ds, batch_size=64, shuffle=False, collate_fn=collate
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=betas, weight_decay=wd
    )
    total_steps = epochs * len(tr_loader)
    warmup_steps = int(total_steps * warmup_frac)
    use_cosine = recipe.get("cosine", True)

    if use_cosine:
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(
                total_steps - warmup_steps, 1
            )
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    pw = compute_pos_weight(tr_labs)
    pos_weight = torch.tensor([pw], device=device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight, reduction="none"
    )

    best_f1, best_state = 0.0, None
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for feats, labels, _lengths in tr_loader:
            feats, labels = feats.to(device), labels.to(device)
            mask = labels >= 0
            logits = model(feats)
            loss = (
                criterion(logits, labels.clamp(min=0)) * mask
            ).sum() / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_probs, val_labs_out = score_model_full(model, val_loader, device)
        val_all = np.concatenate(val_labs_out)
        val_p = np.concatenate(val_probs)
        val_f1 = f1_score(val_all, (val_p > 0.5).astype(int), zero_division=0)
        val_auc = (
            roc_auc_score(val_all, val_p)
            if len(np.unique(val_all)) > 1
            else 0.0
        )
        current_lr = optimizer.param_groups[0]["lr"]
        avg_loss = epoch_loss / n_batches
        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_f1": val_f1,
            "val_auc": val_auc,
            "lr": current_lr,
        })
        print(
            f"  Epoch {epoch + 1}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"val_f1={val_f1:.4f}  "
            f"val_auc={val_auc:.4f}  "
            f"lr={current_lr:.6f}",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state:
        model.load_state_dict(best_state)

    te_probs, te_labs_out = score_model_full(model, te_loader, device)
    all_probs = np.concatenate(te_probs)
    all_labs = np.concatenate(te_labs_out)
    preds = (all_probs > 0.5).astype(int)
    return {
        "token_auc": (
            roc_auc_score(all_labs, all_probs)
            if len(np.unique(all_labs)) > 1
            else 0.0
        ),
        "token_f1": f1_score(all_labs, preds, zero_division=0),
        "precision": precision_score(all_labs, preds, zero_division=0),
        "recall": recall_score(all_labs, preds, zero_division=0),
        "avg_precision": (
            average_precision_score(all_labs, all_probs)
            if len(np.unique(all_labs)) > 1
            else 0.0
        ),
        "history": history,
    }


# ------------------------------------------------------------------
# Logistic regression baseline
# ------------------------------------------------------------------


def run_logreg(
    train_feats: list[np.ndarray],
    train_labels: list[np.ndarray],
    test_feats: list[np.ndarray],
    test_labels: list[np.ndarray],
    test_examples: list[dict],
) -> tuple[dict, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Train and evaluate a logistic regression baseline."""
    x_tr = np.concatenate([f for f in train_feats if len(f) > 0])
    y_tr = np.concatenate([lab for lab in train_labels if len(lab) > 0])
    x_te = np.concatenate([f for f in test_feats if len(f) > 0])
    y_te = np.concatenate([lab for lab in test_labels if len(lab) > 0])

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_tr, y_tr)
    y_prob = clf.predict_proba(x_te)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    token_f1 = f1_score(y_te, y_pred, zero_division=0)
    token_auc = roc_auc_score(y_te, y_prob)

    per_example_probs: list[np.ndarray] = []
    per_example_preds: list[np.ndarray] = []
    per_example_labels: list[np.ndarray] = []
    idx = 0
    for feat, lab in zip(test_feats, test_labels):
        n = len(feat)
        per_example_probs.append(
            y_prob[idx : idx + n] if n > 0 else np.array([])
        )
        per_example_preds.append(
            y_pred[idx : idx + n] if n > 0 else np.array([])
        )
        per_example_labels.append(lab if n > 0 else np.array([]))
        idx += n

    hallu_idx = [
        i for i, ex in enumerate(test_examples) if ex["has_hallucination"]
    ]
    hallu_examples = [test_examples[i] for i in hallu_idx]
    hallu_probs = [per_example_probs[i] for i in hallu_idx]
    onset = compute_onset_metrics(hallu_examples, hallu_probs)

    metrics = {"token_f1": token_f1, "token_auc": token_auc, **onset}
    return metrics, per_example_probs, per_example_preds, per_example_labels
