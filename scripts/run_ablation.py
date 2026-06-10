#!/usr/bin/env python3
"""Main ablation study script.

Runs all combinations of signal sets (Text, +NLI, +LM, +NLI+LM) and
model architectures (LogReg, CNN, BiLSTM, BiGRU, Transformer, +CRF),
reporting token-level F1/AUC and onset-detection metrics.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data import (
    assemble_features,
    compute_lm_medians,
    derive_nli_features,
    load_base_features,
    load_extra_features,
    postprocess_lm,
)
from src.models import (
    BiGRU,
    BiGRU_CRF,
    BiLSTM,
    BiLSTM_CRF,
    CNN1D,
    TransCRF,
    TransformerEnc,
)
from src.training import run_logreg, score_per_task, train_nn

SEED = 42


def run_ablation(device: torch.device) -> None:
    """Execute the full signal x architecture ablation."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print(f"Device: {device}", flush=True)
    print("Loading features...", flush=True)

    all_tr_base, all_tr_labels, all_tr_examples = load_base_features("train")
    te_base, te_labels, te_examples = load_base_features("test")

    hallu_flags = [int(ex["has_hallucination"]) for ex in all_tr_examples]
    tr_idx, val_idx = train_test_split(
        range(len(all_tr_examples)),
        test_size=0.15,
        random_state=SEED,
        stratify=hallu_flags,
    )
    tr_base = [all_tr_base[i] for i in tr_idx]
    tr_labels = [all_tr_labels[i] for i in tr_idx]
    tr_examples = [all_tr_examples[i] for i in tr_idx]
    val_base = [all_tr_base[i] for i in val_idx]
    val_labels = [all_tr_labels[i] for i in val_idx]
    val_examples = [all_tr_examples[i] for i in val_idx]
    print(
        f"Train: {len(tr_examples)}, Val: {len(val_examples)}, "
        f"Test: {len(te_examples)}",
        flush=True,
    )

    # Load optional NLI / LM features
    all_tr_nli_raw = load_extra_features("train", "nli")
    te_nli_raw = load_extra_features("test", "nli")
    all_tr_lm_raw = load_extra_features("train", "lm")
    te_lm_raw = load_extra_features("test", "lm")

    if all_tr_nli_raw and te_nli_raw:
        print("Deriving enhanced NLI features (7 dims)...", flush=True)
        all_tr_nli = derive_nli_features(all_tr_nli_raw)
        te_nli = derive_nli_features(te_nli_raw)
    else:
        all_tr_nli, te_nli = None, None

    if all_tr_lm_raw and te_lm_raw:
        print("Postprocessing LM features (6 dims)...", flush=True)
        lm_medians = compute_lm_medians(all_tr_lm_raw)
        all_tr_lm = postprocess_lm(all_tr_lm_raw, medians=lm_medians)
        te_lm = postprocess_lm(te_lm_raw, medians=lm_medians)
    else:
        all_tr_lm, te_lm = None, None

    tr_nli = [all_tr_nli[i] for i in tr_idx] if all_tr_nli else None
    val_nli = [all_tr_nli[i] for i in val_idx] if all_tr_nli else None
    tr_lm = [all_tr_lm[i] for i in tr_idx] if all_tr_lm else None
    val_lm = [all_tr_lm[i] for i in val_idx] if all_tr_lm else None

    # Signal configurations -- always define all 4; skip at runtime if missing
    configs: list[tuple[str, bool, bool]] = [
        ("Text only", False, False),
        ("Text + NLI", True, False),
        ("Text + LM", False, True),
        ("Text + NLI + LM", True, True),
    ]

    # Model architectures
    models: list[tuple[str, type | None]] = [
        ("LogReg", None),
        ("1D-CNN", CNN1D),
        ("BiLSTM", BiLSTM),
        ("BiGRU", BiGRU),
        ("Transformer", TransformerEnc),
        ("BiGRU-CRF", BiGRU_CRF),
        ("BiLSTM-CRF", BiLSTM_CRF),
        ("Trans-CRF", TransCRF),
    ]
    all_results: list[dict] = []
    task_results: list[dict] = []

    for sig_name, use_nli, use_lm in configs:
        if use_nli and (tr_nli is None or te_nli is None):
            print(f"\nSkipping '{sig_name}': NLI features not found.", flush=True)
            continue
        if use_lm and (tr_lm is None or te_lm is None):
            print(f"\nSkipping '{sig_name}': LM features not found.", flush=True)
            continue
        tr_feats = assemble_features(tr_base, tr_nli, tr_lm, use_nli, use_lm)
        v_feats = assemble_features(val_base, val_nli, val_lm, use_nli, use_lm)
        te_feats = assemble_features(te_base, te_nli, te_lm, use_nli, use_lm)
        dim = tr_feats[0].shape[1] if len(tr_feats[0]) > 0 else 0

        for mod_name, cls in models:
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            random.seed(SEED)
            print(
                f"\n{'=' * 50}\n  {sig_name} x {mod_name} (dim={dim})"
                f"\n{'=' * 50}",
                flush=True,
            )
            if cls is None:
                r, ex_probs, ex_preds, ex_labs = run_logreg(
                    tr_feats, tr_labels, te_feats, te_labels, te_examples
                )
            else:
                model = cls(dim=dim, h=64).to(device)
                r, ex_probs, ex_preds, ex_labs = train_nn(
                    model,
                    tr_feats,
                    tr_labels,
                    v_feats,
                    val_labels,
                    val_examples,
                    te_feats,
                    te_labels,
                    te_examples,
                    device,
                )

            r["signals"] = sig_name
            r["model"] = mod_name
            all_results.append(r)
            print(
                f"  Token F1={r['token_f1']:.4f}  AUC={r['token_auc']:.4f}",
                flush=True,
            )

            ptr = score_per_task(ex_probs, ex_preds, ex_labs, te_examples)
            for tt, metrics in ptr.items():
                task_results.append({
                    "signals": sig_name,
                    "model": mod_name,
                    "task_type": tt,
                    **metrics,
                })

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(out_dir / "per_task_results.json", "w") as f:
        json.dump(task_results, f, indent=2, default=str)
    print("\nSaved to results/ablation_results.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the signal x architecture ablation study."
    )
    parser.add_argument(
        "--mode",
        default="ablation",
        choices=["ablation"],
        help="Experiment mode (default: ablation)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "ablation":
        run_ablation(device)


if __name__ == "__main__":
    main()
