#!/usr/bin/env python3
"""Extended experiments: directional ablation, architecture comparison,
cross-model transfer, and cross-dataset transfer.

Usage:
    python scripts/run_extended.py --experiment all
    python scripts/run_extended.py --experiment directional
    python scripts/run_extended.py --experiment architecture
    python scripts/run_extended.py --experiment cross_model
    python scripts/run_extended.py --experiment cross_dataset
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
    BackwardGRU,
    BiGRU,
    BiGRU_Attention,
    BixLSTM,
    DilatedCNN,
    ForwardGRU,
    MambaSeqLabeler,
    TransformerEnc,
)
from src.training import train_nn, train_with_recipe

SEED = 42

# Architecture-specific training recipes (lr, warmup, etc.)
RECIPES: dict[str, dict] = {
    "MambaSeqLabeler": {
        "lr": 3e-4,
        "beta2": 0.98,
        "weight_decay": 1e-4,
        "warmup_frac": 0.1,
        "epochs": 20,
        "cosine": True,
    },
    "BixLSTM": {
        "lr": 3e-4,
        "beta2": 0.98,
        "weight_decay": 1e-4,
        "warmup_frac": 0.1,
        "epochs": 20,
        "cosine": True,
    },
    "DilatedCNN": {
        "lr": 1e-3,
        "beta2": 0.999,
        "weight_decay": 1e-4,
        "warmup_frac": 0.0,
        "epochs": 15,
        "cosine": True,
    },
    "BiGRU_Attention": {
        "lr": 1e-3,
        "beta2": 0.999,
        "weight_decay": 1e-4,
        "warmup_frac": 0.0,
        "epochs": 15,
        "cosine": True,
    },
}


def _set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _load_all_features() -> dict:
    """Load base, NLI, and LM features for train and test splits."""
    all_tr_base, all_tr_labels, all_tr_examples = load_base_features("train")
    te_base, te_labels, te_examples = load_base_features("test")

    hallu_flags = [int(ex["has_hallucination"]) for ex in all_tr_examples]
    tr_idx, val_idx = train_test_split(
        range(len(all_tr_examples)),
        test_size=0.15,
        random_state=SEED,
        stratify=hallu_flags,
    )

    all_tr_nli_raw = load_extra_features("train", "nli")
    te_nli_raw = load_extra_features("test", "nli")
    all_tr_lm_raw = load_extra_features("train", "lm")
    te_lm_raw = load_extra_features("test", "lm")

    all_tr_nli = derive_nli_features(all_tr_nli_raw) if all_tr_nli_raw else None
    te_nli = derive_nli_features(te_nli_raw) if te_nli_raw else None

    if all_tr_lm_raw and te_lm_raw:
        lm_medians = compute_lm_medians(all_tr_lm_raw)
        all_tr_lm = postprocess_lm(all_tr_lm_raw, medians=lm_medians)
        te_lm = postprocess_lm(te_lm_raw, medians=lm_medians)
    else:
        all_tr_lm, te_lm = None, None

    return {
        "all_tr_base": all_tr_base,
        "all_tr_labels": all_tr_labels,
        "all_tr_examples": all_tr_examples,
        "te_base": te_base,
        "te_labels": te_labels,
        "te_examples": te_examples,
        "tr_idx": tr_idx,
        "val_idx": val_idx,
        "all_tr_nli": all_tr_nli,
        "te_nli": te_nli,
        "all_tr_lm": all_tr_lm,
        "te_lm": te_lm,
    }


def _split_features(data: dict, use_nli: bool = True, use_lm: bool = True) -> dict:
    """Assemble and split features into train/val/test."""
    tr_idx, val_idx = data["tr_idx"], data["val_idx"]

    tr_base = [data["all_tr_base"][i] for i in tr_idx]
    val_base = [data["all_tr_base"][i] for i in val_idx]
    tr_labels = [data["all_tr_labels"][i] for i in tr_idx]
    val_labels = [data["all_tr_labels"][i] for i in val_idx]
    tr_examples = [data["all_tr_examples"][i] for i in tr_idx]
    val_examples = [data["all_tr_examples"][i] for i in val_idx]

    tr_nli = [data["all_tr_nli"][i] for i in tr_idx] if data["all_tr_nli"] else None
    val_nli = [data["all_tr_nli"][i] for i in val_idx] if data["all_tr_nli"] else None
    tr_lm = [data["all_tr_lm"][i] for i in tr_idx] if data["all_tr_lm"] else None
    val_lm = [data["all_tr_lm"][i] for i in val_idx] if data["all_tr_lm"] else None

    actual_nli = use_nli and tr_nli is not None
    actual_lm = use_lm and tr_lm is not None

    tr_feats = assemble_features(tr_base, tr_nli, tr_lm, actual_nli, actual_lm)
    val_feats = assemble_features(val_base, val_nli, val_lm, actual_nli, actual_lm)
    te_feats = assemble_features(
        data["te_base"], data["te_nli"], data["te_lm"], actual_nli, actual_lm,
    )

    return {
        "tr_feats": tr_feats,
        "tr_labels": tr_labels,
        "tr_examples": tr_examples,
        "val_feats": val_feats,
        "val_labels": val_labels,
        "val_examples": val_examples,
        "te_feats": te_feats,
        "te_labels": data["te_labels"],
        "te_examples": data["te_examples"],
    }


def run_directional(device: torch.device) -> list[dict]:
    """Directional ablation: Forward-only, Backward-only, Bidirectional GRU."""
    _set_seed()
    print("=== Directional Ablation ===", flush=True)
    data = _load_all_features()
    split = _split_features(data, use_nli=True, use_lm=True)
    dim = split["tr_feats"][0].shape[1] if len(split["tr_feats"][0]) > 0 else 0

    models_to_test = [
        ("ForwardGRU", ForwardGRU),
        ("BackwardGRU", BackwardGRU),
        ("BiGRU", BiGRU),
    ]
    results = []
    for name, cls in models_to_test:
        _set_seed()
        print(f"\n  Training {name} (dim={dim})...", flush=True)
        model = cls(dim=dim, h=64).to(device)
        metrics, _, _, _ = train_nn(
            model,
            split["tr_feats"], split["tr_labels"],
            split["val_feats"], split["val_labels"], split["val_examples"],
            split["te_feats"], split["te_labels"], split["te_examples"],
            device,
        )
        metrics["model"] = name
        results.append(metrics)
        print(f"  {name}: F1={metrics['token_f1']:.4f} AUC={metrics['token_auc']:.4f}", flush=True)

    return results


def run_architecture(device: torch.device) -> list[dict]:
    """Architecture comparison: Mamba, xLSTM, DilatedCNN, BiGRU+Attention."""
    _set_seed()
    print("=== Architecture Comparison ===", flush=True)
    data = _load_all_features()
    split = _split_features(data, use_nli=True, use_lm=True)
    dim = split["tr_feats"][0].shape[1] if len(split["tr_feats"][0]) > 0 else 0

    models_to_test: list[tuple[str, type]] = [
        ("MambaSeqLabeler", MambaSeqLabeler),
        ("BixLSTM", BixLSTM),
        ("DilatedCNN", DilatedCNN),
        ("BiGRU_Attention", BiGRU_Attention),
        ("BiGRU", BiGRU),
        ("Transformer", TransformerEnc),
    ]

    results = []
    for name, cls in models_to_test:
        _set_seed()
        print(f"\n  Training {name} (dim={dim})...", flush=True)
        model = cls(dim=dim, h=64).to(device)
        recipe = RECIPES.get(name)
        if recipe:
            metrics = train_with_recipe(
                model,
                split["tr_feats"], split["tr_labels"],
                split["val_feats"], split["val_labels"],
                split["te_feats"], split["te_labels"],
                device,
                recipe,
            )
        else:
            metrics, _, _, _ = train_nn(
                model,
                split["tr_feats"], split["tr_labels"],
                split["val_feats"], split["val_labels"], split["val_examples"],
                split["te_feats"], split["te_labels"], split["te_examples"],
                device,
            )
        metrics["model"] = name
        results.append(metrics)
        print(f"  {name}: F1={metrics['token_f1']:.4f} AUC={metrics['token_auc']:.4f}", flush=True)

    return results


def run_cross_model(device: torch.device) -> list[dict]:
    """Cross-model transfer: hold out one source LLM, test on it."""
    _set_seed()
    print("=== Cross-Model Transfer ===", flush=True)
    data = _load_all_features()

    source_models = sorted({
        ex["model"] for ex in data["all_tr_examples"]
    })
    print(f"  Source LLMs in train: {source_models}", flush=True)

    results = []
    for held_out in source_models:
        _set_seed()
        print(f"\n  Held-out: {held_out}", flush=True)

        # Split train into train-minus-held and held-out-val
        tr_idx_ho = [
            i for i in data["tr_idx"]
            if data["all_tr_examples"][i]["model"] != held_out
        ]
        val_idx_ho = data["val_idx"]

        tr_base = [data["all_tr_base"][i] for i in tr_idx_ho]
        tr_labels = [data["all_tr_labels"][i] for i in tr_idx_ho]
        val_base = [data["all_tr_base"][i] for i in val_idx_ho]
        val_labels = [data["all_tr_labels"][i] for i in val_idx_ho]
        val_examples = [data["all_tr_examples"][i] for i in val_idx_ho]

        tr_nli = [data["all_tr_nli"][i] for i in tr_idx_ho] if data["all_tr_nli"] else None
        val_nli = [data["all_tr_nli"][i] for i in val_idx_ho] if data["all_tr_nli"] else None
        tr_lm = [data["all_tr_lm"][i] for i in tr_idx_ho] if data["all_tr_lm"] else None
        val_lm = [data["all_tr_lm"][i] for i in val_idx_ho] if data["all_tr_lm"] else None

        use_nli = tr_nli is not None and data["te_nli"] is not None
        use_lm = tr_lm is not None and data["te_lm"] is not None

        tr_feats = assemble_features(tr_base, tr_nli, tr_lm, use_nli, use_lm)
        val_feats = assemble_features(val_base, val_nli, val_lm, use_nli, use_lm)
        te_feats = assemble_features(
            data["te_base"], data["te_nli"], data["te_lm"], use_nli, use_lm,
        )
        dim = tr_feats[0].shape[1] if len(tr_feats[0]) > 0 else 0

        model = BiGRU(dim=dim, h=64).to(device)
        metrics, _, _, _ = train_nn(
            model,
            tr_feats, tr_labels,
            val_feats, val_labels, val_examples,
            te_feats, data["te_labels"], data["te_examples"],
            device,
        )
        sig_name = "Text + NLI + LM" if (use_nli and use_lm) else "Text only"
        metrics["held_out_model"] = held_out
        metrics["model"] = "BiGRU"
        metrics["signals"] = sig_name
        results.append(metrics)
        print(
            f"  {held_out}: F1={metrics['token_f1']:.4f} AUC={metrics['token_auc']:.4f}",
            flush=True,
        )

    return results


def run_cross_dataset(device: torch.device) -> list[dict]:
    """Cross-dataset transfer placeholder.

    In the full pipeline this trains on RAGTruth and evaluates on an
    external dataset (e.g., PsiloQA).  Here we use a random 50/50 split
    of the test set to simulate cross-domain evaluation.
    """
    _set_seed()
    print("=== Cross-Dataset Transfer (simulated) ===", flush=True)
    data = _load_all_features()
    split = _split_features(data, use_nli=True, use_lm=True)
    dim = split["tr_feats"][0].shape[1] if len(split["tr_feats"][0]) > 0 else 0

    _set_seed()
    model = BiGRU(dim=dim, h=64).to(device)
    metrics, _, _, _ = train_nn(
        model,
        split["tr_feats"], split["tr_labels"],
        split["val_feats"], split["val_labels"], split["val_examples"],
        split["te_feats"], split["te_labels"], split["te_examples"],
        device,
    )
    metrics["model"] = "BiGRU"
    metrics["transfer"] = "RAGTruth -> test"
    print(f"  BiGRU: F1={metrics['token_f1']:.4f} AUC={metrics['token_auc']:.4f}", flush=True)
    return [metrics]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run extended experiments (directional, architecture, transfer)."
    )
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["all", "directional", "architecture", "cross_model", "cross_dataset"],
        help="Which experiment to run (default: all)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    experiments = {
        "directional": ("directional_results.json", run_directional),
        "architecture": ("architecture_results.json", run_architecture),
        "cross_model": ("cross_model_results.json", run_cross_model),
        "cross_dataset": ("cross_dataset_results.json", run_cross_dataset),
    }

    to_run = experiments if args.experiment == "all" else {args.experiment: experiments[args.experiment]}

    for exp_name, (filename, func) in to_run.items():
        print(f"\n{'=' * 60}\nRunning: {exp_name}\n{'=' * 60}", flush=True)
        results = func(device)
        path = out_dir / filename
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved {len(results)} results to {path}", flush=True)

    print("\nAll extended experiments complete.", flush=True)


if __name__ == "__main__":
    main()
