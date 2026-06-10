#!/usr/bin/env python3
"""Generate publication-quality figures for temporal hallucination detection paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MODEL_ORDER = [
    "LogReg", "1D-CNN", "BiLSTM", "BiGRU",
    "Transformer", "BiGRU-CRF", "BiLSTM-CRF", "Trans-CRF",
]
SIGNAL_ORDER = ["Text only", "Text + NLI", "Text + LM", "Text + NLI + LM"]
HELD_OUT_ORDER = [
    "gpt-4-0613", "gpt-3.5-turbo-0613",
    "llama-2-70b-chat", "llama-2-13b-chat", "llama-2-7b-chat",
    "mistral-7B-instruct",
]
HELD_OUT_LABELS = {
    "gpt-4-0613": "GPT-4",
    "gpt-3.5-turbo-0613": "GPT-3.5",
    "llama-2-70b-chat": "Llama-2-70B",
    "llama-2-13b-chat": "Llama-2-13B",
    "llama-2-7b-chat": "Llama-2-7B",
    "mistral-7B-instruct": "Mistral-7B",
}

CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]


def _setup_style() -> None:
    sns.set_style("whitegrid")
    mpl.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
    })


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, format="pdf", dpi=300)
    plt.close(fig)
    print(f"  Saved {path.name}")


def figure_ablation_heatmap(data_dir: Path, output_dir: Path) -> None:
    records = _load_json(data_dir / "ablation_results_v5.json")
    df = pd.DataFrame(records)

    pivot = df.pivot_table(index="model", columns="signals", values="token_auc")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=SIGNAL_ORDER)

    best_val = pivot.max().max()
    best_loc = np.argwhere(pivot.values == best_val)[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        vmin=0.55,
        vmax=0.90,
        cbar_kws={"label": "Token AUC", "shrink": 0.8},
    )

    ax.add_patch(mpl.patches.Rectangle(
        (best_loc[1], best_loc[0]), 1, 1,
        fill=False, edgecolor="black", linewidth=2.5,
    ))

    ax.set_xlabel("Signal Configuration")
    ax.set_ylabel("Model Architecture")
    ax.set_title("Token-Level AUC by Model and Signal Configuration")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    _save_fig(fig, output_dir / "ablation_heatmap.pdf")


def figure_per_task_comparison(data_dir: Path, output_dir: Path) -> None:
    records = _load_json(data_dir / "per_task_results.json")
    df = pd.DataFrame(records)

    target_models = ["LogReg", "BiGRU", "BiLSTM", "Transformer"]
    mask = (df["signals"] == "Text + NLI + LM") & (df["model"].isin(target_models))
    subset = df[mask].copy()

    task_order = ["Summary", "QA", "Data2txt"]
    subset["task_type"] = pd.Categorical(subset["task_type"], categories=task_order, ordered=True)
    subset["model"] = pd.Categorical(subset["model"], categories=target_models, ordered=True)
    subset = subset.sort_values(["task_type", "model"])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = [CB_PALETTE[i] for i in range(len(target_models))]

    x = np.arange(len(task_order))
    width = 0.18
    offsets = np.arange(len(target_models)) - (len(target_models) - 1) / 2

    for i, model in enumerate(target_models):
        vals = subset[subset["model"] == model].set_index("task_type").reindex(task_order)["token_auc"]
        bars = ax.bar(x + offsets[i] * width, vals, width * 0.9, label=model, color=colors[i])
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=5.5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(task_order)
    ax.set_ylabel("Token AUC")
    ax.set_title("Performance by Task Type (All Signals)")
    ax.set_ylim(0.55, 0.95)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.grid(axis="x", visible=False)

    _save_fig(fig, output_dir / "per_task_comparison.pdf")


def figure_cross_model_transfer(data_dir: Path, output_dir: Path) -> None:
    records = _load_json(data_dir / "cross_model_results.json")
    df = pd.DataFrame(records)

    target_archs = ["LogReg", "BiGRU", "Transformer"]
    mask = df["signals"] == "Text + NLI + LM"
    subset = df[mask].copy()

    pivot = subset.pivot_table(index="held_out_model", columns="model", values="token_auc")
    pivot = pivot.reindex(index=HELD_OUT_ORDER, columns=target_archs)

    avg_row = pivot.mean(axis=0)
    avg_df = pd.DataFrame(avg_row).T
    avg_df.index = ["Average"]
    pivot = pd.concat([pivot, avg_df])

    display_labels = [HELD_OUT_LABELS.get(m, m) for m in HELD_OUT_ORDER] + ["Average"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        vmin=0.60,
        vmax=0.90,
        cbar_kws={"label": "Token AUC", "shrink": 0.8},
    )

    ax.set_yticklabels(display_labels, rotation=0)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Held-Out Source LLM")
    ax.set_title("Cross-Model Generalization (All Signals)")

    ax.axhline(y=len(HELD_OUT_ORDER), color="black", linewidth=1.5)

    _save_fig(fig, output_dir / "cross_model_transfer.pdf")


def figure_transfer_comparison(data_dir: Path, output_dir: Path) -> None:
    fwd = pd.DataFrame(_load_json(data_dir / "transfer_results.json"))
    rev = pd.DataFrame(_load_json(data_dir / "reverse_transfer_results.json"))

    fwd_bigru = fwd[fwd["model"] == "BiGRU"].set_index("signals").reindex(SIGNAL_ORDER)
    rev_bigru = rev[rev["model"] == "BiGRU"].set_index("signals").reindex(SIGNAL_ORDER)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(SIGNAL_ORDER))
    width = 0.32

    bars_fwd = ax.bar(
        x - width / 2, fwd_bigru["token_auc"], width,
        label="RAGTruth -> PsiloQA", color=CB_PALETTE[0],
    )
    bars_rev = ax.bar(
        x + width / 2, rev_bigru["token_auc"], width,
        label="PsiloQA -> RAGTruth", color=CB_PALETTE[1],
    )

    for bar in bars_fwd:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6,
        )
    for bar in bars_rev:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SIGNAL_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Token AUC")
    ax.set_title("Dataset Transfer: BiGRU")
    ax.set_ylim(0.55, 0.80)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)

    _save_fig(fig, output_dir / "transfer_comparison.pdf")


def figure_length_analysis(data_dir: Path, output_dir: Path) -> None:
    records = _load_json(data_dir / "length_analysis.json")
    df = pd.DataFrame(records)

    logreg = df[df["model"] == "LogReg"].sort_values("mean_length")
    bigru = df[df["model"] == "BiGRU"].sort_values("mean_length")

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(
        logreg["mean_length"], logreg["token_auc"],
        "o-", color=CB_PALETTE[0], label="LogReg", linewidth=1.5, markersize=5,
    )
    ax.plot(
        bigru["mean_length"], bigru["token_auc"],
        "s-", color=CB_PALETTE[1], label="BiGRU", linewidth=1.5, markersize=5,
    )

    ax.fill_between(
        logreg["mean_length"].values,
        logreg["token_auc"].values,
        bigru["token_auc"].values,
        alpha=0.15, color=CB_PALETTE[1], label="Temporal advantage",
    )

    quartile_labels = ["Q1 (short)", "Q2", "Q3", "Q4 (long)"]
    for _i, (ml, auc_lr, auc_bg, _ql) in enumerate(zip(
        logreg["mean_length"], logreg["token_auc"], bigru["token_auc"], quartile_labels
    )):
        delta = auc_bg - auc_lr
        mid_y = (auc_lr + auc_bg) / 2
        ax.annotate(
            f"d={delta:.3f}",
            xy=(ml, mid_y),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=6,
            color=CB_PALETTE[1],
            va="center",
        )

    ax.set_xlabel("Mean Token Length")
    ax.set_ylabel("Token AUC")
    ax.set_title("Effect of Sequence Length on Performance")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(logreg["mean_length"].values)
    ax2.set_xticklabels(quartile_labels, fontsize=6)
    ax2.tick_params(length=0)

    _save_fig(fig, output_dir / "length_analysis.pdf")


def figure_crf_calibration(data_dir: Path, output_dir: Path) -> None:
    records = _load_json(data_dir / "crf_calibration.json")
    df = pd.DataFrame(records)

    model_order = ["BiGRU-CRF", "BiLSTM-CRF", "Trans-CRF"]
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values("model")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(model_order))
    width = 0.32

    bars_sm = ax.bar(
        x - width / 2, df["softmax_auc"], width,
        label="Softmax", color=CB_PALETTE[0],
    )
    bars_fb = ax.bar(
        x + width / 2, df["fb_auc"], width,
        label="Forward-Backward", color=CB_PALETTE[2],
    )

    for sm_bar, fb_bar, delta in zip(bars_sm, bars_fb, df["auc_delta"]):
        top = max(sm_bar.get_height(), fb_bar.get_height())
        mid_x = (sm_bar.get_x() + fb_bar.get_x() + fb_bar.get_width()) / 2
        ax.annotate(
            f"d={delta:+.3f}",
            xy=(mid_x, top + 0.005),
            ha="center", va="bottom", fontsize=6.5, fontweight="bold",
            color="#333333",
        )

    for bar in bars_sm:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.02,
            f"{bar.get_height():.3f}", ha="center", va="top", fontsize=6, color="white",
        )
    for bar in bars_fb:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.02,
            f"{bar.get_height():.3f}", ha="center", va="top", fontsize=6, color="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_order)
    ax.set_ylabel("Token AUC")
    ax.set_title("CRF Decoding: Softmax vs Forward-Backward")
    ax.set_ylim(0.55, 0.92)
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)

    _save_fig(fig, output_dir / "crf_calibration.pdf")


def figure_signal_contribution(data_dir: Path, output_dir: Path) -> None:
    records = _load_json(data_dir / "ablation_results_v5.json")
    df = pd.DataFrame(records)
    bigru = df[df["model"] == "BiGRU"].set_index("signals").reindex(SIGNAL_ORDER)

    auc_text = bigru.loc["Text only", "token_auc"]
    auc_nli = bigru.loc["Text + NLI", "token_auc"]
    auc_lm = bigru.loc["Text + LM", "token_auc"]
    auc_all = bigru.loc["Text + NLI + LM", "token_auc"]

    base = auc_text
    gain_nli = auc_nli - auc_text
    gain_lm = auc_lm - auc_text
    gain_combined = auc_all - auc_text
    gain_interaction = gain_combined - gain_nli - gain_lm

    labels = ["Text\n(base)", "+NLI", "+LM", "Interaction", "Total"]
    running = [base, base + gain_nli, base + gain_nli + gain_lm,
               base + gain_nli + gain_lm + gain_interaction]
    gains = [gain_nli, gain_lm, gain_interaction]

    colors_bars = ["#4C72B0", CB_PALETTE[2], CB_PALETTE[1], CB_PALETTE[3], "#E24A33"]

    y_floor = 0.82
    fig, ax = plt.subplots(figsize=(5, 3))

    ax.bar(labels[0], base - y_floor, bottom=y_floor, color=colors_bars[0],
           edgecolor="white", linewidth=0.8, width=0.6)
    ax.text(0, y_floor + (base - y_floor) / 2, f"{base:.3f}",
            ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    bottoms_inc = [base, base + gain_nli, base + gain_nli + gain_lm]
    gain_labels_text = [f"+{gain_nli:.4f}", f"+{gain_lm:.4f}", f"+{gain_interaction:.4f}"]
    for i, (g, b, gl) in enumerate(zip(gains, bottoms_inc, gain_labels_text)):
        ax.bar(labels[i + 1], g, bottom=b, color=colors_bars[i + 1],
               edgecolor="white", linewidth=0.8, width=0.6)
        ax.text(i + 1, b + g + 0.0005, gl,
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=colors_bars[i + 1])

    ax.bar(labels[4], auc_all - y_floor, bottom=y_floor, color=colors_bars[4],
           edgecolor="white", linewidth=0.8, width=0.6)
    ax.text(4, y_floor + (auc_all - y_floor) / 2, f"{auc_all:.3f}",
            ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    for i in range(len(gains)):
        ax.plot([i + 0.3, i + 0.7], [running[i], running[i]],
                color="gray", linewidth=0.7, linestyle="--")

    ax.set_ylabel("Token AUC")
    ax.set_title("Signal Contribution Waterfall (BiGRU)")
    ax.set_ylim(y_floor, auc_all + 0.006)
    ax.grid(axis="x", visible=False)

    ax.spines["bottom"].set_visible(False)
    diag_len = 0.01
    ax.plot((-diag_len, diag_len), (-diag_len, diag_len),
            transform=ax.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax.plot((1 - diag_len, 1 + diag_len), (-diag_len, diag_len),
            transform=ax.transAxes, color="k", clip_on=False, linewidth=0.8)

    _save_fig(fig, output_dir / "signal_contribution.pdf")


def figure_pr_curves(data_dir: Path, output_dir: Path) -> None:
    pr_path = data_dir / "pr_curves.json"
    if not pr_path.exists():
        print("  Skipped (pr_curves.json not found)")
        return

    pr_data = _load_json(pr_path)

    target_models = ["LogReg", "MLP", "1D-CNN", "BiLSTM", "BiGRU", "Transformer"]
    colors = {m: CB_PALETTE[i] for i, m in enumerate(target_models)}
    markers = {"LogReg": "x", "MLP": "v", "1D-CNN": "^", "BiLSTM": "D", "BiGRU": "o", "Transformer": "s"}

    fig, ax = plt.subplots(figsize=(5, 4))

    for model in target_models:
        if model not in pr_data:
            continue
        points = pr_data[model]
        recalls = [p["recall"] for p in points]
        precisions = [p["precision"] for p in points]
        ax.plot(
            recalls, precisions,
            marker=markers.get(model, "o"), markersize=4,
            color=colors.get(model, "gray"), label=model, linewidth=1.5,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (All Signals, seed=42)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper right")

    _save_fig(fig, output_dir / "pr_curves.pdf")


def figure_robustness_seeds(data_dir: Path, output_dir: Path) -> None:
    summary_path = data_dir / "robustness_summary.json"
    if not summary_path.exists():
        print("  Skipped (robustness_summary.json not found)")
        return

    summary = _load_json(summary_path)

    target_models = ["LogReg", "MLP", "1D-CNN", "BiLSTM", "BiGRU", "Transformer"]
    model_data = {s["model"]: s for s in summary if s["model"] in target_models}

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for ax, metric, label in [
        (axes[0], "token_auc", "Token AUC"),
        (axes[1], "example_auc", "Example-Level AUC"),
    ]:
        means = [model_data[m][f"{metric}_mean"] for m in target_models if m in model_data]
        stds = [model_data[m][f"{metric}_std"] for m in target_models if m in model_data]
        names = [m for m in target_models if m in model_data]
        colors = [CB_PALETTE[i] for i in range(len(names))]

        bars = ax.bar(names, means, yerr=stds, capsize=4, color=colors,
                      edgecolor="white", linewidth=0.8)
        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=6,
            )

        ax.set_ylabel(label)
        ax.set_ylim(0.55, 0.88)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.grid(axis="x", visible=False)

    axes[0].set_title("Token-Level AUC (5 seeds)")
    axes[1].set_title("Example-Level AUC (5 seeds)")

    _save_fig(fig, output_dir / "robustness_seeds.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for temporal hallucination paper."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "figures",
        help="Directory for output PDF figures",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()

    generators = [
        ("Figure 1: Ablation Heatmap", figure_ablation_heatmap),
        ("Figure 2: Per-Task Comparison", figure_per_task_comparison),
        ("Figure 3: Cross-Model Transfer", figure_cross_model_transfer),
        ("Figure 4: Transfer Comparison", figure_transfer_comparison),
        ("Figure 5: Length Analysis", figure_length_analysis),
        ("Figure 6: CRF Calibration", figure_crf_calibration),
        ("Figure 7: Signal Contribution", figure_signal_contribution),
        ("Figure 8: PR Curves", figure_pr_curves),
        ("Figure 9: Robustness Seeds", figure_robustness_seeds),
    ]

    for name, func in generators:
        print(f"Generating {name}...")
        func(args.data_dir, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
