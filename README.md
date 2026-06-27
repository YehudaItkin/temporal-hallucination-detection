# Temporal Hallucination Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20977858.svg)](https://doi.org/10.5281/zenodo.20977858)

Token-level hallucination detection using temporal signal modeling.
This repository contains the code and scripts to reproduce all experiments
from the paper.

## Environment Setup

### Option A: conda (recommended)

```bash
conda create -n hallu_detect python=3.11 -y
conda activate hallu_detect
pip install -r requirements.txt
```

### Option B: pip + venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Install the package

```bash
pip install -e .
```

## Reproduction Steps

All commands should be run from the repository root directory.

### Step 1: Prepare Data

Download RAGTruth from HuggingFace and convert span annotations to
word-level binary labels:

```bash
python scripts/prepare_data.py
```

This creates `prepared_data/train.json` and `prepared_data/test.json`.

**Expected runtime:** 5-10 minutes (downloads ~200 MB from HuggingFace).

### Step 2: Extract NLI Features

Compute per-sentence entailment/contradiction/neutral probabilities using
DeBERTa and map them to word-level tokens:

```bash
python -m src.nli_features
```

This creates `nli_features/train_nli.json` and `nli_features/test_nli.json`.

**Expected runtime:** 1-2 hours on GPU, 6-8 hours on CPU.

### Step 3: Extract LM Features

Compute per-token log-probability, entropy, and rank from TinyLlama:

```bash
python -m src.lm_features
```

This creates `lm_features/train_lm.json` and `lm_features/test_lm.json`.

**Expected runtime:** 2-4 hours on GPU, 12-18 hours on CPU.

### Step 4: Run Ablation Study

Run the main signal x architecture ablation (all 4 signal configurations
and 8 model architectures):

```bash
python scripts/run_ablation.py
```

Results are saved to `results/ablation_results.json` and
`results/per_task_results.json`.

**Expected runtime:** 30-60 minutes on GPU.

### Step 4b: Run Extended Experiments (optional)

Run directional ablation, architecture comparison (Mamba, xLSTM,
DilatedCNN, BiGRU+Attention), cross-model transfer, and cross-dataset
transfer:

```bash
python scripts/run_extended.py --experiment all
```

Or run individual experiments:

```bash
python scripts/run_extended.py --experiment directional
python scripts/run_extended.py --experiment architecture
python scripts/run_extended.py --experiment cross_model
python scripts/run_extended.py --experiment cross_dataset
```

**Expected runtime:** 1-2 hours on GPU for all experiments.

### Step 5: Generate Figures

Generate all publication-quality PDF figures:

```bash
python scripts/generate_figures.py --data-dir results --output-dir figures
```

Figures are saved as PDF files in `figures/`.

## Hardware Requirements

- **Minimum:** 16 GB RAM, any modern CPU (steps 1, 4, 5)
- **Recommended:** NVIDIA GPU with at least 8 GB VRAM (steps 2, 3, 4b)
- **Storage:** ~2 GB for data and extracted features

Steps 2 and 3 (feature extraction) are the most resource-intensive and
benefit significantly from GPU acceleration.  Steps 4 and 4b train
lightweight models (< 1M parameters) and run in under an hour on GPU.

## Project Structure

```
src/
    __init__.py          # Package init
    features.py          # Surface-level token features (20 dims)
    nli_features.py      # NLI feature extraction via DeBERTa
    lm_features.py       # LM feature extraction via TinyLlama
    data.py              # Data loading, feature assembly, batching
    models.py            # Neural architectures (CNN, GRU, LSTM, Transformer, Mamba, xLSTM, CRF)
    training.py          # Training loops and evaluation
    onset_metrics.py     # Hallucination onset detection metrics
scripts/
    prepare_data.py      # Step 1: download and prepare RAGTruth
    run_ablation.py      # Step 4: main ablation study
    run_extended.py      # Step 4b: extended experiments
    generate_figures.py  # Step 5: generate paper figures
```
