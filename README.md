# Demographic Bias in Face Recognition under Image Perturbations

**Authors:** Meng Wei, Nico Zeitz — TU Darmstadt

---

## Overview

This project studies how common image perturbations affect face verification performance across four demographic groups (African, Asian, Caucasian, Indian) using the [RFW dataset](http://www.whdeng.cn/RFW/index.html) and an IResNet-34 face recognition model.

For each perturbation type and intensity level, the pipeline:

1. Runs the face recognition model over all image pairs and saves cosine similarity scores.
2. Computes biometric verification metrics (EER, FNMR at various FMR operating points) per group.
3. Computes fairness metrics (FDR, DoB) that quantify performance disparity across groups.
4. Generates plots for all metrics and saves a structured CSV report.

Results can be explored interactively in `src/notebook.ipynb`.

---

## Setup

Requires Python ≥ 3.13. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### Required files (not in repo)

| Path | Description |
|------|-------------|
| `data/RFW_test_pairs/` | RFW test pairs — one subdirectory per race (`African_test/`, `Asian_test/`, `Caucasian_test/`, `Indian_test/`), each containing numbered pair folders (`pair0_True/`, `pair1_False/`, …) |
| `data/195520backbone_optimized.pt` | TorchScript checkpoint of the IResNet-34 face recognition model |

---

## Scripts

### 1. `src/extract.py` — Feature extraction

Runs the model over all races and perturbation conditions and saves cosine similarity scores to disk. Already-computed conditions are skipped by default.

```bash
uv run python src/extract.py
```

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint PATH` | `data/195520backbone_optimized.pt` | Model checkpoint |
| `--dataset PATH` | `data/RFW_test_pairs` | RFW test pairs root |
| `--output PATH` | `output/` | Output directory |
| `--races RACE …` | all four | Demographic groups to process |
| `--perturbations-config PATH` | `config/perturbations.json` | Perturbations JSON file or inline JSON string |
| `--no-cache` | — | Recompute even if results already exist |
| `--batch-size INT` | `64` | DataLoader batch size |
| `--num-workers INT` | `4` | DataLoader worker threads |
| `--seed INT` | `42` | Random seed |

### 2. `src/analysis.py` — Analysis and plotting

Loads saved features, computes all metrics, writes `output/verification_report.csv` and `output/fairness_report.csv`, and generates all plots under `output/plots/`.

```bash
uv run python src/analysis.py
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output PATH` | `output/` | Output directory |
| `--races RACE …` | all four | Demographic groups to include |
| `--perturbations-config PATH` | `config/perturbations.json` | Perturbations JSON file or inline JSON string |
| `--force` | — | Regenerate plots even if they already exist |
| `--no-overall` | — | Exclude the pooled overall curve from ROC/DET plots |
| `--metrics OP …` | all four | Operating points to include in summarization plots: `EER`, `ZeroFMR`, `FMR100`, `FMR1000` |

### 3. `src/perturbation_examples.py` — Perturbation visualisation

Generates a grid of example images for every perturbation type and intensity level and saves them to `output/perturbation_examples/`.

```bash
uv run python src/perturbation_examples.py
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output PATH` | `output/` | Output directory |
| `--dataset PATH` | `data/RFW_test_pairs` | RFW test pairs root (used to pick a sample image) |
| `--cols INT` | `6` | Number of columns in the image grid |

---

## Notebook

`src/notebook.ipynb` provides interactive analysis on top of the saved reports. It includes:

- Score distribution plots per group and perturbation level
- ROC and DET curves with and without perturbation
- Summary plots of EER, FNMR, FDR, and DoB vs. perturbation strength
- Animated sweep across perturbation intensities per noise type
- Tables highlighting the worst-performing perturbation level per type

Open with:

```bash
uv run jupyter lab src/notebook.ipynb
```

---

## Perturbations

Defined in `config/perturbations.json`. Each entry specifies a perturbation name and its parameter value. The file is read at runtime — add, remove, or adjust entries without any code changes.

| Type | Parameter | Levels |
|------|-----------|--------|
| **Gaussian blur** | σ (kernel std dev) | 0.2, 0.4, …, 4.0 (20 levels in steps of 0.2) |
| **Brightness** | β (multiplier applied to pixel values) | 0.05–3.0 (21 levels, non-uniform spacing to sample near β=1 more densely) |
| **Gaussian noise** | σ (std dev on 0–255 scale) | 2, 4, 6, 8, 10, 20, 30, 40 |
| **Salt & pepper** | p (probability of corrupted pixel) | 0.03, 0.06, …, 0.30 (10 levels) |
| **Contrast** | α (scaling factor around mean) | 0.2, 0.4, 0.6, 0.8, 1.2, 1.5, 2.0, 3.0, 4.0 |
| **JPEG compression** | q (quality, lower = more compression) | 80, 60, 40, 20, 10, 5 |
| **Motion blur** | k (horizontal kernel size in pixels) | 3, 5, 9, 13, 17, 21, 29 |

β=1, α=1, and σ=0 correspond to the unperturbed original. Brightness and contrast values below 1 reduce intensity/contrast; values above 1 increase it.

---

## Project structure

```
.
├── config/
│   └── perturbations.json          # perturbation sweep definitions
├── data/                           # required data files (not in repo, see Setup)
├── output/                         # generated outputs (see Output structure)
└── src/
    ├── notebook.ipynb              # interactive analysis
    ├── extract.py                  # feature extraction entry point
    ├── analysis.py                 # analysis + plotting entry point
    ├── perturbation_examples.py    # perturbation visualisation entry point
    ├── common/
    │   ├── config.py               # shared path and default constants
    │   ├── perturbations.py        # perturbation definitions and loader
    │   ├── seed.py                 # reproducibility helpers
    │   └── utils.py                # shared utilities
    ├── data/
    │   ├── dataset.py              # RFWPairsDataset (torch Dataset)
    │   └── features.py             # feature extraction, saving, and loading
    ├── model/
    │   ├── model.py                # model loading and image transforms
    │   └── iresnet.py              # IResNet-34 architecture
    └── evaluation/
        ├── metrics.py              # EER, FNMR@FMR, FDR, DoB computation
        ├── report.py               # report dataclasses and CSV generation
        └── visualize.py            # all plotting functions
```

---

## Output structure

```
output/
├── verification_report.csv              # one row per (perturbation, param, group)
├── fairness_report.csv                  # one row per (perturbation, param)
├── features/
│   └── <race>/<condition>/
│       ├── similarities.npy
│       └── labels.npy
├── perturbation_examples/
│   └── <perturbation_name>.{png,svg,pdf}
└── plots/
    ├── <race>/<condition>/score_distribution.{png,svg,pdf}
    ├── overall/<condition>/score_distribution.{png,svg,pdf}
    ├── roc_det/<perturbation>/<condition>.{png,svg,pdf}
    ├── EER/<perturbation>.{png,svg,pdf}
    ├── ZeroFMR/<perturbation>.{png,svg,pdf}
    ├── FMR100/<perturbation>.{png,svg,pdf}
    ├── FMR1000/<perturbation>.{png,svg,pdf}
    ├── FDR/<perturbation>.{png,svg,pdf}
    └── DoB/<perturbation>.{png,svg,pdf}
```

---

## Reports

### Verification report (`verification_report.csv`)

One row per `(perturbation, param, group)` combination. Groups are the four demographic groups plus `overall` (scores pooled across all groups).

| Column | Description |
|--------|-------------|
| `perturbation` | Perturbation type name (e.g. `gaussian_blur`) or `original` |
| `param` | Parameter value as `key=value` string (e.g. `σ=1.0`) |
| `group` | Demographic group or `overall` |
| `EER` | Equal Error Rate |
| `EER_threshold` | Decision threshold at EER |
| `ZeroFMR_FMR` | FMR at ZeroFMR operating point (≈ 0) |
| `ZeroFMR_FNMR` | FNMR at ZeroFMR operating point |
| `ZeroFMR_threshold` | Threshold at ZeroFMR |
| `FMR100_FMR` | FMR at FMR100 operating point (≈ 0.01) |
| `FMR100_FNMR` | FNMR at FMR = 1% |
| `FMR100_threshold` | Threshold at FMR100 |
| `FMR1000_FMR` | FMR at FMR1000 operating point (≈ 0.001) |
| `FMR1000_FNMR` | FNMR at FMR = 0.1% |
| `FMR1000_threshold` | Threshold at FMR1000 |

### Fairness report (`fairness_report.csv`)

One row per `(perturbation, param)` combination. Fairness metrics are computed across the four demographic groups (the `overall` pooled group is excluded).

| Column | Description |
|--------|-------------|
| `perturbation` | Perturbation type name or `original` |
| `param` | Parameter value as `key=value` string |
| `<OP>_FDR` | Fairness Discrimination Rate at operating point OP (see below) |
| `<OP>_FDR_A` | FDR component A: max pairwise FMR range across groups |
| `<OP>_FDR_B` | FDR component B: max pairwise FNMR range across groups |
| `<OP>_DoB` | Degree of Bias at operating point OP (see below) |

`<OP>` is one of `EER`, `ZeroFMR`, `FMR100`, `FMR1000`.

#### Fairness Discrimination Rate (FDR)

FDR measures how uniformly the system performs across groups at a fixed operating point. It is defined as:

```
FDR(τ) = 1 − (α · A + (1−α) · B)
```

where A is the range of FMR across groups, B is the range of FNMR across groups, and α=0.5 by default. **FDR = 1 means perfectly fair; FDR = 0 means maximally biased.**

#### Degree of Bias (DoB)

DoB is the population standard deviation of FNMR across the demographic groups at a fixed operating point. A higher DoB indicates greater disparity in error rates across groups.
