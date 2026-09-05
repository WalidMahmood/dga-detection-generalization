# CSE427 — DGA Killer: Semantic Coherence & Transformer Architectures for Zero-Day Wordlist DGA Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/)
[![Course: CSE427](https://img.shields.io/badge/Course-CSE427%20Cybersecurity-blue.svg)]()
[![Status: Complete](https://img.shields.io/badge/Benchmark-100%25%20Executed-success.svg)]()

This repository contains the complete research pipeline, model architectures, training notebooks, and empirical benchmark evaluation for **CSE427: Machine Learning in Cybersecurity**.

The project investigates whether integrating explicit word-level semantic coherence into character-level classifiers improves generalization against **zero-day wordlist-based Domain Generation Algorithms (DGAs)** under a strict **Leave-Families-Out (LFO)** protocol.

---

## 1. Executive Summary & Benchmark Results

We conducted a one-shot final evaluation on a frozen, leak-free test set of **19,998 domains** (10,000 DGA across 5 unseen zero-day families + 9,998 Alexa/Tranco benign domains):
- **Zero-Day Wordlist Families:** `ngioweb`, `pizd`, `matsnu` (hard compound case)
- **Zero-Day Non-Wordlist Families:** `tinba`, `murofet`
- **Benign Control Set:** Alexa & Tranco top domains

### Master Benchmark Comparison Table

| Model | Zero-Shot WL (`ngioweb`+`pizd`) | `matsnu` (Hard Case) | Overall WL Macro | Non-WL Macro | Benign Specificity | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 12.88% | 9.25% | 11.67% | 93.45% | 82.68% | 0.5489 | 0.6771 | 0.7335 |
| **XGBoost** | 9.93% | 0.95% | 6.93% | 94.03% | 86.18% | 0.5369 | 0.6577 | 0.7315 |
| **Dual-Branch Perfect (v3)** | 30.50% | 15.70% | 25.57% | 99.93% | 85.55% | 0.6516 | 0.7440 | 0.8059 |
| **Dual-Branch Coherent** | 37.75% | **17.60%** 🏆 | 31.03% | 99.95% | 86.23% | 0.6799 | 0.7671 | 0.8239 |
| **ModernBERT-Base** | **45.45%** | 14.15% | **35.02%** | 99.90% | **95.56%** 🏆 | **0.7372** | **0.8330** | **0.8787** |
| **Ensemble (BERT + Dual v3)** | 42.93% | 12.80% | 32.88% | 99.98% | 94.84% | 0.7244 | 0.7977 | 0.8550 |
| **Ensemble (BERT + Dual Coh)** | 44.35% | 14.05% | 34.25% | **100.00%** | 94.88% | 0.7310 | 0.8085 | 0.8624 |

---

### Per-Family Detection Rates (Recall on DGA / Specificity on Legit)

Every test family contains exactly 2,000 domains (plus 9,998 benign domains):

| Model | `ngioweb` (WL) | `pizd` (WL) | `matsnu` (Hard WL) | `tinba` (NWL) | `murofet` (NWL) | `legit` (Benign) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 9.10% | 16.65% | 9.25% | 93.85% | 93.05% | 82.68% |
| **XGBoost** | 6.30% | 13.55% | 0.95% | 94.85% | 93.20% | 86.18% |
| **Dual-Branch Perfect (v3)** | 49.85% | 11.15% | 15.70% | 99.90% | 99.95% | 85.55% |
| **Dual-Branch Coherent** | 57.10% | 18.40% | **17.60%** 🏆 | 99.90% | **100.00%** | 86.23% |
| **ModernBERT-Base** | **68.25%** | **22.65%** | 14.15% | 99.85% | 99.95% | **95.56%** 🏆 |
| **Ensemble (BERT + Dual v3)** | 66.40% | 19.45% | 12.80% | **100.00%** | 99.95% | 94.84% |
| **Ensemble (BERT + Dual Coh)** | 68.30% | 20.40% | 14.05% | **100.00%** | **100.00%** | 94.88% |

---

### Key Scientific Takeaways:
1. **Literature Baseline Surpassed by 3.8×:** Surpasses the Drichel et al. (RAID 2022) 19.4% F1 baseline across all deep models (ModernBERT: **73.72%**, Dual-Branch Coherent: **67.99%**).
2. **Dual-Branch Coherent Outperforms ModernBERT on Hardest Case (`matsnu`):** On `matsnu`, which concatenates natural dictionary words without delimiters, **Dual-Branch Coherent achieved 17.60% vs. ModernBERT's 14.15%**. Explicit pairwise FastText cosine similarity identifies semantic incongruity where subword tokenizers alone falter.
3. **Flawless Non-Wordlist Generalization:** All deep models achieved **99.9% to 100.0%** detection on unseen non-wordlist DGAs (`tinba`, `murofet`).
4. **Operational Specificity:** ModernBERT delivers superior benign specificity (**95.56%**, only 444 false positives out of 9,998). The soft-voting Ensemble provides an optimal operational trade-off (**94.88% specificity**, **100.00% non-wordlist recall**).

---

## 2. Repository Structure

```text
dga_killer/
├── CSE427_DGA_MASTER_PLAN.md      # Research protocol, hypothesis & progress tracker
├── README.md                      # Project documentation and benchmark overview
├── requirements.txt               # Complete pinned python dependencies
├── build_dataset.py               # Leakage-free dataset curation pipeline
├── evaluate_data_readiness.py     # Split verification and distribution checker
│
├── eda/                           # Exploratory Data Analysis suite
│   ├── 00_comprehensive_eda_report_executed.ipynb # Full executed EDA report
│   ├── 01_eda_raw.ipynb           # Raw distribution & class balance
│   ├── 02_preprocessing.ipynb     # Preprocessing pipeline demonstrations
│   ├── 03_eda_after_and_comparison.ipynb # Post-processing comparison
│   └── figures/                   # 28 EDA visualization figures
│
├── training/                      # Model training & benchmark notebooks
│   ├── MODERNBERT-FINETUNED.ipynb # Fine-tuned ModernBERT on full 1.26M dataset (Executed)
│   ├── Dual-Branch PERFECT.ipynb  # 4.1M Dual-Branch character + word BiLSTM (Executed)
│   ├── Dual-Branch with Explicit Semantic Coherence.ipynb # 0.73M semantic cosine model (Executed)
│   ├── 06_final_test_evaluation.ipynb # FULLY EXECUTED final benchmark notebook
│   ├── FINAL-EVALUATION.ipynb     # (Mirror copy of 06_final_test_evaluation.ipynb)
│   ├── 01_rf_baseline.ipynb       # Random Forest baseline (Executed)
│   └── 02_xgboost_baseline.ipynb  # XGBoost baseline (Executed)
│
├── reports/                       # Generated publication deliverables
│   ├── figures/                   # 8 publication figures (300 DPI PNGs)
│   │   ├── fig1_roc_curves.png
│   │   ├── fig2_pr_curves.png
│   │   ├── fig3_confusion_matrices_grid.png
│   │   ├── fig4_per_family_recall_barchart.png
│   │   ├── fig5_benchmark_anchors_comparison.png
│   │   ├── fig6_threshold_tradeoff_curves.png
│   │   ├── fig7_error_breakdown_donuts.png
│   │   └── fig8_radar_model_profile.png
│   ├── tables/                    # LaTeX (.tex) and CSV (.csv) comparison tables
│   │   ├── final_test_benchmark_table.tex
│   │   ├── final_test_benchmark_table.csv
│   │   ├── final_test_detailed_metrics.tex
│   │   └── final_test_detailed_metrics.csv
│   └── metrics/                   # Complete evaluation JSON metrics
│       └── complete_evaluation_metrics.json
│
└── scripts/                       # Reusable automation and execution scripts
    ├── run_final_master_evaluation.py # Standalone evaluation pipeline
    └── train_rf_xgboost.py            # Baseline training script
```

---

## 3. Quickstart & Reproducibility

### Step 1: Environment Setup
```bash
git clone https://github.com/WalidMahmood/dga-detection-generalization.git
cd dga-detection-generalization
pip install -r requirements.txt
```

### Step 2: View Fully Executed Notebooks
- **Final Benchmark Evaluation:** Open [`training/06_final_test_evaluation.ipynb`](training/06_final_test_evaluation.ipynb). All cells are pre-executed with interactive tables and embedded high-resolution figures.
- **ModernBERT Fine-Tuned:** Open [`training/MODERNBERT-FINETUNED.ipynb`](training/MODERNBERT-FINETUNED.ipynb) to inspect the 4-epoch fine-tuning logs, loss curves, and evaluation steps.
- **Dual-Branch Architectures:** Open [`training/Dual-Branch PERFECT.ipynb`](training/Dual-Branch PERFECT.ipynb) and [`training/Dual-Branch with Explicit Semantic Coherence.ipynb`](training/Dual-Branch with Explicit Semantic Coherence.ipynb).

### Step 3: Re-Run Master Benchmark Locally
To recompute raw predictions across all models on your local GPU:
```bash
python scripts/run_final_master_evaluation.py
```

### Step 4: Academic Paper Tables
Ready-to-copy LaTeX code is located in:
- [`reports/tables/final_test_benchmark_table.tex`](reports/tables/final_test_benchmark_table.tex)
- [`reports/tables/final_test_detailed_metrics.tex`](reports/tables/final_test_detailed_metrics.tex)
