# CSE427 — DGA Killer: Semantic Coherence & Transformer Architectures for Zero-Day Wordlist DGA Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Status: Completed](https://img.shields.io/badge/Status-Completed-success.svg)]()

This repository contains the complete experimental framework, models, datasets, and evaluation artifacts for **CSE427: Machine Learning in Cybersecurity**. 

The project investigates whether integrating explicit word-level semantic coherence into character-level classifiers improves generalization against **zero-day wordlist-based Domain Generation Algorithms (DGAs)** under a strict **Leave-Families-Out (LFO)** protocol.

---

## 1. Executive Summary & Benchmark Results

We conducted a one-shot final evaluation on a frozen, leak-free test set of **19,998 domains** (10,000 DGA across 5 unseen families + 9,998 benign domains):
- **Wordlist Zero-Day Families:** `ngioweb`, `pizd`, `matsnu` (hard compound case)
- **Non-Wordlist Zero-Day Families:** `tinba`, `murofet`
- **Benign Control Set:** Alexa / Tranco top domains

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

### Key Findings:
1. **Beats Baseline Floor by 3.8×:** Surpasses the Drichel et al. (RAID 2022) 19.4% F1 baseline by a massive margin (ModernBERT: **73.72%**, Dual-Branch Coherent: **67.99%**).
2. **Dual-Branch Coherent Wins on `matsnu`:** Explicit FastText pairwise cosine similarity captures semantic incongruity between dictionary compounds, beating ModernBERT (**17.60% vs 14.15%**).
3. **Flawless Non-Wordlist Generalization:** All deep models achieved **99.9% to 100.0%** detection on unseen non-wordlist DGAs.
4. **Operational Specificity:** ModernBERT delivers exceptional benign specificity (**95.56%**, only 444 false positives out of 9,998), while the soft-voting Ensemble provides an optimal operational trade-off (**94.88% specificity**, **100% non-wordlist recall**).

---

## 2. Repository Structure

```
g:/dga_killer/
├── CSE427_DGA_MASTER_PLAN.md      # Research protocol, hypothesis & progress tracker
├── README.md                      # Project landing documentation
│
├── data/
│   ├── processed/                 # Frozen train (1.26M), val (42k), test (20k) datasets
│   └── vocab/                     # Character vocabulary dictionary
│
├── eda/                           # Exploratory data analysis notebooks
│   ├── 00_comprehensive_eda_report.ipynb
│   ├── 01_eda_raw.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_eda_after_and_comparison.ipynb
│
├── training/                      # Model training & benchmark notebooks
│   ├── MODERNBERT-FINETUNED.ipynb # Fine-tuned ModernBERT on full 1.26M dataset
│   ├── Dual-Branch PERFECT.ipynb  # 4.1M Dual-Branch character + word BiLSTM
│   ├── Dual-Branch with Explicit Semantic Coherence.ipynb # 0.73M semantic cosine model
│   ├── 06_final_test_evaluation.ipynb # FULLY EXECUTED final benchmark notebook
│   ├── FINAL-EVALUATION.ipynb     # (Mirror copy of 06_final_test_evaluation.ipynb)
│   ├── 01_rf_baseline.ipynb       # Random Forest (11 lexical features)
│   ├── 02_xgboost_baseline.ipynb  # XGBoost (11 lexical features)
│   └── weights/                   # Trained model weights & FastText binary
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
│   └── metrics/                   # Complete evaluation JSON metrics
│
├── scripts/                       # Reusable automation and execution scripts
│   ├── run_final_master_evaluation.py # Standalone evaluation pipeline
│   └── build_rich_06_notebook.py      # Notebook generator
│
└── requirements.txt               # Complete pinned python dependencies
```

---

## 3. Quickstart & How to View

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. View Executed Notebooks
- **Final Benchmark Evaluation:** Open [`training/06_final_test_evaluation.ipynb`](training/06_final_test_evaluation.ipynb) or [`training/FINAL-EVALUATION.ipynb`](training/FINAL-EVALUATION.ipynb). All cells are pre-executed with interactive tables and embedded high-resolution figures.
- **ModernBERT Fine-Tuned:** Open [`training/MODERNBERT-FINETUNED.ipynb`](training/MODERNBERT-FINETUNED.ipynb) to inspect the full 4-epoch training dynamics and evaluation on 1.26M domains.
- **Dual-Branch Architectures:** Open [`training/Dual-Branch PERFECT.ipynb`](training/Dual-Branch PERFECT.ipynb) and [`training/Dual-Branch with Explicit Semantic Coherence.ipynb`](training/Dual-Branch with Explicit Semantic Coherence.ipynb).

### 3. Re-Run Master Benchmark
To recompute raw predictions across all models on your local GPU:
```bash
python scripts/run_final_master_evaluation.py
```

### 4. Academic Paper Tables
Ready-to-copy LaTeX code is located in:
- [`reports/tables/final_test_benchmark_table.tex`](reports/tables/final_test_benchmark_table.tex)
- [`reports/tables/final_test_detailed_metrics.tex`](reports/tables/final_test_detailed_metrics.tex)

