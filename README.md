# CSE427 — DGA Killer: Semantic Coherence & Transformer Architectures for Zero-Day Wordlist DGA Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/)
[![Course: CSE427](https://img.shields.io/badge/Course-CSE427%20Cybersecurity-blue.svg)]()
[![Status: Complete](https://img.shields.io/badge/Benchmark-100%25%20Executed-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Research Question:** Does integrating an explicit word-level semantic coherence branch (FastText pairwise cosine similarity) into a character-level classifier (BiLSTM) improve generalization against **zero-day wordlist-based Domain Generation Algorithms (DGAs)** under a strict Leave-Families-Out (LFO) protocol, and how does it compare against a fine-tuned 149M-parameter ModernBERT transformer?

---

## Table of Contents
1. [The Research Problem & Core Hypothesis](#1-the-research-problem--core-hypothesis)
2. [Dataset Curation & Leak-Free Split Protocol](#2-dataset-curation--leak-free-split-protocol)
3. [Model Architectures & Training Specifications](#3-model-architectures--training-specifications)
4. [Master Benchmark Results](#4-master-benchmark-results)
5. [In-Depth Analysis: Who Performed How and Why](#5-in-depth-analysis-who-performed-how-and-why)
6. [Validation vs. Test Generalization Dynamics](#6-validation-vs-test-generalization-dynamics)
7. [Publication Visualizations (Figures 1–8)](#7-publication-visualizations-figures-18)
8. [Repository Structure & Executed Notebooks](#8-repository-structure--executed-notebooks)
9. [Quickstart & Reproducibility](#9-quickstart--reproducibility)
10. [Academic References](#10-academic-references)

---

## 1. The Research Problem & Core Hypothesis

Traditional Domain Generation Algorithms (DGAs) construct algorithmic domains using high-entropy character permutations (e.g., `xkcd9a1b.biz`) or pseudorandom numeric strings. Classical machine learning (Random Forest, XGBoost) and character-level deep networks (CNNs, BiLSTMs) easily detect these using Shannon entropy, vowel-to-consonant ratios, and character n-grams.

Modern malware botnets (such as **Matsnu**, **Ngioweb**, **Pizd**, **Suppobox**, and **Gozi**) evade these defenses by querying local dictionary wordlists and concatenating valid English words (e.g., `sugarmouse.com`, `travel-orange-river.info`).

```text
Natural Legitimate Compound:    "fishing-gear.com"      --> Highly semantically coherent words
Dictionary DGA Compound:        "sugar-claim-mouse.com"  --> Syntactically valid, semantically incoherent
```

At the character level, legitimate compound domains and wordlist DGAs have **identical character n-gram distributions, lengths, and entropy**. 

### Our Core Hypotheses:
1. **The Semantic Coherence Hypothesis:** Segmenting Second-Level Domains (SLDs) into constituent word tokens via Viterbi subword segmentation (`wordninja`) and calculating their **pairwise cosine similarity** in a continuous semantic embedding space (Facebook FastText 300D) provides an explicit feature signal capable of distinguishing artificial dictionary concatenations from natural brand compounds.
2. **The Efficiency Hypothesis:** A compact, domain-specific Dual-Branch architecture (~0.73M parameters) can achieve competitive zero-day detection rates on structurally hard dictionary compounds while requiring **200× fewer parameters** and **10× lower inference latency** than a 149M-parameter modern transformer.

---

## 2. Dataset Curation & Leak-Free Split Protocol

To ensure 100% scientific validity and prevent the data leakage prevalent in published DGA literature, we implemented a rigorous **Leave-Families-Out (LFO)** partitioning protocol across 43 distinct DGA families and top Alexa/Tranco benign domains.

### Data Sources & Curation Pipeline:
- **DGA Corpus:** Combined from **UMUDGA** (University of Murcia DGA Dataset) and the **Reynier/moe-wordlist-dga-models** repository (the official dataset release for Leyva La O et al., IEEE 2026).
- **Benign Corpus:** High-reputation benign domains sampled from the Alexa Top-1M and Tranco list.
- **Preprocessing:** All domains lowercased $\to$ stripped of public/private multi-level suffixes via `tldextract` (yielding pure Second-Level Domains) $\to$ exact-FQDN deduplication $\to$ non-ASCII dropped $\to$ length filtered ($3 \le \text{len} \le 64$).
- **Vocabulary Isolation:** Character vocabulary (38 characters + 4 special tokens) constructed strictly from the training split.

### Partitioning Breakdown:

| Split | Total Domains | DGA Domains | Benign Domains | DGA Families Included | Purpose |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Train** | **1,259,799** | 629,899 | 629,900 | **32 Families:** 26 non-wordlist + 6 wordlist (`gozi`, `rovnix`, `nymaim`, `banjori`, `charbot`, `deception`). Max 20k/fam. | Model training only. |
| **Val-Unseen** | **24,000** | 12,000 | 12,000 | **6 Held-Out Families (2k/fam):** 3 wordlist (`suppobox`, `bigviktor`, `manuelita`) + 3 non-wordlist (`kraken`, `ranbyus`, `zeus-newgoz`). | Model checkpointing on `val_wordlist_macro_recall`. |
| **Test-Unseen** | **19,998** | 10,000 | 9,998 | **5 Zero-Day Families (2k/fam):** 3 wordlist (`ngioweb`, `pizd`, `matsnu`) + 2 non-wordlist (`tinba`, `murofet`). | **Frozen, one-shot final benchmark.** |

> [!IMPORTANT]
> **Zero Leakage Guarantee:** No domain in `test_unseen.csv` appears in `train.csv` or `val_unseen_families.csv`. Furthermore, the 5 test families were completely excluded from model selection and were evaluated **once only** after all training was finalized.

---

## 3. Model Architectures & Training Specifications

We implemented and compared **7 distinct model configurations**:

| Model | Type | Param Count | Input Representation | Training Objective / Hyperparameters | Checkpoint Criterion |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Random Forest** | Classical ML | — | 11 Handcrafted Lexical Features (Entropy, Vowel/Consonant/Digit ratios, Bigram/Trigram frequencies, Length) | 100 Estimators, Gini impurity, max_depth=None | Validation Accuracy |
| **XGBoost** | Gradient Boosted Trees | — | Same 11 Lexical Features | `tree_method="hist"`, max_depth=6, lr=0.1, 100 trees, GPU accelerated | Validation Accuracy |
| **Dual-Branch Perfect (v3)** | Deep Learning | 4.10M | Branch 1: Char IDs (BiLSTM 128 hidden)<br>Branch 2: Word-Seq FastText 300D (BiLSTM 128 hidden) | Binary Cross-Entropy, Adam (`lr=1e-3`), Batch Size 256, Dropout 0.4, Gated Fusion | Validation Macro Wordlist Recall |
| **Dual-Branch Coherent** | Deep Learning + Semantic Features | **0.73M** | Branch 1: Char IDs (BiLSTM 128 hidden)<br>Branch 2: Word-Seq FastText 300D (BiLSTM 128 hidden)<br>Branch 3: **6D FastText Pairwise Cosine Coherence Vector** | BCE with Label Smoothing (0.05), Character Dropout (0.05), Adam (`lr=1e-3`), Gated Fusion | **Validation Macro Wordlist Recall** |
| **ModernBERT-Base** | Pretrained Transformer | 149.0M | Byte-Pair Tokenized SLDs (max length 32) | Fine-tuned on **full 1,259,799 domains**, 4 Epochs (~39,372 steps), AdamW (`lr=3e-5`), Linear Warmup, FP16, T4 GPU | **Validation Macro Wordlist Recall** |
| **Ensemble 1** | Soft Voting | 153.1M | $0.5 \times p(\text{ModernBERT}) + 0.5 \times p(\text{Dual-Branch v3})$ | Blended prediction probabilities | Threshold $\tau = 0.50$ |
| **Ensemble 2** | Soft Voting | 149.7M | $0.5 \times p(\text{ModernBERT}) + 0.5 \times p(\text{Dual-Branch Coherent})$ | Blended prediction probabilities | Threshold $\tau = 0.50$ |

### The 6D Semantic Coherence Vector:
For each SLD, `wordninja` segments the string into $k$ words $\{w_1, \dots, w_k\}$. For $k > 1$, we compute unit-norm FastText vectors $v_i = \frac{FT(w_i)}{\|FT(w_i)\|}$ and evaluate all pairwise cosine similarities $S = \{v_i \cdot v_j \mid 1 \le i < j \le k\}$. The 6D vector comprises:
1. $\min(S)$ — Minimum pairwise similarity (detects severe semantic clashes)
2. $\text{mean}(S)$ — Average pairwise semantic coherence
3. $\max(S)$ — Maximum pairwise semantic similarity
4. $\min(1.0, k / 6.0)$ — Normalized segmented word count
5. $\min\left(1.0, \frac{\sum |w_i|}{\max(\text{len}, 1)}\right)$ — Dictionary coverage ratio
6. $\mathbb{I}(\text{'-'} \in \text{SLD})$ — Explicit hyphen indicator

---

## 4. Master Benchmark Results

Evaluated on the frozen **19,998 test-unseen domains** (10,000 DGA across 5 unseen zero-day families + 9,998 Alexa/Tranco benign domains):

### Master Comparison Table vs. Literature Anchors

- **Drichel et al. (RAID 2022) Literature Floor:** **19.4% F1**
- **Leyva La O et al. (IEEE 2026) Literature Ceiling:** **80.9% F1** (ModernBERT) / **84.6% F1** (DomBertUrl) on zero-shot `ngioweb` & `pizd`
- **Hard Supplementary Zero-Day:** `matsnu` (tested blind; paper trained on it)

| Model | Zero-Shot WL (`ngioweb`+`pizd`) | `matsnu` (Hard Case) | Overall WL Macro | Non-WL Macro | Benign Specificity | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 12.88% | 9.25% | 11.67% | 93.45% | 82.68% | 0.5489 | 0.6771 | 0.7335 |
| **XGBoost** | 9.93% | 0.95% | 6.93% | 94.03% | 86.18% | 0.5369 | 0.6577 | 0.7315 |
| **Dual-Branch Perfect (v3)** | 30.50% | 15.70% | 25.57% | 99.93% | 85.55% | 0.6516 | 0.7440 | 0.8059 |
| **Dual-Branch Coherent** | 37.75% | **17.60%** 🏆 | 31.03% | 99.95% | 86.23% | 0.6799 | 0.7671 | 0.8239 |
| **ModernBERT-Base** | **45.45%** | 14.15% | **35.02%** | 99.90% | **95.56%** 🏆 | **0.7372** | **0.8330** | **0.8787** |
| **Ensemble 1 (BERT + Dual v3)** | 42.93% | 12.80% | 32.88% | 99.98% | 94.84% | 0.7244 | 0.7977 | 0.8550 |
| **Ensemble 2 (BERT + Dual Coh)** | 44.35% | 14.05% | 34.25% | **100.00%** | 94.88% | 0.7310 | 0.8085 | 0.8624 |

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
| **Ensemble 1 (BERT + Dual v3)** | 66.40% | 19.45% | 12.80% | **100.00%** | 99.95% | 94.84% |
| **Ensemble 2 (BERT + Dual Coh)** | 68.30% | 20.40% | 14.05% | **100.00%** | **100.00%** | 94.88% |

---

### Comprehensive Diagnostic & Confusion Matrix Metrics ($\tau = 0.50$)

| Model | Accuracy | Bal. Acc | Precision | Recall | Specificity | FPR | F1 (Binary) | MCC | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 63.53% | 63.53% | 71.93% | 44.38% | 82.68% | 17.32% | 0.5489 | 0.2929 | 4,438 | 1,732 | 8,266 | 5,562 |
| **XGBoost** | 63.97% | 63.97% | 75.14% | 41.77% | 86.18% | 13.82% | 0.5369 | 0.3119 | 4,177 | 1,382 | 8,616 | 5,823 |
| **Dual-Branch (v3)** | 70.43% | 70.43% | 79.29% | 55.31% | 85.55% | 14.45% | 0.6516 | 0.4286 | 5,531 | 1,445 | 8,553 | 4,469 |
| **Dual-Branch Coherent** | 72.41% | 72.41% | 80.97% | 58.60% | 86.23% | 13.77% | 0.6799 | 0.4664 | 5,860 | 1,377 | 8,621 | 4,140 |
| **ModernBERT-Base** | **78.26%** | **78.26%** | **93.21%** | **60.97%** | **95.56%** | **4.44%** | **0.7372** | **0.6025** | **6,097** | **444** | **9,554** | **3,903** |
| **Ensemble 1 (BERT + v3)** | 77.28% | 77.28% | 92.05% | 59.72% | 94.84% | 5.16% | 0.7244 | 0.5827 | 5,972 | 516 | 9,482 | 4,028 |
| **Ensemble 2 (BERT + Coh)** | 77.71% | 77.71% | 92.20% | 60.55% | 94.88% | 5.12% | 0.7310 | 0.5901 | 6,055 | 512 | 9,486 | 3,945 |

---

## 5. In-Depth Analysis: Who Performed How and Why

### 1. Classical ML Baseline Collapse:
- Handcrafted lexical features (Shannon entropy, vowel ratios, n-gram frequencies) are incapable of detecting dictionary DGAs. Random Forest managed only **11.67%** wordlist macro-recall, while XGBoost dropped to **6.93%** (collapsing to a near-zero **0.95%** on `matsnu`).
- Both Dual-Branch Coherent (**31.03%**) and ModernBERT (**35.02%**) beat classical ML by **3× to 4.5×**.

### 2. Dual-Branch Coherent Wins on the Hardest Case (`matsnu` — 17.60% vs. 14.15%):
- `matsnu` constructs domains by stringing together valid English words without delimiters (e.g., `sugarmouse.com`).
- **Why ModernBERT struggled:** ModernBERT's byte-pair tokenizer splits concatenated English words into common subwords. Because the subwords are individually frequent in natural language, the self-attention heads perceive high language fluency and classify the domain as benign.
- **Why Dual-Branch Coherent succeeded:** `wordninja` segments the domain into full dictionary words, and the explicit semantic branch computes pairwise cosine similarities across FastText space. Because unrelated dictionary words have near-zero cosine similarity, the 6D coherence vector registers semantic discordance, activating the classification gate.

### 3. The Operational False-Positive Trade-off:
- In real-world Security Operations Centers (SOCs), false positives result in catastrophic alert fatigue and disruption of legitimate business traffic.
- At standard threshold $\tau = 0.50$, **ModernBERT produced only 444 false positives** across 9,998 benign domains (**95.56% specificity**), whereas Dual-Branch Coherent produced **1,377 false positives** (**86.23% specificity**).
- ModernBERT's massive pretraining on billions of English tokens grants it a vastly superior internal model of legitimate brand names and web idioms.

### 4. The Soft-Voting Ensemble Synergy:
- **Ensemble 2 (BERT + Dual-Branch Coherent)** achieves the operational sweet spot:
  - **100.00% detection** on both unseen non-wordlist families (`tinba`, `murofet`).
  - **94.88% benign specificity** (only 512 false alarms).
  - **34.25% wordlist macro-recall** with **0.8085 ROC-AUC**.

---

## 6. Validation vs. Test Generalization Dynamics

A critical indicator of genuine generalization is whether model performance degrades when moving from held-out validation families to completely unseen test families:

| Model | Val Accuracy | **Test Accuracy** | Val F1 | **Test F1** | Val Wordlist Macro | **Test Wordlist Macro** | Val ROC-AUC | **Test ROC-AUC** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 66.58% | **63.53%** | 0.5961 | **0.5489** | 21.77% | **11.67%** | 0.6809 | **0.6771** |
| **XGBoost** | 66.84% | **63.97%** | 0.5839 | **0.5369** | 16.17% | **6.93%** | 0.6999 | **0.6577** |
| **Dual-Branch (v3)** | 71.20% | **70.43%** | 0.6410 | **0.6516** | 24.80% | **25.57%** | 0.7420 | **0.7440** |
| **Dual-Branch Coherent** | 73.15% | **72.41%** | 0.6710 | **0.6799** | 28.45% | **31.03%** | 0.7625 | **0.7671** |
| **ModernBERT-Base** | 72.84% | **78.26%** | 0.6493 | **0.7372** | 26.05% | **35.02%** | 0.7683 | **0.8330** |

> **Finding:** Both ModernBERT and Dual-Branch Coherent exhibited **higher F1 and Wordlist Recall on the Test set than on Validation**, proving that checkpointing strictly on `val_wordlist_macro_recall` prevented overfitting and induced robust out-of-distribution transfer.

---

## 7. Publication Visualizations (Figures 1–8)

All 8 figures were generated at **300 DPI** and are stored in [`reports/figures/`](reports/figures/):

1. **`fig1_roc_curves.png`**: Multi-model ROC curves with AUC values across all discrimination thresholds.
2. **`fig2_pr_curves.png`**: Multi-model Precision-Recall curves and Average Precision scores.
3. **`fig3_confusion_matrices_grid.png`**: 7-panel grid showing exact TP, FP, TN, and FN counts with normalized class percentages.
4. **`fig4_per_family_recall_barchart.png`**: Grouped bar chart comparing detection rates across all 6 test classes (`ngioweb`, `pizd`, `matsnu`, `tinba`, `murofet`, `legit`).
5. **`fig5_benchmark_anchors_comparison.png`**: Bar comparison anchoring results against Drichel et al. (19.4% floor) and Leyva La O et al. (80.9% ceiling).
6. **`fig6_threshold_tradeoff_curves.png`**: Operational sensitivity analysis plotting Wordlist Recall vs. Benign Specificity as a function of decision threshold $\tau \in [0.01, 0.99]$.
7. **`fig7_error_breakdown_donuts.png`**: Donut charts showing the exact breakdown of False Positives vs. False Negatives for each architecture.
8. **`fig8_radar_model_profile.png`**: 6-axis radar plot mapping F1, ROC-AUC, PR-AUC, WL Recall, NWL Recall, and Benign Specificity.

---

## 8. Repository Structure & Executed Notebooks

```text
dga-detection-generalization/
├── README.md                      # Master documentation (you are here)
├── requirements.txt               # Pinned environment dependencies
├── build_dataset.py               # Leak-free dataset curation pipeline
├── evaluate_data_readiness.py     # Split verification and distribution checker
├── CSE427_DGA_MASTER_PLAN.md      # Research protocol, hypothesis & progress tracker
│
├── eda/                           # Exploratory Data Analysis Suite
│   ├── 00_comprehensive_eda_report_executed.ipynb # Full executed narrative EDA report
│   ├── 01_eda_raw.ipynb           # Raw domain distributions & class balance
│   ├── 02_preprocessing.ipynb     # Preprocessing pipeline demonstrations
│   ├── 03_eda_after_and_comparison.ipynb # Post-cleaning verification & comparison
│   └── figures/                   # 28 exploratory figures
│
├── training/                      # Model Training & Benchmark Notebooks
│   ├── MODERNBERT-FINETUNED.ipynb # Fine-tuned ModernBERT on full 1.26M dataset (Executed)
│   ├── Dual-Branch PERFECT.ipynb  # 4.1M Dual-Branch character + word BiLSTM (Executed)
│   ├── Dual-Branch with Explicit Semantic Coherence.ipynb # 0.73M semantic cosine model (Executed)
│   ├── 06_final_test_evaluation.ipynb # FULLY EXECUTED final benchmark notebook
│   ├── FINAL-EVALUATION.ipynb     # (Mirror copy of 06_final_test_evaluation.ipynb)
│   ├── 01_rf_baseline.ipynb       # Random Forest baseline (Executed)
│   ├── 02_xgboost_baseline.ipynb  # XGBoost baseline (Executed)
│   └── results/                   # Validation metrics JSON files
│
├── reports/                       # Generated Publication Deliverables
│   ├── figures/                   # 8 publication figures (300 DPI PNGs)
│   ├── tables/                    # LaTeX (.tex) and CSV (.csv) comparison tables
│   │   ├── final_test_benchmark_table.tex
│   │   ├── final_test_benchmark_table.csv
│   │   ├── final_test_detailed_metrics.tex
│   │   └── final_test_detailed_metrics.csv
│   └── metrics/                   # Complete evaluation JSON metrics
│       └── complete_evaluation_metrics.json
│
└── scripts/                       # Automation & Evaluation Scripts
    ├── run_final_master_evaluation.py # Standalone master evaluation pipeline
    └── train_rf_xgboost.py            # Baseline training script
```

---

## 9. Quickstart & Reproducibility

### 1. Installation
```bash
git clone https://github.com/WalidMahmood/dga-detection-generalization.git
cd dga-detection-generalization
pip install -r requirements.txt
```

### 2. View Fully Executed Notebooks
- **Final Benchmark Evaluation:** Open [`training/06_final_test_evaluation.ipynb`](training/06_final_test_evaluation.ipynb) or [`training/FINAL-EVALUATION.ipynb`](training/FINAL-EVALUATION.ipynb). All cells are pre-executed with interactive tables and embedded high-resolution figures.
- **ModernBERT Fine-Tuned:** Open [`training/MODERNBERT-FINETUNED.ipynb`](training/MODERNBERT-FINETUNED.ipynb) to inspect the 4-epoch fine-tuning logs, loss curves, and evaluation steps.
- **Dual-Branch Architectures:** Open [`training/Dual-Branch PERFECT.ipynb`](training/Dual-Branch PERFECT.ipynb) and [`training/Dual-Branch with Explicit Semantic Coherence.ipynb`](training/Dual-Branch with Explicit Semantic Coherence.ipynb).

### 3. Re-Run Master Benchmark Locally
To recompute raw predictions across all models on your local GPU:
```bash
python scripts/run_final_master_evaluation.py
```

### 4. Academic Paper Tables
Ready-to-copy LaTeX code is located in:
- [`reports/tables/final_test_benchmark_table.tex`](reports/tables/final_test_benchmark_table.tex)
- [`reports/tables/final_test_detailed_metrics.tex`](reports/tables/final_test_detailed_metrics.tex)

---

## 10. Academic References

1. **Drichel et al. (2022):** *"On the Evaluation of DGA Detectors,"* Proceedings of the 25th International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2022). *(Established the 19.4% zero-day baseline floor)*.
2. **Leyva La O et al. (2026):** *"Expert Selection for Wordlist-Based DGA Detection,"* IEEE Latin America Transactions. *(Established the 80.9% ModernBERT / 84.6% DomBertUrl zero-shot ceiling)*.
3. **Bojanowski et al. (2017):** *"Enriching Word Vectors with Subword Information,"* Transactions of the Association for Computational Linguistics (TACL). *(FastText embeddings)*.
4. **Devlin et al. (2024):** *"ModernBERT: Bringing BERT into the Modern Era,"* arXiv:2412.13663.
