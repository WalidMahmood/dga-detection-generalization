import json
from pathlib import Path

nb_path = Path(r"g:\dga_killer\training\06_final_test_evaluation.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 06 — Final Master Test-Unseen Benchmark & Comprehensive Model Evaluation\n",
            "\n",
            "> **Master Plan Phase 5 Protocol (§0, §2, §5, §6):** This benchmark evaluation is executed **ONCE ONLY** on the completely frozen `test_unseen.csv` (19,998 domains, 10,000 DGA across 5 unseen families + 9,998 Alexa/Tranco benign domains) after all models are fully trained.\n",
            ">\n",
            "> **Models Evaluated:**\n",
            "> 1. **Random Forest Baseline** (11 lexical/statistical features)\n",
            "> 2. **XGBoost Baseline** (11 lexical/statistical features)\n",
            "> 3. **Dual-Branch Perfect (v3)** (Bi-LSTM character + FastText word sequence, 4.1M params)\n",
            "> 4. **Dual-Branch Coherent** (Bi-LSTM character + FastText word sequence + 6D Semantic Cosine Coherence, 0.73M params)\n",
            "> 5. **ModernBERT-Base** (149M pretrained transformer fine-tuned on full 1.26M dataset)\n",
            "> 6. **Ensemble 1** (ModernBERT + Dual-Branch Perfect v3 soft voting)\n",
            "> 7. **Ensemble 2** (ModernBERT + Dual-Branch Coherent soft voting)\n",
            ">\n",
            "> **Benchmark Anchors:**\n",
            "> - **Floor:** Drichel et al. (RAID 2022) — **19.4% F1**\n",
            "> - **Ceiling:** Leyva La O et al. (IEEE 2026) — **80.9% F1** (ModernBERT) / **84.6% F1** (DomBertUrl) on zero-shot wordlist families (`ngioweb` & `pizd`)\n",
            "> - **Hard Supplementary Zero-Day:** `matsnu` (tested blind; paper trained on it)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, json, math\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "from IPython.display import Image, display, HTML\n",
            "\n",
            "# Paths setup\n",
            "BASE_DIR = Path(r\"G:\\dga_killer\")\n",
            "REPORTS_DIR = BASE_DIR / \"reports\"\n",
            "FIG_DIR = REPORTS_DIR / \"figures\"\n",
            "TABLE_DIR = REPORTS_DIR / \"tables\"\n",
            "METRIC_DIR = REPORTS_DIR / \"metrics\"\n",
            "\n",
            "print(f\"Reports directory: {REPORTS_DIR.resolve()}\")\n",
            "print(f\"Figures available: {len(list(FIG_DIR.glob('*.png')))} figures\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Master Benchmark Comparison Table\n",
            "\n",
            "The table below compares all 7 models against the primary evaluation criteria:\n",
            "- **Zero-Shot Wordlist Macro Recall** on `ngioweb` + `pizd` (directly comparable to Leyva La O et al. 2026)\n",
            "- **`matsnu` Recall** (the hardest dictionary-compound zero-day)\n",
            "- **Overall Wordlist Macro Recall** (`ngioweb`, `pizd`, `matsnu`)\n",
            "- **Non-Wordlist Macro Recall** (`tinba`, `murofet`)\n",
            "- **Benign Specificity (Legit Accuracy)** on 9,998 benign domains\n",
            "- **Overall Binary F1-Score, ROC-AUC, and PR-AUC (Average Precision)**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "benchmark_df = pd.read_csv(TABLE_DIR / \"final_test_benchmark_table.csv\")\n",
            "\n",
            "# Format nicely for display\n",
            "display_bench = benchmark_df.copy()\n",
            "pct_cols = [\"Zero-Shot WL (Ngioweb+Pizd)\", \"Matsnu (Hard Case)\", \"Overall Wordlist Macro\", \"Non-Wordlist Macro\", \"Benign Accuracy\"]\n",
            "for col in pct_cols:\n",
            "    display_bench[col] = display_bench[col].apply(lambda x: f\"{x*100:.2f}%\")\n",
            "for col in [\"F1-Score\", \"ROC-AUC\", \"PR-AUC\"]:\n",
            "    display_bench[col] = display_bench[col].apply(lambda x: f\"{x:.4f}\")\n",
            "\n",
            "display(HTML(\"<h3>Table 1: Final Test-Unseen Benchmark Comparison</h3>\"))\n",
            "display(display_bench)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Per-Family Detailed Recall & Accuracy Breakdown\n",
            "\n",
            "Breakdown across every individual family in the frozen test set:\n",
            "- `ngioweb` (Wordlist DGA, n=2,000)\n",
            "- `pizd` (Wordlist DGA, n=2,000)\n",
            "- `matsnu` (Hard Wordlist DGA, n=2,000)\n",
            "- `tinba` (Arithmetic / Non-Wordlist DGA, n=2,000)\n",
            "- `murofet` (Hash / Non-Wordlist DGA, n=2,000)\n",
            "- `legit` (Alexa / Tranco Benign, n=9,998)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "with open(METRIC_DIR / \"complete_evaluation_metrics.json\", \"r\") as f:\n",
            "    metrics_json = json.load(f)\n",
            "\n",
            "fam_df = pd.DataFrame(metrics_json[\"per_family_breakdown\"]).T\n",
            "# Reorder columns logically\n",
            "cols = [\"ngioweb\", \"pizd\", \"matsnu\", \"tinba\", \"murofet\", \"legit\"]\n",
            "fam_df = fam_df[[c for c in cols if c in fam_df.columns]]\n",
            "\n",
            "display_fam = fam_df.copy()\n",
            "for c in display_fam.columns:\n",
            "    display_fam[c] = display_fam[c].apply(lambda x: f\"{x*100:.2f}%\")\n",
            "\n",
            "display(HTML(\"<h3>Table 2: Per-Family Recall (DGA) and Specificity (Legit)</h3>\"))\n",
            "display(display_fam)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Comprehensive Binary Classification Metrics\n",
            "\n",
            "Includes Balanced Accuracy, Precision, Recall, Specificity, False Positive Rate (FPR), Matthews Correlation Coefficient (MCC), and raw Confusion Matrix counts (TP, FP, TN, FN)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "detailed_df = pd.read_csv(TABLE_DIR / \"final_test_detailed_metrics.csv\")\n",
            "float_cols = [\"Accuracy\", \"Balanced Acc\", \"Precision\", \"Recall\", \"Specificity\", \"FPR\", \"F1 (Binary)\", \"F1 (Macro)\", \"MCC\", \"ROC-AUC\", \"PR-AUC (AP)\"]\n",
            "display_detailed = detailed_df.copy()\n",
            "for c in float_cols:\n",
            "    display_detailed[c] = display_detailed[c].apply(lambda x: f\"{x:.4f}\")\n",
            "\n",
            "display(HTML(\"<h3>Table 3: Comprehensive Diagnostic & Classification Metrics</h3>\"))\n",
            "display(display_detailed)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Publication Visualizations (Figures 1 to 8)\n",
            "\n",
            "### Figure 1: Receiver Operating Characteristic (ROC) Curves\n",
            "Threshold-independent discrimination across all models. ModernBERT-Base achieves **0.833 AUC**, followed by Ensemble (**0.808**), Dual-Branch Coherent (**0.767**), and Dual-Branch v3 (**0.744**), all substantially outperforming Random Forest (**0.677**) and XGBoost (**0.658**)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / \"fig1_roc_curves.png\")))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Figure 2: Precision-Recall (PR) Curves\n",
            "Precision-Recall trade-off and Average Precision (PR-AUC). High PR-AUC reflects the ability to detect malicious domains while minimizing false alarms in operational networks."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / \"fig2_pr_curves.png\")))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Figure 3: 7-Panel Confusion Matrix Grid\n",
            "Complete breakdown of True Positives, False Positives, True Negatives, and False Negatives for all 7 architectures at standard threshold $\\tau=0.50$."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / \"fig3_confusion_matrices_grid.png\")))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Figure 4: Per-Family Recall & Specificity Bar Chart\n",
            "Head-to-head grouped bar chart across all 6 classes. Note the decisive dominance of all deep models on non-wordlist families (`tinba`, `murofet` at ~100%), the significant uplift of Dual-Branch Coherent on `matsnu` (**17.60%** vs ModernBERT's **14.15%**), and ModernBERT's superior benign specificity (**95.56%**)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / \"fig4_per_family_recall_barchart.png\")))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Figure 5: Benchmark Anchor Comparison\n",
            "Direct comparison against the published literature benchmarks:\n",
            "- **Floor:** Drichel et al. (RAID 2022) baseline (19.4% F1)\n",
            "- **Ceiling:** Leyva La O et al. (IEEE 2026) zero-shot wordlist ceiling (80.9% F1)\n",
            "- All deep learning models clear the Drichel floor by up to 2.3×."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / \"fig5_benchmark_anchors_comparison.png\")))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Figure 6: Threshold Decision Boundary Trade-Off Curves\n",
            "Plots Wordlist Detection Recall vs Benign Specificity as a function of decision threshold $\\tau \\in [0.01, 0.99]$. Demonstrates how security operators can calibrate the threshold according to deployment cost functions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / \"fig6_threshold_tradeoff_curves.png\")))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Figure 7: Error Composition Donut Charts\n",
            "Visual breakdown of total test errors into False Positives (benign flagged as DGA) vs False Negatives (DGA missed). ModernBERT has only 444 False Positives (10.2% of errors), whereas Dual-Branch models have ~1,377 False Positives (25.0% of errors)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / \"fig7_error_breakdown_donuts.png\")))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Figure 8: 6-Axis Model Profiling Radar Chart\n",
            "Multi-criteria comparison covering F1-Score, ROC-AUC, PR-AUC, Wordlist Macro Recall, Non-Wordlist Macro Recall, and Benign Specificity."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / \"fig8_radar_model_profile.png\")))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. LaTeX Code for Academic Paper / Project Report\n",
            "\n",
            "The code below prints the ready-to-paste LaTeX tables for your IEEE/ACM paper submission."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"% ==========================================================================\")\n",
            "print(\"% TABLE I: CSE427 FINAL TEST-UNSEEN BENCHMARK COMPARISON\")\n",
            "print(\"% ==========================================================================\")\n",
            "with open(TABLE_DIR / \"final_test_benchmark_table.tex\", \"r\") as f:\n",
            "    print(f.read())\n",
            "\n",
            "print(\"\\n% ==========================================================================\")\n",
            "print(\"% TABLE II: DETAILED CLASSIFICATION METRICS\")\n",
            "print(\"% ==========================================================================\")\n",
            "with open(TABLE_DIR / \"final_test_detailed_metrics.tex\", \"r\") as f:\n",
            "    print(f.read())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. How to Re-Run Full Live Inference Locally\n",
            "\n",
            "To recompute all raw predictions and re-generate all figures from scratch using the local GPU, execute the master script:\n",
            "```bash\n",
            "python scripts/run_final_master_evaluation.py\n",
            "```"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully wrote {len(cells)} cells to {nb_path}")
