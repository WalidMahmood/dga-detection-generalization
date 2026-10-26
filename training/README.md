# Training — Baselines, Architectures & Final Benchmark

Each notebook trains on the 32-family train split, validates on the 6 held-out val-unseen families, and never touches `test_unseen.csv` until the final evaluation notebook. All notebooks are committed **executed** (outputs intact); see [`docs/NOTEBOOK_GUIDE.md`](../docs/NOTEBOOK_GUIDE.md) for the full map, environments, and engineering log.

## Notebooks

| # | Notebook | Model | Hardware | Artifacts |
|---|----------|-------|----------|-----------|
| 01 | `01_rf_baseline.ipynb` | Random Forest (100 trees, class-weighted, 11 lexical features) | Local CPU | `weights/rf_best.pkl`, `results/baselines/rf_metrics.json`, `figures/rf_*.png` |
| 02 | `02_xgboost_baseline.ipynb` | XGBoost (400 trees, depth 8, lr 0.05, `hist` on GPU) | Local RTX 3050 | `weights/xgb_best.pkl`, `results/baselines/xgb_metrics.json`, `figures/xgb_*.png` |
| 03 | `Dual-Branch PERFECT.ipynb` (executed) + `03_dual_branch_kaggle_fp32_perfect.ipynb` (code-only, earlier revision) | Dual-Branch v3 (4.10M): char BiLSTM + FastText word-sequence BiLSTM, gated fusion | Kaggle T4 16 GB, FP32, effective batch 128 | `weights/dual_branch_best.pt` |
| 04 | `Dual-Branch with Explicit Semantic Coherence.ipynb` (executed) + `05_dual_branch_coherent.ipynb` (code-only) | **Dual-Branch Coherent (0.73M)**: same branches + explicit 6D FastText pairwise-cosine coherence vector | Kaggle T4 16 GB | `weights/dual_branch_coherent_best.pt` |
| 05 | `MODERNBERT-FINETUNED.ipynb` (executed) + `04_modernbert_finetune.ipynb` (code-only) | ModernBERT-Base (149M) fine-tune, 4 epochs, AdamW lr 3e-5, FP16 | Kaggle T4 16 GB | `weights/modernbertweights/` (HF format) |
| 06 | `06_final_test_evaluation.ipynb` (mirror: `FINAL-EVALUATION.ipynb`) | Frozen test-unseen benchmark of all 7 systems; generates `reports/` figures, tables, metrics | Kaggle T4 / artifacts | `reports/{figures,tables,metrics}/` |

## Shared protocol

- **Checkpoint criterion:** macro recall on the val-unseen *wordlist* families (`suppobox`, `bigviktor`, `manuelita`), with a non-wordlist recall guardrail floor — rationale in [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md) §5.
- **Sampling:** weighted sampler targets ~1:1 wordlist:non-wordlist per batch.
- **Determinism:** `SEED=42` across Python/NumPy/PyTorch; char-dropout 5% (train only); label smoothing 0.05 in the deep models.
- **FastText:** `cc.en.300.bin`, frozen — never fine-tuned.
- **Test discipline:** `test_unseen.csv` is loaded exactly once, in notebook 06, after every training decision is frozen.

## What the baseline notebooks print

For **both train and val** (so overfitting is visible): per-class `classification_report`, accuracy, F1, ROC-AUC, PR-AUC, confusion matrices (`*_confusion_{train,val}.png`), ROC + PR curves (`*_curves_{train,val}.png`), per-family recall bars on val (wordlist vs non-wordlist color-coded), correlation heatmap, impurity/gain + permutation + SHAP importances (`*_feature_*.png`), and `results/baselines/*_feature_importance.csv`.

## Reproducing

1. Mount the dataset (see [`docs/DATASETS.md`](../docs/DATASETS.md)) and place `cc.en.300.bin` on the path.
2. Run notebooks in order 01 → 02 → 03 → 04 → 05 → 06 on a ≥16 GB GPU (deep models ran on Kaggle T4).
3. Or regenerate all figures/tables/metrics from checkpoints with `scripts/run_final_master_evaluation.py`, and verify the published numbers with `scripts/verify_results.py`.

Model weights are reproducible from the notebooks and are not committed to the repository.
