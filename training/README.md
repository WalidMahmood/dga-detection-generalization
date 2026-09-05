# Training — Baselines & Models

Separate notebooks per model, each trains on `train` (32 families) and validates on `val-unseen` (6 new families) only. `test-unseen` is never touched until `06_Final_Test`.

## Notebooks

| # | Notebook | Model | Hardware | Time |
|---|----------|-------|----------|------|
| 01 | `01_rf_baseline.ipynb` | Random Forest (300 trees, 11 lexical features) | CPU | ~10 min |
| 02 | `02_xgboost_baseline.ipynb` | XGBoost (400 trees, depth 8, lr 0.05) — `gpu_hist` on RTX 3050 if `USE_GPU=True` | RTX 3050 GPU | ~20 min |
| 03 | `03_char_bilstm.ipynb` | Char Bi-LSTM (train-only vocab 38) | RTX 3050 | ~2h |
| 04 | `04_fasttext_only.ipynb` | Frozen FastText `.bin` + wordninja | RTX 3050 | ~1.5h |
| 05 | `05_dual_branch.ipynb` | Dual (Bi-LSTM + FastText concat) | RTX 3050 | ~3h |
| 06 | `06 ModernBERT` | ModernBERT-base fine-tune on **this** split | Kaggle T4/P100 | ~5h |

Only 01 & 02 are in this folder now (small baselines). 03-06 will be added next.

## What each notebook prints (everything)

For **both train and val** (so you see overfit):
- `classification_report` (precision/recall/F1 per class), `accuracy`, `F1`, `ROC-AUC`, `PR-AUC`
- Confusion matrix (`confusion_train/val.png`)
- ROC + PR curves side-by-side (`curves_train/val.png`)
- Per-family recall bar on val (blue=wordlist, orange=non-wordlist, `per_family_recall_val.png`)
- Feature part (RF/XGBoost only): correlation heatmap (`rf_correlation.png`), impurity + permutation + SHAP (`rf_feature_*.png`, `xgb_feature_*.png`), CSV `results/baselines/*_feature_importance.csv`
- Wordlist macro-recall = checkpoint criterion (mean of `suppobox/bigviktor/manuelita`)

## Outputs

- Weights: `training/weights/rf_best.pkl`, `xgb_best.pkl` (also `results/baselines/`)
- Metrics: `results/baselines/rf_metrics.json`, `xgb_metrics.json`
- Figures: `training/figures/*.png` (300 DPI, ready for paper)

## How to run on RTX 3050

```bash
# in VS Code / Jupyter Lab
open training/01_rf_baseline.ipynb → Run All
open training/02_xgboost_baseline.ipynb → set USE_GPU=True → Run All
```

No test data is used. Final one-shot test will be `training/06_Final_Test_Unseen.ipynb` loading all `*_best.pkl`.

## Legacy

`scripts/train_rf_xgboost.py` is the same logic as the notebooks as a `.py` script — kept for reference, but notebooks are the primary deliverable.
