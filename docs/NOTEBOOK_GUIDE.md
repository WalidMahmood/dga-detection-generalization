# Notebook Guide — Executed Runs, Environments & Artifacts

Every notebook in [`training/`](../training) is committed **with its outputs intact** — these are the actual runs behind every number in this repository. This page maps each file to its role, hardware, and artifacts.

---

## 1. The canonical runs

| Notebook | Role | Status | Hardware | Key result recorded in outputs |
|---|---|---|---|---|
| `01_rf_baseline.ipynb` | Random Forest on 11 lexical features | Executed (local) | CPU | val WL macro-recall 21.77% |
| `02_xgboost_baseline.ipynb` | XGBoost on the same features | Executed (local) | RTX 3050 (GPU `hist`) | val WL macro-recall 16.17% |
| `Dual-Branch PERFECT.ipynb` | **Dual-Branch v3 (4.10M)** — char BiLSTM + FastText word-seq BiLSTM, gated fusion | Executed | Kaggle T4 16 GB, FP32 | best val WL macro-recall 38.92% (epoch 1 of early-stopped run), val F1 0.6925 |
| `Dual-Branch with Explicit Semantic Coherence.ipynb` | **Dual-Branch Coherent (0.73M)** — adds the 6D coherence vector | Executed | Kaggle T4 16 GB | best val WL macro-recall 34.84%, non-WL ≈ 1.00, includes the coherence-hypothesis verification cell |
| `MODERNBERT-FINETUNED.ipynb` | ModernBERT-Base fine-tune (149M), 4 epochs on the full 1.26M train split | Executed | Kaggle T4 16 GB, FP16 | val F1 0.6493, val WL macro-recall 26.05%, threshold sweep |
| `06_final_test_evaluation.ipynb` | **Frozen test-unseen benchmark** of all 7 systems + publication figures + LaTeX tables | Executed | Kaggle T4 / artifacts | the master tables in the README |

Clean code-only companions (identical pipelines, no outputs): `03_dual_branch_kaggle_fp32_perfect.ipynb` (an earlier revision of the dual-branch lineage), `04_modernbert_finetune.ipynb` (byte-identical code to `MODERNBERT-FINETUNED.ipynb`), `05_dual_branch_coherent.ipynb` (differs from its executed twin only by a CSV-loading guard). `FINAL-EVALUATION.ipynb` is a byte-identical mirror of `06_final_test_evaluation.ipynb`, kept for direct viewing.

## 2. Datasets & inputs used by every notebook

- `data/processed/{train.csv, val_unseen.csv, test_unseen.csv}` + `char_vocab.json` — locally under `DGA_DATA_ROOT`, on Kaggle mounted from [`walidmahmood/dga-merged-umudga-rhena`](https://www.kaggle.com/datasets/walidmahmood/dga-merged-umudga-rhena) (with glob-based auto-discovery fallback).
- `cc.en.300.bin` — FastText 300D (FacebookAI via fastText.cc), frozen during training.
- `answerdotai/ModernBERT-base` at a pinned commit (recorded in `training/weights/modernbertweights/__huggingface_repos__.json`).

## 3. Determinism & shared training setup

- `SEED = 42` seeds `random`, `numpy`, `torch` in every notebook.
- Checkpoint criterion everywhere: **val-unseen wordlist macro-recall** (`suppobox`, `bigviktor`, `manuelita`) with a non-wordlist recall guardrail floor.
- Weighted sampling: ~6:1 wordlist:non-wordlist per batch (neutralizes the 80/20 volume split, see [METHODOLOGY.md](METHODOLOGY.md) §4).
- Char-level dropout 5% (train only); label smoothing 0.05 in the deep models.

## 4. Artifacts produced

| Artifact | Where |
|---|---|
| RF / XGBoost pickles | `training/weights/{rf_best.pkl, xgb_best.pkl}` (not redistributed) |
| Baseline metrics | `training/results/baselines/{rf_metrics.json, xgb_metrics.json}` |
| Baseline QA plots (confusion, curves, SHAP/permutation importance, per-family recall) | `training/figures/{rf_*,xgb_*}.png` (committed) |
| Dual-Branch checkpoints | `training/weights/{dual_branch_best.pt, dual_branch_coherent_best.pt}` |
| ModernBERT fine-tuned model (HF format) | `training/weights/modernbertweights/` |
| Publication figures + LaTeX/CSV tables + master metrics | `reports/{figures,tables,metrics}/` |

## 5. Engineering notes (from the executed logs)

Two debugging episodes are visible in the committed notebooks and are part of the record:

1. **The dead gate.** In an earlier dual-branch revision, the fusion gate was computed but *discarded* (`fused = cat([h_char, h_word])`), making "gated fusion" a plain concat. The fix (`g⊙h_char + (1−g)⊙h_word`) is what ships in the executed v3 notebook; gate-value diagnostics in that run show the gate actually differentiating (mean 0.195 overall vs 0.230 wordlist / 0.183 non-wordlist).
2. **The `kraken` false-negative cluster.** Persistent `kraken` misses were root-caused to a data artifact, not the model: kraken's raw domains are dynamic-DNS hostnames (`*.mooo.com`, `*.dynserv.com`, `*.yi.org`), which collapse to only ~499 unique SLDs after TLD stripping — a handful of frequent hosts dominate the family, and benign-looking dynamic-DNS hosts leak into the SLD. This motivated the enforced non-wordlist recall floor on checkpoints.

Also documented in the run logs: the fp16 → fp32 + gradient-accumulation switch (effective batch 128) and the train/val F1-gap monitor (target < 0.25) used to control overfitting on the 1.26M-domain corpus.

## 6. Reproducing the runs

See the README [Reproducibility section](../README.md#11-reproducibility): verify reported numbers with `scripts/verify_results.py` (no data or weights needed), or re-run the notebooks in order 01 → 02 → 03/v3 → Coherent → 04 → 06 with the dataset mounted and `cc.en.300.bin` available. Full re-evaluation from checkpoints: `scripts/run_final_master_evaluation.py`.
