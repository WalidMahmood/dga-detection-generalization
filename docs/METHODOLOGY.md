# Methodology — Protocol Design & Rationale

This document records *why* the evaluation is designed the way it is. For the numbers, see the README; for data provenance, see [`DATASETS.md`](DATASETS.md); for the literature grounding, see [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md).

---

## 1. Design goals

1. **Zero leakage** — no test family may influence training, architecture choice, or checkpoint selection in any form.
2. **Blind final evaluation** — the test split is evaluated exactly once, after every decision is frozen.
3. **Wordlist-focused** — the protocol is built to measure generalization to *unseen wordlist families specifically*, the case the literature handles worst (see the review, §2–5).
4. **Shortcut-proof** — the design must not allow a model to learn "word-like structure = benign", since benign domains are also word-like.

## 2. The Leave-Families-Out split

| Split | Families | Domains | Role |
|---|---|---|---|
| Train | 32 (26 non-WL + 6 WL) | 1,259,799 | Gradient updates |
| Val-Unseen | 6 (3 WL: suppobox, bigviktor, manuelita; 3 non-WL: kraken, ranbyus, zeus-newgoz) | 23,998 | Epoch-level checkpoint selection |
| Test-Unseen | 5 (3 WL: ngioweb, pizd, matsnu; 2 non-WL: tinba, murofet) | 19,998 | One-shot frozen benchmark |

Families are version-grouped before splitting (`gozi_gpl/luther/nasa/rfc4343 → gozi`, `vawtrak_v1/2/3 → vawtrak`, etc.) so no variant of one family straddles splits. Benign traffic is 1:1 matched per split.

### Why wordlist families stay in training (6 of 9 training wordlist families)

An earlier design held out *every* wordlist family. That was rejected: with no wordlist exposure at all, a model can win by learning the shortcut "word-like structure = benign" — because benign domains are word-like too. Keeping `gozi`, `rovnix`, `nymaim`, `banjori`, `charbot`, `deception` in training forces the model to learn the actual signal that distinguishes *malicious* dictionary concatenations from *legitimate* ones. The transfer question is then measured on families whose dictionaries the model has never seen.

### Why `pizd` is in test-unseen

`pizd` is one of only three families the strongest published baseline (Leyva La O et al., 2026) genuinely held out. Leaving it in training would have burned the single clean apples-to-apples comparison point against the 80.9% published ceiling. `ngioweb` was already test-only for the same reason.

### Why `manuelita` is in validation, not test

`manuelita` is a known-difficult wordlist family. Holding it out until the single final test pass means discovering a blind spot after there is no budget left to respond to it. Evaluating it at validation time gives an honest, actionable diagnostic during training — at the acknowledged cost of not being able to report a clean one-shot blind score for it (recorded in the README limitations).

### Why `matsnu` is the headline hard case

`matsnu` concatenates dictionary words *without* delimiters (`sugarmouse`), which is strictly harder than hyphenated forms (`travel-orange-river`) for subword-based models, and it was part of the ceiling study's *training* set — so any score on it is under a condition that study never attempted.

## 3. Preprocessing (and why it is not optional)

All domains are reduced to pure SLDs via `tldextract`, lowercased, exact-FQDN deduplicated, non-ASCII dropped, length bounded to 3–64. The EDA suite (`eda/02_preprocessing.ipynb`, `03_eda_after_and_comparison.ipynb`) quantifies the effect rather than assuming it:

- **Entropy:** mean entropy drops 3.30 → 2.81 (benign) and 3.52 → 3.16 (DGA) after TLD removal — a non-trivial share of raw-string entropy comes from the fixed TLD suffix, not the generated label. Entropy measured on raw FQDNs systematically overstates DGA randomness.
- **Artifact ratios:** digit/hyphen ratios collapse to a spike at zero after processing — raw-string statistics mostly measure formatting artifacts.
- **Unit test:** dot frequency must fall from 6.66% to *exactly* 0.0000% — a cheap executable check that the SLD isolation is correct.

Both sequence branches therefore operate on processed SLDs: otherwise the character branch partially learns TLD popularity, and the word segmenter would try to segment TLD suffixes as dictionary content.

The character vocabulary (38 characters + 4 special tokens, 42 total) is built **from the training split only**; measured OOV rate is 0 on val and test.

## 4. Class-imbalance handling

Training contains 26 non-wordlist families (~510k domains) vs 6 wordlist families (~120k) — an ~80/20 volume split even with identical per-family caps. Mitigation is at training time, not by changing the caps:

- A weighted random sampler applies ~6:1 oversampling to wordlist domains, enforcing approximately 1:1 wordlist:non-wordlist per mini-batch.
- All deep models checkpoint on **wordlist macro-recall over the 3 validation wordlist families**, so optimization pressure targets the research question rather than the easy majority.

## 5. Checkpoint criterion

- **Primary:** macro recall on `suppobox`, `bigviktor`, `manuelita` (val-unseen wordlist).
- **Guardrail:** non-wordlist val recall must stay above a floor (0.60 in the dual-branch runs) for a checkpoint to be eligible — prevents a model that wins wordlist recall by collapsing on algorithmic families.
- Validation families are used repeatedly for checkpoint selection (that is what validation is for); test families are never inspected before the single final pass.

## 6. Training environments & determinism

| Stage | Hardware | Notes |
|---|---|---|
| RF / XGBoost baselines | Local RTX 3050 / CPU | scikit-learn / XGBoost CPU-GPU |
| Dual-Branch v3, Coherent | Kaggle, Tesla T4 16 GB | FP32, effective batch 128 via grad accumulation |
| ModernBERT fine-tune | Kaggle, Tesla T4 16 GB | FP16, 4 epochs, ~39.4k steps, ~16 min/epoch |
| Final evaluation | Kaggle T4 / local | Single pass over frozen `test_unseen.csv` |

`SEED=42` seeds Python, NumPy, and PyTorch in every notebook. char-dropout (5%, train only) provides sequence-level augmentation; label smoothing (0.05) is used in the deep models.

## 7. Evaluation protocol

- Frozen `test_unseen.csv`: 5 unseen families × 2,000 + 9,998 benign; loaded once, predictions computed once.
- Metrics: accuracy, balanced accuracy, precision, recall, specificity, FPR, binary F1, macro-F1, MCC, ROC-AUC, PR-AUC (AP), full confusion counts — plus per-family recall for the wordlist and non-wordlist groups.
- Default threshold τ = 0.50 everywhere; the full τ ∈ [0.01, 0.99] trade-off is reported in `fig6_threshold_tradeoff_curves.png`.

## 8. Threats to validity (summary)

The full list lives in the [README limitations section](../README.md#9-limitations--threats-to-validity). The two most consequential: only 3 blind wordlist test families (case studies, not an average), and anchors from the literature use different metrics and family mixes — cross-study numbers are contextual, the internally consistent comparisons are the ones in this repository.
