# Findings — Evidence Index

Every headline claim in the README, traced to its numbers and artifacts. All test-set figures come from a single frozen evaluation pass over `test_unseen.csv` (19,998 domains), committed in [`reports/metrics/complete_evaluation_metrics.json`](../reports/metrics/complete_evaluation_metrics.json) and reproduced by `scripts/verify_results.py`.

---

## F1. Classical lexical features are obsolete for wordlist DGAs

**Claim:** Handcrafted lexical statistics (Shannon entropy, vowel/consonant ratios, n-gram frequencies) fail on unseen wordlist families.

**Numbers (frozen test set):**

| Model | Wordlist macro-recall | `matsnu` recall | Non-wordlist macro-recall |
|---|---:|---:|---:|
| Random Forest | 11.67% | 9.25% (185/2,000) | 93.45% |
| XGBoost | 6.93% | 0.95% (19/2,000) | 94.03% |

The same feature sets perform well on pseudo-random families (93–94%), confirming the failure is specific to the wordlist class — lexical statistics separate entropy regimes, and wordlist DGAs sit *below* benign entropy.

**Evidence:** `benchmark_comparison` and `detailed_metrics` in the metrics JSON; feature-importance analyses in [`training/figures/`](../training/figures) (`rf_feature_permutation.png`, `xgb_feature_gain.png`); F1–F2 of the threshold analysis in `fig6_threshold_tradeoff_curves.png`.

---

## F2. The 6D coherence vector is a genuine ablation gain, not added capacity

**Claim:** Injecting the explicit semantic coherence vector into an identical dual-branch backbone improves zero-day wordlist detection while shrinking the model.

**Numbers:**

| | Dual-Branch v3 | Dual-Branch Coherent | Δ |
|---|---:|---:|---:|
| Parameters | 4.10M | 0.73M | **−5.6×** |
| Wordlist macro-recall | 25.57% | 31.03% | **+5.46 pts (+21.4% relative)** |
| `matsnu` recall | 15.70% | 17.60% | +1.90 pts |
| `ngioweb` recall | 49.85% | 57.10% | +7.25 pts |
| Binary F1 | 0.6516 | 0.6799 | +0.028 |
| ROC-AUC | 0.7440 | 0.7671 | +0.023 |

Because the gain comes from a *smaller* network, it cannot be attributed to capacity. The improvement spans families (`ngioweb` +7.25, `matsnu` +1.90, `pizd` +7.25 points) rather than a single outlier.

**Evidence:** master table + `detailed_metrics` in the JSON; architecture comparison in `reports/figures/model_architectures_comparison.png`; the hypothesis-verification cell inside the executed [`training/Dual-Branch with Explicit Semantic Coherence.ipynb`](../training/Dual-Branch%20with%20Explicit%20Semantic%20Coherence.ipynb) (e.g., `fishing-shop` min-cos 0.22 vs `sugar-claim-mouse` −0.02, `fieldmarry` 0.01).

---

## F3. The `matsnu` inversion — 0.73M beats 149M ModernBERT on the hardest zero-day family

**Claim:** On unseparated dictionary concatenations, the proposed 0.73M Dual-Branch Coherent outperforms the fine-tuned 149M ModernBERT under a fully blind protocol.

**Numbers:**

| Model | `matsnu` recall | Detections (of 2,000) |
|---|---:|---:|
| Dual-Branch Coherent (0.73M) | **17.60%** | **352** |
| ModernBERT-Base (149M) | 14.15% | 283 |
| Relative gain | | **+24.4%** (69 additional detections) |
| Parameter ratio | | **204×** fewer |

**Mechanism:** `matsnu` concatenates dictionary words with no delimiter (`sugarmouse`). ModernBERT's BPE tokenizer segments it into frequent, grammatical subwords — self-attention scores the string as fluent brand language. `wordninja` recovers the same headwords, and near-zero pairwise FastText cosine similarity registers semantic disharmony. On hyphen-delimited families (`ngioweb`: 68.25% vs 57.10%; `pizd`: 22.65% vs 18.40%) the ranking reverses in ModernBERT's favor — the inversion is specific to the unseparated case.

**Evidence:** `per_family_breakdown` in the JSON; `fig4_per_family_recall_barchart.png`; Table IV in `reports/tables/final_test_detailed_metrics.csv`.

---

## F4. ModernBERT wins overall — stated as part of the result

**Claim:** The transformer dominates aggregate and specificity metrics.

**Numbers:** Best F1 (0.7372 vs 0.6799), ROC-AUC (0.8330 vs 0.7671), PR-AUC (0.8787), benign specificity 95.56% (444 FP / 9,998) vs 86.23% (1,377 FP), MCC 0.6025, accuracy 78.26%.

**Interpretation:** web-scale pretraining gives ModernBERT a far better model of legitimate naming conventions — real brand blends (`techradar`, `ticketmaster`) that a 0.73M network flags as anomalous. In an SOC where a false positive pages a human, this matters. The deployment trade-off is explicit: **ModernBERT = best aggregate operating point; Coherent = best zero-day wordlist recall per parameter; Ensemble 2 = the balanced compromise** (100.00% non-wordlist recall, 94.88% specificity, 34.25% WL macro, 512 FP).

**Evidence:** `detailed_metrics` in the JSON; `fig3_confusion_matrices_grid.png`, `fig7_error_breakdown_donuts.png`, `fig8_radar_model_profile.png`.

---

## F5. Efficiency per parameter

**Claim:** Wordlist recall per million parameters differs by ~180× between the proposed model and ModernBERT.

**Numbers:**

| Model | Params | WL macro-recall | Recall / M-Param |
|---|---:|---:|---:|
| Dual-Branch Coherent | 0.73M | 31.03% | 42.51 |
| Dual-Branch v3 | 4.10M | 25.57% | 6.24 |
| ModernBERT-Base | 149.0M | 35.02% | 0.24 |

~3 MB vs ~570 MB of weights; the BiLSTM stack is CPU-viable for resolver-side inference (a design property — no latency benchmark is claimed; ModernBERT's throughput profile requires GPU batching).

**Evidence:** derived from `benchmark_comparison`; parameter counts from the executed training notebooks.

---

## F6. Checkpointing on held-out wordlist recall transfers to unseen families

**Claim:** Selecting checkpoints on `val_wordlist_macro_recall` (over 3 validation wordlist families the models never train on) selects transferable structure, not validation memorization.

**Numbers (validation → frozen test):**

| Model | Val WL macro | Test WL macro | Δ |
|---|---:|---:|---:|
| Random Forest | 21.77% | 11.67% | ↓10.10 |
| XGBoost | 16.17% | 6.93% | ↓9.24 |
| Dual-Branch v3 | 24.80% | 25.57% | ↑0.77 |
| Dual-Branch Coherent | 28.45% | 31.03% | ↑2.58 |
| ModernBERT-Base | 26.05% | 35.02% | ↑8.97 |

Classical models — which fit lexical statistics rather than semantics — degrade sharply on truly unseen families; both semantic models improve. A model that memorized validation-family vocabulary would be expected to regress.

**Evidence:** validation metrics in each executed training notebook's final cells; test metrics in the JSON; Table V in `reports/tables/`.

---

## F7. Published wordlist-DGA scores are inflated by wordlist-in-training evaluation

**Claim:** High published "zero-shot" numbers are measured under conditions that leak the malware's dictionary into training; under a fully blind protocol the same architecture class collapses.

**Numbers:** Leyva La O et al. (2026) report 80.9% F1 (ModernBERT) / 84.6% (DomBertUrl) while training on eight wordlist families including `matsnu`, `suppobox`, `manuelita`, and holding out `bigviktor`, `ngioweb`, `pizd`. In this study's fully blind benchmark, ModernBERT fine-tuned on the *same underlying corpus* achieves 45.45% zero-shot recall on `ngioweb`+`pizd` and 14.15% on blind `matsnu`. The gap between 80.9% and 45.45% on comparable families is the size of the leakage effect for this architecture class.

**Caveat:** F1 vs recall and family-mix differences make this an approximation, not an exact comparison (see Limitations in the README). The qualitative point — dictionary-in-training changes the problem — is robust.

**Evidence:** README §2 and §5 anchors; [`docs/LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) §4–5 (verified source figures).
