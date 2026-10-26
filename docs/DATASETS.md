# Datasets — Provenance, Merge Rules & Audit

The benchmark corpus is 1,303,795 labeled domains (43 DGA families + benign), constructed from three public sources, merged at the family level, deduplicated at the domain-string level, and partitioned with the Leave-Families-Out protocol described in [`METHODOLOGY.md`](METHODOLOGY.md).

---

## 1. Sources

### UMUDGA — Zago, Gil Pérez & Martínez Pérez (2020)
- **What:** 50 malware DGA family folders + benign/legit class; 30M+ labeled domains.
- **Where:** Mendeley Data, DOI [10.17632/y8ph45msv8.1](https://doi.org/10.17632/y8ph45msv8.1); companion papers in *Data in Brief* 30:105400 and *Computers & Security* 92:101719.
- **Supplies:** all 26 non-wordlist training families plus 7 wordlist/wordlist-adjacent families (`gozi`, `rovnix`, `nymaim`, `pizd`, `banjori`, `matsnu`, `suppobox`).
- **Known property (stated in limitations):** domains are generated deterministically by executing each family's DGA with fixed seeds — realistic strings, not live sinkhole captures.

### Reynier/moe-wordlist-dga-models — Leyva La O, Catania & Gonzalez (2026)
- **What:** the dataset release accompanying *Expert Selection for Wordlist-Based DGA Detection* (IEEE Latin America Transactions 24(1)), on [Hugging Face](https://huggingface.co/datasets/Reynier/moe-wordlist-dga-models).
- **Supplies:** the 5 wordlist families absent from UMUDGA (`charbot`, `deception`, `manuelita`, `bigviktor`, `ngioweb`).
- **Note:** this repo's own documentation credits UMUDGA as one of its underlying sources — real family overlap exists, which is why deduplication operates on domain strings, not family names (§3).

### Benign — Alexa Top-1M & Tranco
- High-reputation benign domains, sampled to match DGA volume 1:1 per split.
- **Known property (stated in limitations):** top-sites benign traffic is structurally "easy mode" relative to adversarial negatives.

## 2. Family classification (wordlist vs non-wordlist)

Wordlist status follows the UMUDGA paper's own † markings (`gozi`, `matsnu`, `rovnix`, `suppobox` confirmed) plus its collision analysis (`nymaim`, `pizd` wordlist-adjacent), with `banjori` retained as a disputed/mixed case in training only. `charbot`, `deception`, `manuelita`, `bigviktor`, `ngioweb` are wordlist per Leyva La O et al. (2026).

## 3. Merge & dedup rules

1. **Family-level union** of the two DGA sources; multi-variant families collapse to one label before split assignment (`gozi_gpl/luther/nasa/rfc4343 → gozi`, `fobber_v1/v2 → fobber`, `suppobox_1/2/3 → suppobox`, `murofet_v1/2/3 → murofet`, `kraken_v1/v2 → kraken`, `ranbyus_v1/v2 → ranbyus`, `vawtrak_v1/2/3 → vawtrak`) — so no variant of a family can straddle splits.
2. **Domain-string deduplication** (exact FQDN, lowercased) — necessary because the sources overlap (e.g., `pizd` has ~100% string overlap between UMUDGA and the HF release). After preprocessing, cross-split exact matches were re-checked and totalled 509 removals (`pykspa_noise` 499, `suppobox` 1, `pizd` 9) — i.e., **zero cross-split contamination in the frozen splits**.
3. **Per-family caps:** train ≤ 20,000/family; val/test 2,000/family; benign 1:1 matched per split.
4. **Preprocessing:** `tldextract` SLD isolation → lowercase → FQDN dedup → non-ASCII drop → length filter 3–64. Retention 99.99%.

## 4. Split volumes (final, audited)

| Split | DGA | Benign | Total | Families |
|---|---:|---:|---:|---|
| Train | 629,899 | 629,900 | 1,259,799 | 32 (26 non-WL + 6 WL) |
| Val-Unseen | 11,999 | 11,999 | 23,998 | 6 held-out (3 WL + 3 non-WL) |
| Test-Unseen | 10,000 | 9,998 | 19,998 | 5 zero-day (3 WL + 2 non-WL) — frozen |

Supply checks and per-family volumes: [`DATA_READINESS_EVALUATION.md`](../DATA_READINESS_EVALUATION.md).

## 5. Quality audit (pre-training, automated)

Run by `evaluate_data_readiness.py`; full report in [`DATA_READINESS_EVALUATION.md`](../DATA_READINESS_EVALUATION.md):

- Class balance ≈ 1:1 in every split (ratio ≈ 1.0002).
- Character vocabulary: 42 tokens (4 special + 38 regular), built **train-only**; **0 OOV characters** in any split.
- 0 empty/non-ASCII SLDs; 99.99% preprocessing retention.
- Domain length (SLD): train mean 12.07 (σ 5.35), test mean 13.28 — comparable across splits.

## 6. Accessing / rebuilding

- **Merged corpus (as used):** [`kaggle.com/datasets/walidmahmood/dga-merged-umudga-rhena`](https://www.kaggle.com/datasets/walidmahmood/dga-merged-umudga-rhena) — `data/processed/{train.csv, val_unseen.csv, test_unseen.csv}` + `char_vocab.json`.
- **Rebuild from sources:** download UMUDGA (Mendeley DOI above) and the HF release, then run `build_dataset.py` from the repo root with the `DGA_DATA_ROOT` environment variable pointing at the folder containing both source folders. The script emits raw + processed splits, the split audit, and the train-only vocab. Verify with `evaluate_data_readiness.py`.

## 7. Licensing & attribution

This repository ships **code, splits metadata, and evaluation artifacts only** — not the datasets themselves. If you use the merged corpus, you must also cite the three sources above (UMUDGA per its DOI; the Leyva La O et al. 2026 paper for the HF release; Tranco/Alexa per their terms). Dataset licensing follows the original sources.
