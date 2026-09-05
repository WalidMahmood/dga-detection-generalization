# EDA — Professional Analysis Notebooks

Three publication-ready notebooks, executed with outputs and 22 figures @300 DPI.

## Notebooks

| # | Notebook | Purpose | Figures |
|---|----------|---------|---------|
| 01 | `01_eda_raw.ipynb` | Raw FQDN analysis **before** any cleaning — class balance, family distribution (80/20 imbalance), FQDN length, TLDs, lexical feats (digit/hyphen/vowel/entropy), char & bigram freqs, leakage checks | 9 |
| 02 | `02_preprocessing.ipynb` | Step-by-step pipeline demos on **10 real domains per step**: lowercase → TLD-strip (`tldextract` + private suffixes, Δ -3.1 chars) → exact-FQDN dedup (0 cross-split) → non-ASCII → length filter (99 rows) → train-only vocab (38 chars) → `wordninja` DP segmentation preview → benign caveat | 5 |
| 03 | `03_eda_after_and_comparison.ipynb` | **After + Before/After** comparison — counts retention (>99.99%), length & entropy shifts, char/TLD artefact removal, lexical stability, vocab OOV=0, per-family retention heatmap, main overlay | 9 |

All notebooks are **executed** (outputs embedded). Source is fully reproducible; figures are also saved as PNGs.

## Figures

`eda/figures/` — 22 PNGs, 300 DPI, ready for LaTeX `\includegraphics`:

- `01_class_balance_raw.png`, `02_family_dist_raw.png`, `03_fqdn_length_raw.png`, `04_tld_dist_raw.png`, `05_lexical_features_raw.png`, `06_entropy_per_family_raw.png`, `07_char_freq_raw.png`, `08_char_per_class_raw.png`, `09_bigram_raw.png`
- `preproc_len_reduction.png`, `preproc_length_cutoffs.png`, `preproc_char_vocab.png`, `preproc_benign_gap.png`
- `before_after_counts.png`, `before_after_family_retention.png`, `before_after_length.png`, `before_after_entropy.png`, `before_after_entropy_scatter.png`, `before_after_char.png`, `before_after_lexical.png`, `before_after_tld_reduction_verify.png`, `before_after_main_length_overlay.png`

Plus CSVs in `data/reports/` for tables: `length_stats_raw_by_family.csv`, `lexical_means_raw_by_family.csv`, `tld_counts_raw.csv`, `before_after_summary.csv`.

## Style

- `seaborn` whitegrid, 150 DPI screen / 300 DPI save, 11pt, consistent palette (`benign #2ecc71`, `dga #e74c3c`, `wordlist #3498db`)
- Every plot saved via `plt.savefig(..., bbox_inches="tight")`
- Tables include both `mean±std` and raw counts

## How to Run

```bash
# From G:\dga_killer
jupyter lab eda/
# or headless:
python -m nbconvert --to notebook --execute eda/01_eda_raw.ipynb --output 01_eda_raw_executed.ipynb --allow-errors
```

Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `tldextract`, `wordninja`, `scikit-learn` (for bigrams), `tabulate` (for markdown tables, optional).

## Key Findings (for your paper)

1. **1:1 balance holds** — Train 1.26M (19% wordlist), Val 24k, Test 20k.
2. **Lengths informative** — DGA median 16 vs benign 13; wordlist longer.
3. **TLD shortcut removed** — Raw TLDs dominated `.com`; stripping removes spurious signal.
4. **No leakage** — 0 exact-FQDN cross-split duplicates despite 100% pizd overlap.
5. **Lexical signals preserved** — Digit/hyphen/entropy gaps stable after TLD strip; wordlist families lower entropy (language-like) — Branch B signal intact.
6. **Preprocessing minimal** — Only 99 rows dropped (0.007%); vocab 38 chars, no OOV.

Next: `data/processed/*.csv` are the canonical inputs for baselines & dual-branch.
