# CSE427 — DGA Detection: Master Plan & Progress Tracker

> **Purpose of this file:** single source of truth. Update checkboxes as you go. If you stop
> for a day and come back, read this file top to bottom before touching code.
>
> **Companion document:** `Literature_Review_Methodology.docx` — the full literature review
> with every citation verified against primary sources, including the same architecture and
> split decisions as this file, kept in sync. If the two ever disagree, this file is the
> more current one for day-to-day decisions; port the fix into the lit review too.

---

## 0. Research Question (locked)

Does adding a semantic/word-level branch (FastText) to a character-level DGA classifier
(Bi-LSTM) improve generalization to **unseen wordlist-based DGA families**, under a
Leave-Families-Out protocol?

**10-day course window. Course is 4 months but you effectively have 10 days to execute.**

**What counts as success (be honest about this, don't let the writeup overclaim):**
- **Required claim:** dual-branch beats the char-only and FastText-only baselines on
  test-unseen wordlist families, and clears the Drichel (2022) 19.444% zero-day floor by a
  meaningful margin. This alone is a legitimate, defensible result.
- **Stretch claim (see Section 5 "Can we actually beat ModernBERT?"):** dual-branch
  matches or beats a ModernBERT-base baseline *trained and evaluated on your own exact
  split*. This is ambitious but grounded in real prior art, not fluff — see below. Don't
  promise this outright in the proposal/intro; let the numbers decide, and have a fallback
  framing ready (efficiency, sample-efficiency, or per-family wins) regardless of the
  aggregate outcome.

---

## ✅ RESOLVED — val had no hard-case visibility + benchmark mismatch (v2)

Two separate problems found, both fixed by the same rebalance:

**Problem 1:** matsnu and manuelita are two of the field's known-hardest wordlist families
(a separate LLM-based DGA-detection study found manuelita/matsnu/suppobox specifically
underperform, ~0.65 precision, because word-based domains are structurally harder to
classify). Both were sitting in test-unseen only — invisible during model selection,
discoverable only after the one-shot final test.

**Problem 2 (bigger):** the HF repo (`Reynier/moe-wordlist-dga-models`) is the actual
data/code release for Leyva La O et al. (2026), "Expert Selection for Wordlist-Based DGA
Detection," IEEE Latin America Transactions — the exact paper this plan cites as the
81–85% benchmark ceiling. In THAT paper's own protocol:
- **Trained/"known" families (8):** charbot, deception, gozi, manuelita, matsnu, nymaim,
  rovnix, suppobox
- **True zero-shot/"generalization" families (3):** bigviktor, ngioweb, pizd — the 80.9%
  (ModernBERT) / 84.6% (DomBertUrl) ceiling numbers are measured on THIS set only.

So matsnu and manuelita were never zero-shot in the source paper — comparing our blind
test-unseen result on them against that 81–85% ceiling would be apples-to-oranges.
Meanwhile `pizd` — one of their actual 3 zero-shot families — was sitting in our TRAIN
split, wasting the one clean comparison point we'd have for it.

**Decision:**
- Train wordlist: drop `pizd` (6 families: gozi, rovnix, nymaim, banjori, charbot, deception)
- Val-unseen wordlist: add `manuelita` (3 families: suppobox, bigviktor, manuelita)
- Test-unseen wordlist: swap `manuelita` → `ngioweb, pizd` (3 families: matsnu, ngioweb, pizd)

Result: 2 of 3 final test families (ngioweb, pizd) are the SAME families the 81–85%
ceiling was measured on — a fair comparison. matsnu is a bonus result, reported honestly
as a harder test than the source paper attempted (they trained on it; we don't). manuelita
gets evaluated during Phase 4 tuning instead of only at the very end.

Limitations-section sentence (copy into Section 8 / final writeup):

> "Of the three test-unseen wordlist families, ngioweb and pizd match the zero-shot
> generalization protocol used in Leyva La O et al. (2026), making our results on those two
> directly comparable to their reported 80.9–84.6% F1 ceiling. matsnu is reported as a
> supplementary result under a strictly harder condition than that paper's own evaluation,
> since matsnu was part of their training set rather than held out."

---

## 1. Family Pool (43 families total)

### Non-wordlist (31) — UMUDGA
alureon, bedep, ccleaner, chinad, corebot, cryptolocker, dircrypt, dyre, fobber (v1,v2),
kraken (v1,v2), locky, murofet (v1,v2,v3), necurs, padcrypt, proslikefan, pushdo, pykspa,
pykspa_noise, qadars, qakbot, ramdo, ramnit, ranbyus (v1,v2), shiotob, simda, sisron, symmi,
tempedreve, tinba, vawtrak (v1,v2,v3), zeus-newgoz

### Wordlist / wordlist-adjacent (7) — UMUDGA
gozi (gpl, luther, nasa, rfc4343), rovnix, nymaim, pizd, banjori, matsnu, suppobox (1,2,3)

### Wordlist (5) — Hugging Face (`Reynier/moe-wordlist-dga-models`)
charbot, deception, manuelita, bigviktor, ngioweb

> **⚠️ Implementation-safety rule, easy to get wrong:** families with version suffixes
> (fobber v1/v2; kraken v1/v2; murofet v1/v2/v3; ranbyus v1/v2; vawtrak v1/v2/v3) must have
> **ALL versions travel together in the same split, always.** When you build the
> family→split config dict in Phase 1, don't let a loader accidentally treat `fobber_v1`
> and `fobber_v2` as independent keys that could land in different splits — if v1 ends up
> in train and v2 in test, that's leakage (the model has effectively seen a near-duplicate
> of the "unseen" family). Group by base family name, not by file name, when building the
> mapping.

> **Note:** `pizd` appears in **both** UMUDGA and in HF's `test-generalization/`. The HF
> repo's own README credits UMUDGA as an underlying source — so this may not be the only
> overlap. Don't trust family-name matching alone; dedup by actual domain string in Phase 1
> for any family name that exists in both sources.

### Benign/legit
UMUDGA `legit` folder — balanced 1:1 against malicious samples in **every** split. See
Section 4's benign-source caveat before assuming this is a "fair" negative class.

---

## 2. Final Split (locked — v2, rebalanced for hard-case visibility + benchmark validity)

| Split | Non-wordlist | Wordlist | Purpose |
|---|---|---|---|
| **Train** (26 + 6 = 32 fam) | alureon, bedep, ccleaner, chinad, corebot, cryptolocker, dircrypt, dyre, fobber, locky, necurs, padcrypt, proslikefan, pushdo, pykspa, pykspa_noise, qadars, qakbot, ramdo, ramnit, shiotob, simda, sisron, symmi, tempedreve, vawtrak | gozi, rovnix, nymaim, banjori, charbot, deception | Gradient updates |
| **Val-unseen** (3 + 3 = 6 fam) | zeus-newgoz, ranbyus, kraken | suppobox, bigviktor, manuelita | Epoch-level checkpointing / early stopping / architecture calls. NEVER backprop. |
| **Test-unseen** (2 + 3 = 5 fam) | tinba, murofet | matsnu, ngioweb, pizd | Touched **once**, after everything is frozen. |

`ngioweb` and `pizd` are the test-unseen wordlist duo directly comparable to the Leyva La O
(2026) ceiling — `matsnu` is the bonus/harder supplementary result.

**Hard rule:** test-unseen is not to be looked at, profiled, tuned against, or visualized
until Phase 5. If you peek, the result is contaminated — start the eval section over.

---

## 3. Sample Caps

| Split | Cap/family | Note |
|---|---|---|
| Train | ≤20,000/family (or all if fewer) | stops huge UMUDGA families dominating |
| Val-unseen | 1,500–2,000/family | matches Leyva La O (2026) convention |
| Test-unseen | 1,500–2,000/family | same, for benchmark comparability |
| Benign | 1:1 matched per split | standard balanced practice |

Expected scale: ~400K–600K train domains (malicious) + matched benign.

⚠️ **Supply check needed:** in the source HF repo's own eval set, bigviktor and ngioweb had
only ~2,001 samples each (pizd had ~9,560). If your UMUDGA/HF pull has a similarly thin pool
for these two, you'll barely clear the 1,500–2,000 cap with no headroom. Confirm actual
available counts in Phase 1 before assuming the cap is achievable for every family.

⚠️ **Family-count vs. family-type imbalance (raised and worth taking seriously):** with
26 non-wordlist vs. 6 wordlist families in train, even at equal per-family caps you get up
to ~520K non-wordlist malicious domains vs. up to ~120K wordlist — roughly an 80/20 split
by volume. Per-family caps alone don't fix this; it's a family-*count* imbalance, not a
per-family sample imbalance. See Section 5's stratified-sampling fix — this is a training-
time fix, not a data-cap fix, because reducing non-wordlist families would defeat the
purpose of training a general DGA detector, and inflating the wordlist cap doesn't work
because most wordlist families don't have the supply for it (see caveat above).

---

## 4. Data Preprocessing (NEW — do this properly, it's easy to get subtly wrong)

1. **Lowercase everything.** Consistently, across benign and malicious, all splits.
2. **Operate on the registered/SLD label, strip the TLD.** This matches the standard
   convention in char-level DGA literature (Woodbridge et al. and follow-on work) — the
   TLD (`.com`/`.net`/etc.) rarely carries family-specific signal and mixing it in adds
   noise. Make this a documented, deliberate choice applied *identically* to benign and
   malicious across train/val/test — don't let it vary by source file format.
3. **Dedup by exact domain string — within a split AND across splits.** This is the
   single most important step given the UMUDGA/HF overlap risk flagged in Section 1. A
   domain string appearing in both train and test-unseen is leakage regardless of which
   "family" folder it came from.
4. **Handle non-ASCII / punycode.** Most DGA families are pure ASCII by construction;
   decide once whether to normalize (punycode-decode) or drop non-ASCII entries, and apply
   it uniformly. Don't let benign and malicious get different treatment here — that alone
   can create a spurious shortcut feature.
5. **Length sanity filter.** Drop obviously corrupt entries (empty labels, single-char
   labels, anything absurdly long) — don't be aggressive, some DGA families legitimately
   produce long labels.
6. **Branch A vocabulary (char-level):** build the character vocabulary from **TRAIN only**.
   Domain-name character sets are small (a–z, 0–9, hyphen) so this is unlikely to bite, but
   it's a one-line safeguard against val/test leaking into vocabulary construction.
7. **Branch B preprocessing (word segmentation) — this is the part worth doing carefully:**
   - Wordlist-family domains concatenate real dictionary words with **no delimiter**
     (`reindeerhorse.net`). A generic subword tokenizer (like ModernBERT's) approximates
     boundaries from whatever merge rules it learned on ordinary text, which rarely
     contains compound strings like this — it won't reliably recover the true
     `reindeer`+`horse` split.
   - Use a proper **dynamic-programming word segmenter**: Viterbi-style search over an
     English unigram frequency list, maximizing total log-probability of the split (the
     approach behind the `wordninja` package and Wolf Garbe's `WordSegmentationDP` — the
     latter explicitly lists "keyword extraction from URL addresses, domain names... written
     without spaces" as a use case). Either use `wordninja` directly (`pip install
     wordninja`) or replicate its DP approach with a frequency-annotated English wordlist.
     Don't use a naive "maximize dictionary-word count" segmenter — it's known to have a
     small-common-word bias problem (splits everything into "a", "the", etc.); the
     frequency-log-weighted DP approach avoids this.
   - Feed segmented tokens into **pretrained FastText vectors, kept FROZEN** during
     training (see Section 5 for why). Use the `.bin` format (not `.vec`) so FastText's
     subword n-gram fallback still gives a reasonable embedding for segmentation mistakes
     or rare tokens, instead of an OOV zero-vector.
8. **Benign source check.** UMUDGA's `legit` folder is presumably a top-domains-style list.
   This is a known "easy mode" critique in security-ML methodology (top-1M-style benign
   sets are structurally very different from malicious algorithmic domains, which can
   inflate apparent performance) — flag it as a documented limitation rather than silently
   assuming the benign set is realistic. Fixing it (e.g., supplementing with harder
   recently-registered-domain benign samples) is a stretch goal, not a 10-day requirement.

---

## 5. Model Architecture

- **Branch A — char-level: Bi-LSTM.** Not a from-scratch Transformer. A from-scratch
  Transformer with no large-scale pretraining would likely underperform *both* the BiLSTM
  (which has good inductive bias for sequential patterns on small data) *and* the
  ModernBERT baseline (which has 2 trillion tokens of pretraining behind it) — it's the
  worst of both worlds for a 10-day budget. Keep Branch A lightweight; that's part of the
  point of the paper (see below).
- **Branch B — semantic/word-level: frozen pretrained FastText embeddings** over
  DP-segmented domain tokens (Section 4, step 7). **Do not fine-tune the FastText
  embeddings.** This isn't a guess — it directly replicates the design in Koh & Rhodes
  (2018), "Inline Detection of Domain Generation Algorithms with Context-Sensitive Word
  Embeddings," IEEE Big Data 2018 (an earlier pass of this file mislabeled this as
  "Highnam et al." — corrected; verified directly against the paper's own results table).
  They used **frozen pretrained ELMo** (not FastText — FastText here is this project's own
  substitution, a lighter-weight non-contextual alternative better suited to short domain
  strings) combined with a simple classifier and reliably outperformed char-level CNN/LSTM
  approaches specifically on wordlist-based DGA families — confirmed exact figures: 89.5%
  detection @ 1:1,000 FPR after 30 Matsnu examples, 91.2% @ 1:10,000 FPR after 90 examples,
  an order of magnitude faster to train than the char-level baselines they compared
  against. They also stripped the TLD and used a DP/Zipfian word segmenter (`wordninja`)
  before embedding lookup — independent confirmation of two choices already in Section 4.
  Their own analysis found the classifier wasn't memorizing the DGA wordlists themselves, it
  was learning the *semantic coherence signature* of legitimate-word concatenation vs.
  random/non-semantic concatenation. That's the mechanism your Branch B is betting on, and
  it has real precedent, not just intuition.
- **Fusion:** start with concat + small MLP classification head — simplest thing that
  could work, given the timeline. Attention-fusion is a reasonable Phase 4 ablation if time
  allows, not a Phase 3 requirement.

### Baselines (for the comparison table) — **now four, not three**

- RF / XGBoost on hand-crafted lexical features (entropy, n-gram freq, length, digit
  ratio, vowel ratio, etc.)
- Char-only BiLSTM (no semantic branch)
- FastText-only (no char branch)
- **ModernBERT-base, fine-tuned and evaluated on YOUR OWN train/val-unseen/test-unseen
  split** — see below for why this one is not optional.

### Can we actually beat ModernBERT? (honest answer, not a pep talk)

Two separate things were tangled together in the original framing, and they need to be
pulled apart:

**1. You cannot validly compare against the paper's published 80.9%/84.6% numbers and call
it "beating ModernBERT."** Those numbers were produced on Leyva La O et al.'s own split
(their 8 known families, their 3 generalization families, their sample sizes). Your split
is different (32/6/5 families, different caps). A fair "we beat ModernBERT" claim requires
training ModernBERT-base yourself, on your exact train split, evaluated on your exact
val-unseen/test-unseen — that's why it's now baseline #4. This is the single most important
structural fix for making your ambition scientifically valid rather than a leaderboard
number pulled from a different experiment.

**2. Is the underlying bet — dual-branch can beat a 149M-parameter transformer pretrained
on 2 trillion tokens of text — realistic in 10 days?** Be honest with yourself here.
ModernBERT-base has ~149M parameters and was pretrained on roughly 2 trillion tokens of
English and code. That's a large amount of general language knowledge baked in before it
even sees a single domain name. Outright beating it *in aggregate* is a real stretch goal,
not a safe bet — don't write the intro as if it's guaranteed.

What IS well-grounded (per Koh & Rhodes (2018), directly on this exact problem, not a general
ML intuition) is that **full fine-tuning of a large pretrained transformer on a narrow,
low-data task can under-use or overwrite the very general lexical/semantic knowledge that
would help most** — full fine-tuning optimizes toward whatever shortcut minimizes loss on
your ~150K-ish training examples, and character-level statistical shortcuts (entropy,
n-gram frequency) are cheap, effective shortcuts a transformer can and often does fall back
on. A frozen semantic branch architecturally *cannot* take that shortcut — it's forced to
keep using real lexical-semantic structure. That's a genuine, citable mechanism, not fluff.

**So the defensible plan is:** build the dual-branch model on its real merits (frozen
FastText + proper DP segmentation, not just "add FastText and hope"), train ModernBERT
yourself as a fair baseline, and go into the writeup with **several possible honest wins
lined up**, not just one all-or-nothing claim:
- Aggregate F1 beats ModernBERT on your split (the headline win, if it happens)
- Per-family win specifically on segmentation-sensitive families (a targeted, explainable
  claim, defensible even if the aggregate doesn't beat ModernBERT)
- Sample-efficiency comparison (Koh & Rhodes's own framing: near-SOTA results with far
  fewer examples than char-only methods need) — you have the per-family data to check this
- Efficiency/parameter-count comparison (the source paper's own inference-time table shows
  wide latency spread across model classes — a smaller dual-branch model that's
  competitive at a fraction of the parameter count and inference cost is a legitimate,
  citable contribution on its own, independent of whether it wins on raw F1)

Report whichever of these the actual numbers support. Don't pre-commit to only the hardest
one.

---

## 6. Training / Eval Loop

```
for each epoch:
    train on TRAIN families (backprop)
      — use a stratified/weighted sampler, NOT plain random sampling from the pooled
        training set. Target roughly a 1:1 wordlist:non-wordlist ratio per batch
        (oversample wordlist families with replacement as needed — they have far less raw
        volume, ~120K ceiling vs ~520K). This directly addresses the family-count
        imbalance flagged in Section 3. Start at 1:1; treat the exact ratio as a Phase 4
        tunable, not a fixed constant.
    eval on VAL-UNSEEN families (forward only, no grad)
    log: train loss, val-unseen F1/recall per family (wordlist + non-wordlist separately)
    checkpoint criterion: PRIMARY = macro-F1 averaged across the 3 val-unseen WORDLIST
      families (suppobox, bigviktor, manuelita) — NOT blended with non-wordlist. This is
      the actual research question; don't let a checkpoint get selected because it's good
      at non-wordlist families while being mediocre at the ones that matter.
      GUARDRAIL = track non-wordlist val-unseen macro-F1 too. If it drops meaningfully
      below what the char-only baseline achieves on the same families, that's a red flag —
      the fusion may be sacrificing general DGA detection to chase wordlist gains. Don't
      ignore it, but don't let it override the primary criterion either.
    if primary criterion improved: save checkpoint

after training finishes:
    load best checkpoint
    evaluate ONCE on TEST-UNSEEN
    report per-family F1/recall/precision — final numbers, no further tuning after this
```

**Benchmark anchors:**
- Floor: Drichel et al. (2022), "Detecting Unknown DGAs without Context Information" —
  Regex-Error-Detection approach on leave-one-group-out macro-F1, **19.444% median** (this
  is the exact source of the "19-20%" figure, confirmed).
- Ceiling: Leyva La O et al. (2026) unknown wordlist families — 80.9% (ModernBERT) / 84.6%
  (DomBertUrl) F1, measured on bigviktor/ngioweb/pizd specifically (see RESOLVED section).
- Your target zone: clear the floor by a meaningful margin = legitimate required result.
  Matching/beating the ceiling on your own ModernBERT baseline = stretch goal, see Section 5.

---

## 7. Compute Budget (NEW)

- **Local (8GB VRAM):** fine for RF/XGBoost (CPU-bound anyway), char-only BiLSTM,
  FastText-only, and the dual-branch model itself — all lightweight relative to 8GB, domain
  names are short sequences so memory pressure is low regardless of branch.
- **Kaggle free tier:** confirmed specs — 30 GPU-hours/week quota, NVIDIA T4×2 (16GB each,
  ~32GB combined) or P100 (16GB) depending on which you select, **12-hour session cap**.
  Budget accordingly — with a 10-day window you effectively get ~1.5 weeks of quota
  (~45 hours total if you don't waste any), not unlimited compute.
- **ModernBERT-base (149M params) fine-tuning:** run this on Kaggle, not local — more
  headroom, and it's the most compute-hungry item in the plan. Should fit comfortably
  within a single 12-hour session even with conservative batch sizes, given domain names
  are short strings (this is a well-trodden fine-tuning regime, not an exotic one).
- **Session-cap discipline:** checkpoint frequently and support resume-from-checkpoint for
  anything that might run close to 12 hours — don't lose a long run to the session cap.
- **Parallelize, don't serialize:** kick off the ModernBERT baseline fine-tune on Kaggle as
  soon as Phase 1 data is ready, and let it run in the background while you build Phase 2's
  other baselines and start Phase 3 locally. It's the most time-pressure-sensitive item on
  a 10-day clock — don't leave it to the end of Phase 2.
- Confirm your Kaggle notebook has internet access enabled (needed to `pip install
  fasttext wordninja` and download pretrained FastText vectors) before you're mid-Phase-1.

---

## 8. Known Limitations (pre-written — include in final writeup, edit as needed)

- Not a full Leave-One-Family-Out across all 92+ real DGA families — targeted wordlist-
  generalization eval, not general-purpose DGA detection.
- Only 3 held-out wordlist test families — still small; treat per-family results as case
  studies, not statistically robust averages.
- Banjori's wordlist classification is disputed in the literature — included in train as
  borderline, flagged as such.
- Charbot is adversarially-generated (built specifically to evade classifiers, not a real
  malware family) — included in train for robustness; worth noting this connects to a
  broader known threat category of adversarial wordlist-DGA generation designed to attack
  ML classifiers specifically (e.g. WordDGA-style evasion techniques in recent literature).
- ngioweb and pizd match Leyva La O (2026)'s own zero-shot protocol so are directly
  comparable to their 81–85% ceiling; matsnu is a supplementary result under a harder
  condition — see RESOLVED section for the ready-to-use sentence.
- manuelita is evaluated as val-unseen, not test-unseen — chosen deliberately for
  model-selection visibility on a known-hard family; worth one sentence on the trade-off
  (rigor of a clean blind score vs. practical derisking on a 10-day deadline).
- **UMUDGA domains are deterministically generated** by executing each malware family's
  actual DGA code with a fixed random seed — not observed in live sinkhole/passive-DNS
  traffic. The domain *strings* are realistic (it's the real algorithm), but registration
  timing, resolution behavior, and other real-world signals aren't present.
- **Benign set is a top-domains-style list** (UMUDGA's `legit` folder) — a known "easy
  mode" critique in security-ML methodology, since top-1M-style benign domains are
  structurally very different from algorithmically generated ones. Real-world deployment
  would face harder negatives (recently-registered domains, NXDomain traffic).
- **Generalization beyond this benchmark isn't guaranteed.** Recent work (a March 2026
  paper evaluating DGA detectors against real-world smishing-driven domain campaigns from
  2022–2025) found that both traditional and ML-based detectors — trained and validated on
  standard malware-family corpora like this one — generalized poorly to newer real-world
  tactics that blend dictionary words with brand tokens and light randomization. Strong
  performance here is evidence of wordlist-family generalization specifically within this
  benchmark's threat model, not a general robustness guarantee.

---

## PHASE CHECKLIST

### Phase 0 — Setup (Day 0)
- [x] Resolve test-known/test-generalization flag — rebalanced split (see RESOLVED section)
- [ ] Confirm UMUDGA access/download works (Mendeley link, ~might need registration)
- [ ] Confirm HF repo download works (`train_1M.csv`, `train_wl.csv`, `test-known/`, `test-generalization/`)
- [ ] Confirm family-version grouping rule is understood before writing any loader code
      (Section 1 — fobber/kraken/murofet/ranbyus/vawtrak versions must stay together)
- [ ] Decide + write down preprocessing conventions (Section 4): TLD-strip, non-ASCII
      handling — before any code touches the data, not after
- [ ] Decide word-segmentation approach for Branch B (`wordninja` package vs. custom DP) and
      confirm the pretrained FastText `.bin` vectors are downloadable in your training
      environment (Kaggle: enable internet in notebook settings)
- [ ] Set up repo/folder structure: `data/raw/`, `data/processed/`, `src/`, `notebooks/`, `results/`
- [ ] Pin package versions (fasttext, wordninja, torch/tf, sklearn, pandas, transformers for
      ModernBERT) — write `requirements.txt` immediately

### Phase 1 — Data Pipeline (Days 1–2)
- [ ] Write data-loading script: pull family CSVs from UMUDGA + HF sources
- [ ] Apply the split table (Section 2) — hardcode family→split mapping as a config, not
      inline, grouped by base family name (see version-grouping rule)
- [ ] Apply sample caps (Section 3)
- [ ] Pull + balance `legit` 1:1 per split
- [ ] Apply preprocessing pipeline (Section 4): lowercase, TLD-strip, non-ASCII handling,
      length filter
- [ ] Dedup check by actual domain string (not just family name) — within AND across
      splits; HF repo's README credits UMUDGA as an underlying source, so overlap risk is
      real, not hypothetical
- [ ] Verify actual available sample counts for bigviktor, ngioweb, pizd specifically — HF's
      own eval set had only ~2,001/2,001/9,560 — confirm you can actually hit the
      1,500–2,000 cap
- [ ] Build char vocabulary (Branch A) from TRAIN split only
- [ ] Build/verify word-segmentation pipeline (Branch B) on a handful of known wordlist
      domains by hand (e.g. confirm `reindeerhorse` → `reindeer`+`horse`) before trusting it
      at scale
- [ ] Sanity-check family counts against the table — print a table of actual counts vs. planned
- [ ] Save processed splits to disk (train.parquet / val_unseen.parquet / test_unseen.parquet
      or similar) so you never regenerate test-unseen accidentally

### Phase 2 — Baselines (Days 2–4) — now four baselines, budget an extra day
- [ ] RF / XGBoost on hand-crafted lexical features (entropy, n-gram freq, length, digit
      ratio, vowel ratio, etc.) — local, CPU
- [ ] Char-only BiLSTM baseline — local, fits 8GB easily
- [ ] FastText-only baseline (frozen embeddings, same segmentation pipeline as Branch B) —
      local, fits 8GB easily
- [ ] **ModernBERT-base fine-tune on YOUR OWN split — kick off on Kaggle as early as
      possible, run in parallel with the rest of Phase 2/3 (see Compute Budget)**
- [ ] Confirm end-to-end pipeline works: train → val-unseen eval → logged metrics
- [ ] These numbers go in the final comparison table — save them now, don't rerun later

### Phase 3 — Dual-Branch Model (Days 3–5)
- [ ] Implement Branch A (char-level BiLSTM — not a from-scratch Transformer, see Section 5)
- [ ] Implement Branch B (frozen pretrained FastText `.bin` over DP-segmented tokens)
- [ ] Implement fusion (concat + MLP head to start) + classification head
- [ ] Implement stratified/weighted batch sampler (Section 6) — start at 1:1 wordlist:non-wordlist
- [ ] Training loop with val-unseen checkpointing on the wordlist-specific primary criterion
      (Section 6)
- [ ] Log per-family val-unseen F1/recall every epoch (wordlist vs non-wordlist split out)

### Phase 4 — Tuning (Days 5–7)
- [ ] Architecture/hyperparameter decisions based on val-unseen ONLY
- [ ] Specifically check manuelita's val-unseen score, not just the aggregate — it's the
      known-hard family; if the model is hopeless on it, this is your chance to fix it
- [ ] Tune the stratified-sampling ratio if 1:1 isn't working well (guardrail: non-wordlist
      val score shouldn't collapse)
- [ ] Compare dual-branch vs. all FOUR baselines (incl. ModernBERT) on val-unseen
- [ ] If time allows: attention-fusion ablation vs. concat
- [ ] Freeze best checkpoint — write down exactly which checkpoint/config this is

### Phase 5 — Final Test-Unseen Eval (Day 7–8) — ⚠️ ONCE ONLY
- [ ] Confirm training + model selection are fully frozen before opening test-unseen
- [ ] Load best checkpoint
- [ ] Run test-unseen evaluation exactly once — dual-branch AND all four baselines (incl.
      your own ModernBERT fine-tune) on the identical test-unseen set
- [ ] Report per-family F1/recall/precision: tinba, murofet, matsnu, ngioweb, pizd
- [ ] Sanity check: non-wordlist (tinba, murofet) should score high
- [ ] Headline result: wordlist test families vs. non-wordlist gap
- [ ] Benchmark comparison: ngioweb + pizd vs. Leyva La O (2026) 80.9–84.6% ceiling (fair,
      since it's the same protocol); matsnu reported separately as the harder supplementary
      case
- [ ] Check which of the "possible honest wins" from Section 5 the numbers actually support
      — aggregate, per-family, sample-efficiency, or parameter/latency-efficiency — and
      build the results narrative around whichever is real, not whichever was hoped for

### Phase 6 — Failure Analysis (Day 8)
- [ ] Misclassified test-unseen samples, especially matsnu (known-hard family, still blind here)
- [ ] Revisit manuelita's val-unseen misclassifications too (caught earlier in Phase 4, but
      worth including in the writeup's failure analysis alongside the test-unseen families)
- [ ] Qualitative examples for the writeup (a few misclassified domains + why) — check
      specifically whether failures correlate with segmentation errors on Branch B
- [ ] Compare dual-branch's specific error cases against ModernBERT's — this is where a
      "beats ModernBERT on segmentation-sensitive cases specifically" claim would show up
      even if the aggregate doesn't win

### Phase 7 — Writeup (Days 8–10)
- [ ] Methodology section — including the preprocessing decisions from Section 4 (TLD-strip,
      segmentation approach) documented explicitly, not left implicit
- [ ] Results table (all four baselines + dual-branch, val-unseen + test-unseen)
- [ ] Limitations section (Section 8 above, finalized)
- [ ] Compare against Drichel (2022) floor / Leyva La O (2026) ceiling — using YOUR OWN
      ModernBERT baseline numbers for the "beat ModernBERT" claim, not the paper's published
      numbers (see Section 5)
- [ ] State clearly which of the "possible honest wins" (Section 5) the results support
- [ ] Final pass / submission

---

## Log

_(Append dated notes here as you go — decisions made, blockers hit, numbers obtained.)_

- **2026-08-14** — Master plan file created. Cross-checked HF repo structure; flagged
  matsnu/manuelita/ngioweb split mismatch with upstream test-known/test-generalization
  folders.
- **2026-08-14** — Resolved: keeping matsnu/manuelita/ngioweb as test-unseen trio initially,
  own split logic independent of repo's known/generalization labels.
- **2026-08-14** — Rebalanced (v2): confirmed HF repo is the actual Leyva La O (2026) paper
  release — their "known" families (incl. matsnu, manuelita) were TRAINED on, only
  bigviktor/ngioweb/pizd were their true zero-shot set, matching the cited 81–85% ceiling.
  Moved manuelita test→val, pizd train→test. New test-unseen wordlist: matsnu, ngioweb,
  pizd. New val-unseen wordlist: suppobox, bigviktor, manuelita.
- **2026-08-14** — Full plan review (v3): verified every external claim via search (nothing
  left from training-data memory). Key additions: (1) ModernBERT-base added as a 4th
  baseline, trained/evaluated on our own split — required for any valid "beat ModernBERT"
  claim, since the paper's 80.9-84.6% was measured on a different split; (2) architecture
  locked to BiLSTM (Branch A) + frozen pretrained FastText .bin over DP-segmented tokens
  (Branch B), grounded in Koh & Rhodes's directly-on-topic precedent (frozen embeddings
  beat char-only on wordlist DGA, SOTA with ~100 examples); (3) family-count imbalance
  (26 vs 6 train families) addressed via stratified batch sampling + wordlist-specific
  checkpoint criterion, not by changing sample caps; (4) new Data Preprocessing and Compute
  Budget sections added; (5) confirmed Drichel 2022 floor (19.444%, exact match), UMUDGA
  specifics (50 families, deterministic seeded generation), Kaggle free tier (30hrs/week,
  T4x2/P100, 12hr session cap); (6) added limitations re: UMUDGA's lab-generated (not
  in-the-wild) domains, benign-set easy-mode critique, and a March 2026 finding that DGA
  detectors trained on standard corpora like this one don't necessarily generalize to newer
  real-world smishing-driven tactics.
- **2026-08-24** — Cross-checked against the team's literature review document. Found and
  fixed a citation error in this file: the frozen-embedding precedent is Koh & Rhodes
  (2018), IEEE Big Data — not "Highnam et al." as an earlier pass of this file said.
  Verified directly against the paper's arXiv full text and results table (89.5%/91.2%
  confirmed exact). Also verified independently: Lison & Mavroeidis (2017)'s Matsnu
  15.8% recall figure (confirmed via their per-family table, with a nuance — it's a joint
  61-family model, not an isolated single-family experiment) and Drichel (2022)'s 19.444%
  floor (confirmed). One number could NOT be re-verified: Manuelita's claimed 43.1% F1
  "even when trained on it directly" — don't cite it without pulling it from the paper's
  own tables directly. Literature review doc updated and reconciled to match this file's
  split/architecture decisions (Koh & Rhodes frozen-embedding design, 4 baselines incl.
  ModernBERT-base on own split, v2 split with pizd/manuelita swap).
