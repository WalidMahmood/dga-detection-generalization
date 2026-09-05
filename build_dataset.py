#!/usr/bin/env python3
"""
Build unified DGA killer dataset per CSE427 Master Plan v2
- Merge UMUDGA + HF sources, dedup, apply v2 split, caps, balanced benign.
- Outputs raw and processed splits + reports + vocab.
"""
import pathlib, gzip, csv, json, random, re, collections
from pathlib import Path
import pandas as pd
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE = Path(r"G:\dga_killer")
UMUDGA_BASE = BASE / "UMUDGA - University of Murcia Domain Generation Algorithm Dataset" / "UMUDGA - University of Murcia Domain Generation Algorithm Dataset" / "Fully Qualified Domain Names"
HF_BASE = BASE / "moe-wordlist-dga-models" / "datasets"

OUT_ROOT = BASE / "data"
RAW_DIR = OUT_ROOT / "raw"
PROC_DIR = OUT_ROOT / "processed"
REPORT_DIR = OUT_ROOT / "reports"
VOCAB_DIR = OUT_ROOT / "vocab"

for d in [RAW_DIR, PROC_DIR, REPORT_DIR, VOCAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1. Split config v2
# ------------------------------------------------------------------
TRAIN_NON_WORDLIST = ["alureon","bedep","ccleaner","chinad","corebot","cryptolocker","dircrypt","dyre","fobber","locky","necurs","padcrypt","proslikefan","pushdo","pykspa","pykspa_noise","qadars","qakbot","ramdo","ramnit","shiotob","simda","sisron","symmi","tempedreve","vawtrak"]
TRAIN_WORDLIST = ["gozi","rovnix","nymaim","banjori","charbot","deception"]

VAL_NON_WORDLIST = ["zeus-newgoz","ranbyus","kraken"]
VAL_WORDLIST = ["suppobox","bigviktor","manuelita"]

TEST_NON_WORDLIST = ["tinba","murofet"]
TEST_WORDLIST = ["matsnu","ngioweb","pizd"]

FAMILY_TO_SPLIT = {}
for f in TRAIN_NON_WORDLIST+TRAIN_WORDLIST:
    FAMILY_TO_SPLIT[f]="train"
for f in VAL_NON_WORDLIST+VAL_WORDLIST:
    FAMILY_TO_SPLIT[f]="val_unseen"
for f in TEST_NON_WORDLIST+TEST_WORDLIST:
    FAMILY_TO_SPLIT[f]="test_unseen"

WORDLIST_SET = set(TRAIN_WORDLIST + VAL_WORDLIST + TEST_WORDLIST)
NON_WORDLIST_SET = set(TRAIN_NON_WORDLIST + VAL_NON_WORDLIST + TEST_NON_WORDLIST)

GROUP_MAP = {
    'fobber': ['fobber_v1','fobber_v2'],
    'gozi': ['gozi_gpl','gozi_luther','gozi_nasa','gozi_rfc4343'],
    'kraken': ['kraken_v1','kraken_v2'],
    'murofet': ['murofet_v1','murofet_v2','murofet_v3'],
    'ranbyus': ['ranbyus_v1','ranbyus_v2'],
    'suppobox': ['suppobox_1','suppobox_2','suppobox_3'],
    'vawtrak': ['vawtrak_v1','vawtrak_v2','vawtrak_v3'],
}
# reverse
VARIANT_TO_BASE = {}
for base, variants in GROUP_MAP.items():
    for v in variants:
        VARIANT_TO_BASE[v]=base

# Caps
CAP_TRAIN = 20000
CAP_VAL = 2000
CAP_TEST = 2000

# Helper to get base family from raw folder name
def base_family(name):
    return VARIANT_TO_BASE.get(name, name)

# ------------------------------------------------------------------
# 2. Collect UMUDGA domains (union of largest file per variant, lowercased)
# ------------------------------------------------------------------
print("Collecting UMUDGA domains ...")
umudga_pools = collections.defaultdict(set)  # base_family -> set(domain)
for p in UMUDGA_BASE.iterdir():
    if not p.is_dir():
        continue
    if p.name=="legit":
        continue
    list_dir = p/"list"
    if not list_dir.exists():
        continue
    txts = list(list_dir.glob("*.txt"))
    if not txts:
        continue
    # take largest file available for best supply (as earlier inventory)
    biggest = max(txts, key=lambda x: int(x.stem))
    # For grouped families we will later merge, but reading largest per variant is enough for variant-level
    # For families with multiple variants, we read each variant's largest separately and later union
    domains = set()
    try:
        domains = set(l.strip().lower() for l in open(biggest, encoding="utf-8", errors="ignore") if l.strip())
    except Exception as e:
        print(f"error reading {biggest}: {e}")
        continue
    b = base_family(p.name)
    umudga_pools[b].update(domains)

# legit separately
legit_pools_umudga = set()
legit_list_dir = UMUDGA_BASE/"legit"/"list"
biggest_legit = max(list(legit_list_dir.glob("*.txt")), key=lambda x: int(x.stem))
legit_pools_umudga = set(l.strip().lower() for l in open(biggest_legit, encoding="utf-8", errors="ignore") if l.strip())
print(f"UMUDGA legit pool {len(legit_pools_umudga)} from {biggest_legit.name}")

# ------------------------------------------------------------------
# 3. Collect HF domains
# ------------------------------------------------------------------
print("Collecting HF domains ...")
hf_pools = collections.defaultdict(set)

def add_hf_family(fam, domain):
    if domain:
        hf_pools[fam].add(domain.strip().lower())

# train_wl.csv
with open(HF_BASE/"train_wl.csv", encoding="utf-8", errors="ignore") as f:
    for row in csv.DictReader(f):
        add_hf_family(row["family"].strip().lower(), row["domain"])

# train_1M.csv
with open(HF_BASE/"train_1M.csv", encoding="utf-8", errors="ignore") as f:
    for row in csv.DictReader(f):
        fam = row["family"].strip().lower()
        if fam=="legit":
            continue
        add_hf_family(fam, row["domain"])

# test-known
for gz in (HF_BASE/"test-known").glob("*.gz"):
    fam = gz.stem.lower()
    lines = gzip.open(gz,"rt",encoding="utf-8",errors="ignore").read().splitlines()
    if not lines:
        continue
    header = lines[0].lower()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        # robust domain extraction: find token with dot
        dom=None
        # Heuristic: domain is usually second column if leading empty, else first that looks like domain
        # We'll scan parts for containing '.'
        for p in parts:
            p=p.strip()
            if "." in p and p and p.lower() not in ("dga","notdga","legit") and "/" not in p:
                # prefer p that matches domain pattern (no spaces, has dot, length >3)
                if len(p)>3 and " " not in p:
                    dom=p
                    break
        if dom is None:
            # fallback: parts[1] if exists
            if len(parts)>1 and parts[1].strip():
                dom=parts[1].strip()
            else:
                dom=parts[0].strip()
        add_hf_family(fam, dom)

# test-generalization
for gz in (HF_BASE/"test-generalization").glob("*.gz"):
    fam = gz.stem.lower()
    lines = gzip.open(gz,"rt",encoding="utf-8",errors="ignore").read().splitlines()
    if not lines:
        continue
    for line in lines[1:]:
        if not line.strip():
            continue
        dom = line.split(",")[0].strip()
        add_hf_family(fam, dom)

print(f"HF families collected: {sorted(hf_pools.keys())}")
for k in sorted(hf_pools):
    print(f" HF {k:15s} {len(hf_pools[k]):6d}")

# ------------------------------------------------------------------
# 4. Build unified pools per master family (43 families)
# ------------------------------------------------------------------
# For each master family, union UMUDGA + HF if both exist
all_master_families = set(FAMILY_TO_SPLIT.keys())  # 32+6+5 =43
# But we also want to verify pool sizes for these families
unified_pools = {}
for fam in all_master_families:
    s=set()
    if fam in umudga_pools:
        s.update(umudga_pools[fam])
    if fam in hf_pools:
        s.update(hf_pools[fam])
    unified_pools[fam]=s

# For debugging, also keep per-source breakdown
pool_report = {}
for fam in sorted(all_master_families):
    um = len(umudga_pools.get(fam, set()))
    hf = len(hf_pools.get(fam, set()))
    uni = len(unified_pools[fam])
    inter = len(umudga_pools.get(fam,set()) & hf_pools.get(fam,set())) if fam in umudga_pools and fam in hf_pools else 0
    pool_report[fam] = {"umudga": um, "hf": hf, "union": uni, "overlap": inter, "split": FAMILY_TO_SPLIT[fam], "wordlist": fam in WORDLIST_SET}

with open(REPORT_DIR/"pool_inventory.json","w",encoding="utf-8") as f:
    json.dump(pool_report, f, indent=2)

print("\nUnified pools for master families:")
for fam in sorted(pool_report, key=lambda x: (pool_report[x]["split"], x)):
    r=pool_report[fam]
    print(f"{fam:15s} split={r['split']:12s} wl={str(r['wordlist']):5s} um={r['umudga']:7d} hf={r['hf']:7d} union={r['union']:7d} overlap={r['overlap']:7d}")

# ------------------------------------------------------------------
# 5. Sampling per split with caps, deterministic
# ------------------------------------------------------------------
def sample_family(pool_set, cap, seed):
    # deterministic sample: sort to make reproducible, then sample via numpy
    pool_list = sorted(pool_set)  # sorted ensures reproducibility before sampling
    if len(pool_list) <= cap:
        return pool_list
    rng = np.random.RandomState(seed)
    # use choice without replacement
    idx = rng.choice(len(pool_list), size=cap, replace=False)
    return [pool_list[i] for i in idx]

# We'll assign distinct seed per family for variety: hash(fam) mod large + SEED
def seed_for(fam, split):
    # stable hash
    import hashlib
    h=int(hashlib.md5(f"{fam}_{split}".encode()).hexdigest()[:8],16)
    return (SEED + h) % (2**31-1)

raw_records = []  # list of dicts for raw splits
# We'll also track used domains across splits to enforce dedup across splits
used_domains_global = set()
dedup_log = []

for split_name, cap in [("train", CAP_TRAIN), ("val_unseen", CAP_VAL), ("test_unseen", CAP_TEST)]:
    families_in_split = [f for f, s in FAMILY_TO_SPLIT.items() if s==split_name]
    print(f"\n--- Sampling {split_name} cap {cap} families {families_in_split} ---")
    for fam in sorted(families_in_split):
        pool = unified_pools.get(fam, set())
        if not pool:
            print(f"WARNING: pool empty for {fam}")
            continue
        # Remove already used domains globally (dedup across splits)
        avail = pool - used_domains_global
        if len(avail) < len(pool):
            removed = len(pool)-len(avail)
            dedup_log.append({"family":fam,"split":split_name,"removed_cross_split":removed})
        sampled = sample_family(avail, cap, seed_for(fam, split_name))
        # If sampled less than cap because pool exhausted, log
        actual = len(sampled)
        expected = min(cap, len(pool))
        print(f" {fam:15s} pool {len(pool):7d} avail {len(avail):7d} sampled {actual:5d} (cap {cap})")
        # add to global used
        used_domains_global.update(sampled)
        for dom in sampled:
            raw_records.append({"domain_raw": dom, "family": fam, "split": split_name, "label": "dga", "wordlist": fam in WORDLIST_SET, "is_benign": False})

# Now sample benign 1:1 per split
# legit pool is umudga legit (1M). Remove any that overlap with malicious used domains (extremely unlikely but check)
legit_avail = legit_pools_umudga - used_domains_global
print(f"\nLegit pool total {len(legit_pools_umudga)} avail after malicious dedup {len(legit_avail)}")

# Need to sample legit per split matching malicious counts
# Count malicious per split
from collections import Counter
mal_counts = Counter(r["split"] for r in raw_records)
print(f"Malicious counts per split: {mal_counts}")

# For legit, we need to sample per split without replacement across splits
legit_list_sorted = sorted(legit_avail)
rng_legit = np.random.RandomState(SEED+999)
# shuffle indices
idxs = rng_legit.permutation(len(legit_list_sorted))
# iterate splits
ptr=0
for split_name in ["train","val_unseen","test_unseen"]:
    need = mal_counts[split_name]
    if ptr+need > len(legit_list_sorted):
        raise ValueError(f"Not enough legit domains: need {need} for {split_name} but only {len(legit_list_sorted)-ptr} left")
    sampled_legit = [legit_list_sorted[idxs[ptr+i]] for i in range(need)]
    ptr+=need
    print(f" Legit {split_name:12s} need {need:6d} sampled {len(sampled_legit)} ptr now {ptr}")
    for dom in sampled_legit:
        raw_records.append({"domain_raw": dom, "family": "legit", "split": split_name, "label": "benign", "wordlist": False, "is_benign": True})
    # also add to global used (though not needed for legit dedup across splits, but ensures no overlap)
    used_domains_global.update(sampled_legit)

# Check for any remaining cross-split duplicates (should be zero)
# Build dataframe
df_raw = pd.DataFrame(raw_records)
print(f"\nTotal raw records {len(df_raw)}")
print(df_raw.groupby(["split","label"]).size())
print(df_raw.groupby(["split","family"]).size().head(20))

# Dedup check within split
dup_within = df_raw.duplicated(subset=["domain_raw"], keep=False).sum()
print(f"Duplicates within entire raw set (should be 0 after our dedup): {dup_within}")
# Also dup across splits by domain_raw grouped
cross_dup = df_raw.groupby("domain_raw").filter(lambda x: x["split"].nunique()>1)
print(f"Cross-split duplicates remaining: {len(cross_dup)}")

# Save dedup log
with open(REPORT_DIR/"dedup_log.json","w",encoding="utf-8") as f:
    json.dump(dedup_log, f, indent=2)

# ------------------------------------------------------------------
# 6. Save raw splits
# ------------------------------------------------------------------
for split in ["train","val_unseen","test_unseen"]:
    sub = df_raw[df_raw["split"]==split].copy()
    # shuffle deterministically for final file order
    sub = sub.sample(frac=1, random_state=SEED).reset_index(drop=True)
    out_path = RAW_DIR / f"{split}_raw.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved raw {split} -> {out_path} rows {len(sub)}")

# Also save combined all-domains raw for convenience (single file)
df_raw_shuffled = df_raw.sample(frac=1, random_state=SEED).reset_index(drop=True)
df_raw_shuffled.to_csv(OUT_ROOT / "all_domains_raw.csv", index=False)
print(f"Saved combined raw -> {OUT_ROOT/'all_domains_raw.csv'}")

# Inventory report for raw
raw_inventory = {}
for split in ["train","val_unseen","test_unseen"]:
    sub = df_raw[df_raw["split"]==split]
    raw_inventory[split] = {
        "total": int(len(sub)),
        "malicious": int((sub["label"]=="dga").sum()),
        "benign": int((sub["label"]=="benign").sum()),
        "families": sub["family"].value_counts().to_dict(),
        "wordlist_malicious": int(((sub["label"]=="dga") & (sub["wordlist"]==True)).sum()),
        "nonwordlist_malicious": int(((sub["label"]=="dga") & (sub["wordlist"]==False)).sum()),
    }
with open(REPORT_DIR/"raw_inventory.json","w",encoding="utf-8") as f:
    json.dump(raw_inventory, f, indent=2)

# ------------------------------------------------------------------
# 7. Preprocessing function per master plan Sec 4
# ------------------------------------------------------------------
USE_TLDEXTRACT = False
extractor = None
try:
    import tldextract
    USE_TLDEXTRACT = True
    # Disable network fetch, use fallback
    extractor = tldextract.TLDExtract(suffix_list_urls=None, cache_dir=False)
    print("tldextract available, using it for TLD stripping")
except Exception as e:
    print(f"tldextract not available ({e}), fallback to simple split")
    USE_TLDEXTRACT=False
    extractor=None

def strip_tld(domain):
    d = domain.strip().lower()
    # remove protocol/path if any (should not exist)
    d = d.split("/")[0]
    # Remove port
    d = d.split(":")[0]
    # Use tldextract if available: we want SLD label = domain without suffix and without subdomain?
    # For DGA, domains are typically <sld>.<tld>, no subdomains. So we take sld = extracted.domain
    # If no suffix recognized, fallback to rsplit.
    if USE_TLDEXTRACT and extractor:
        ext = extractor(d)
        if ext.domain:
            # ext.domain is the SLD, we intentionally discard suffix and subdomain
            # But if domain is something like 'a.b.example.com' with subdomains, we still just keep 'example'
            # That's exactly the SLD label definition per master plan.
            return ext.domain
        else:
            # fallback
            if "." in d:
                return d.rsplit(".",1)[0].split(".")[-1]
            return d
    else:
        if "." in d:
            # take label before last dot, then if multiple dots, take last label before TLD
            # e.g. 'sub.example.com' -> 'example'
            # So split by '.', take second last if >1 dot else first
            parts = d.split(".")
            if len(parts)>=2:
                return parts[-2]
            else:
                return parts[0]
        return d

def has_non_ascii(s):
    try:
        s.encode("ascii")
        return False
    except:
        return True

# Apply preprocessing
df_proc = df_raw.copy()
# Step1 lowercase everything already done, but ensure
df_proc["domain_raw_lower"] = df_proc["domain_raw"].str.lower()
# Step2 strip TLD to get SLD
df_proc["sld"] = df_proc["domain_raw_lower"].apply(strip_tld)
# Step 4 non ascii flag
df_proc["has_non_ascii"] = df_proc["sld"].apply(has_non_ascii)
# Step 5 length
df_proc["sld_len"] = df_proc["sld"].str.len()
# Step flag empty / single char / absurdly long
df_proc["is_empty"] = df_proc["sld"].str.len()==0
df_proc["is_single_char"] = df_proc["sld"].str.len()==1
df_proc["is_absurdly_long"] = df_proc["sld"].str.len()>63  # per DNS label limit, plus some leeway to 100 might be allowed but we use 63 as strict; will report >63 then allow up to 100? Master says don't be aggressive, so we keep >63 as flag but not drop beyond 100
df_proc["is_too_long"] = df_proc["sld"].str.len()>100

# Stats before filtering
pre_filter_stats = {
    "total": len(df_proc),
    "non_ascii_count": int(df_proc["has_non_ascii"].sum()),
    "empty_count": int(df_proc["is_empty"].sum()),
    "single_char_count": int(df_proc["is_single_char"].sum()),
    "absurdly_long_63": int(df_proc["is_absurdly_long"].sum()),
    "too_long_100": int(df_proc["is_too_long"].sum()),
}
print(f"Pre-filter stats: {pre_filter_stats}")

# Apply filters per master: drop empty, single-char, non-ascii? Decision: drop non-ascii OR normalize? We'll drop non-ascii entries (uniform treatment benign/malicious)
# Also drop absurdly long >100? But master says don't be aggressive, some families legitimately produce long labels. We'll only drop >100 and empty/single.
# Count how many would be dropped
mask_drop = df_proc["is_empty"] | df_proc["is_single_char"] | df_proc["has_non_ascii"] | df_proc["is_too_long"]
print(f"Would drop {mask_drop.sum()} rows due to empty/single/non-ascii/>100")

# Actually keep them for now unless they exist, then drop
df_proc_filtered = df_proc[~mask_drop].copy()
print(f"After filter: {len(df_proc_filtered)} (removed {len(df_proc)-len(df_proc_filtered)})")

# Dedup by sld within and across splits (stricter than domain_raw dedup)
# First dedup by exact domain_raw already done; now dedup by sld+TLD? Master says dedup by exact domain string within and across splits.
# But we additionally check sld collision across splits (different TLD same SLD) - this could hide leakage.
# We'll perform sld dedup across splits: keep earliest split (train > val > test), remove later duplicates
# Need to order splits priority
split_priority = {"train":0, "val_unseen":1, "test_unseen":2}
df_proc_filtered["priority"] = df_proc_filtered["split"].map(split_priority)
# Sort by priority then drop duplicates keeping first
df_proc_filtered_sorted = df_proc_filtered.sort_values(["sld","priority"])
before_sld_dedup = len(df_proc_filtered_sorted)
df_proc_deduped = df_proc_filtered_sorted.drop_duplicates(subset=["sld"], keep="first")
after_sld_dedup = len(df_proc_deduped)
sld_dup_removed = before_sld_dedup - after_sld_dedup
print(f"SLD dedup removed {sld_dup_removed} rows (different TLD same SLD across splits)")
# Also dedup within split by sld (should be zero after global dedup, but check)
# We'll log sld duplicates that were across splits
if sld_dup_removed>0:
    dups = df_proc_filtered_sorted[df_proc_filtered_sorted.duplicated(subset=["sld"], keep=False)].sort_values("sld")
    dups.to_csv(REPORT_DIR/"sld_duplicates.csv", index=False)
    print(f"Saved sld duplicates to {REPORT_DIR/'sld_duplicates.csv'}")

# Remove priority column, sort back by split for file output? For final files we will sort by split then shuffle
df_proc_final = df_proc_deduped.drop(columns=["priority"]).copy()

# If we removed some due to sld dedup, benign 1:1 ratio may be slightly off now; report
post_counts = df_proc_final.groupby(["split","label"]).size()
print(f"Post-proc counts per split/label:\n{post_counts}")

# We should rebalance benign after dedup? Simpler to report imbalance and optionally rebalance by sampling additional benign or dropping? But dedup removed should be tiny.
# If imbalance >1% we could adjust, but we'll just report and keep as is, noting in report.

# ------------------------------------------------------------------
# 8. Build char vocab from TRAIN only (SLDs)
# ------------------------------------------------------------------
train_slds = df_proc_final[(df_proc_final["split"]=="train") & (df_proc_final["label"]=="dga") | (df_proc_final["split"]=="train") ]  # Actually all train
# Simpler: all train slds
train_slds = df_proc_final[df_proc_final["split"]=="train"]["sld"].tolist()
all_chars = set("".join(train_slds))
# sort
char_vocab = sorted(all_chars)
# Add special tokens? master says small set a-z,0-9,- . Include mapping
vocab_dict = {ch: idx+4 for idx, ch in enumerate(char_vocab)}  # reserve 0 PAD,1 UNK,2 SOS,3 EOS maybe
vocab_dict["<PAD>"]=0
vocab_dict["<UNK>"]=1
vocab_dict["<SOS>"]=2
vocab_dict["<EOS>"]=3
# invert? keep extra
print(f"Char vocab size {len(char_vocab)} (unique chars) total with specials {len(vocab_dict)}")
print(f"Chars: {''.join(char_vocab)}")

with open(VOCAB_DIR/"char_vocab.json","w",encoding="utf-8") as f:
    json.dump({"chars": char_vocab, "vocab": vocab_dict, "size": len(vocab_dict)}, f, indent=2)

# Also save simple text file
with open(VOCAB_DIR/"char_vocab.txt","w",encoding="utf-8") as f:
    for ch in char_vocab:
        f.write(ch+"\n")

# ------------------------------------------------------------------
# 9. Save processed splits
# ------------------------------------------------------------------
# Keep columns for modelling: domain_raw, sld, family, split, label (dga/benign), label_binary (1/0), wordlist, sld_len, has_non_ascii etc but core
df_proc_final["label_binary"] = (df_proc_final["label"]=="dga").astype(int)
# Reorder columns
cols = ["domain_raw","sld","sld_len","family","split","label","label_binary","wordlist","is_benign","has_non_ascii"]
# Ensure exists
cols = [c for c in cols if c in df_proc_final.columns]
for split in ["train","val_unseen","test_unseen"]:
    sub = df_proc_final[df_proc_final["split"]==split].copy()
    # shuffle deterministic
    sub = sub.sample(frac=1, random_state=SEED).reset_index(drop=True)
    out_path = PROC_DIR / f"{split}.csv"
    sub[cols].to_csv(out_path, index=False)
    # also parquet if pyarrow available
    try:
        sub[cols].to_parquet(PROC_DIR / f"{split}.parquet", index=False)
    except Exception as e:
        print(f"Parquet failed {e}")
    print(f"Saved processed {split} -> {out_path} rows {len(sub)}")

# Combined
df_proc_final_shuffled = df_proc_final.sample(frac=1, random_state=SEED).reset_index(drop=True)
df_proc_final_shuffled[cols].to_csv(OUT_ROOT / "all_domains_processed.csv", index=False)
print(f"Saved combined processed -> {OUT_ROOT/'all_domains_processed.csv'}")

# ------------------------------------------------------------------
# 10. Reports
# ------------------------------------------------------------------
proc_inventory = {}
for split in ["train","val_unseen","test_unseen"]:
    sub = df_proc_final[df_proc_final["split"]==split]
    proc_inventory[split] = {
        "total": int(len(sub)),
        "malicious": int((sub["label"]=="dga").sum()),
        "benign": int((sub["label"]=="benign").sum()),
        "families": sub["family"].value_counts().to_dict(),
        "wordlist_malicious": int(((sub["label"]=="dga") & (sub["wordlist"]==True)).sum()),
        "nonwordlist_malicious": int(((sub["label"]=="dga") & (sub["wordlist"]==False)).sum()),
        "avg_sld_len": float(sub["sld_len"].mean()),
        "median_sld_len": float(sub["sld_len"].median()),
    }

with open(REPORT_DIR/"processed_inventory.json","w",encoding="utf-8") as f:
    json.dump(proc_inventory, f, indent=2)

# Before/after comparison
comparison = {
    "raw": raw_inventory,
    "processed": proc_inventory,
    "pre_filter_stats": pre_filter_stats,
    "sld_dup_removed": int(sld_dup_removed),
    "dropped_empty_single_nonascii": int(mask_drop.sum()),
}
with open(REPORT_DIR/"before_after_comparison.json","w",encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)

# Summary markdown
summary_md = f"""# DGA Killer Dataset Build Summary

**Seed:** {SEED}
**Caps:** train {CAP_TRAIN}/family, val {CAP_VAL}, test {CAP_TEST}, benign 1:1 per split
**TLD handling:** {'tldextract' if USE_TLDEXTRACT else 'simple rsplit'}
**Char vocab:** {len(char_vocab)} unique chars + 4 specials = {len(vocab_dict)}

## Pool Inventory (Union per Master Family)

"""
for fam in sorted(pool_report, key=lambda x: (pool_report[x]["split"], x)):
    r=pool_report[fam]
    summary_md += f"- {fam:15s} split={r['split']:12s} wl={r['wordlist']} um={r['umudga']:7d} hf={r['hf']:7d} union={r['union']:7d}\n"

summary_md += "\n## Raw Counts\n"
for split, v in raw_inventory.items():
    summary_md += f"- {split}: total {v['total']} mal {v['malicious']} ben {v['benign']} wl_mal {v['wordlist_malicious']} nonwl {v['nonwordlist_malicious']}\n"

summary_md += "\n## Processed Counts\n"
for split, v in proc_inventory.items():
    summary_md += f"- {split}: total {v['total']} mal {v['malicious']} ben {v['benign']} wl_mal {v['wordlist_malicious']} nonwl {v['nonwordlist_malicious']} avg_len {v['avg_sld_len']:.2f}\n"

summary_md += f"\n## Preprocessing Dropped\n- empty/single/non-ascii/>100: {int(mask_drop.sum())}\n- SLD cross-split dedup removed: {sld_dup_removed}\n"

with open(REPORT_DIR/"SUMMARY.md","w",encoding="utf-8") as f:
    f.write(summary_md)

print("\n=== BUILD COMPLETE ===")
print(summary_md)
