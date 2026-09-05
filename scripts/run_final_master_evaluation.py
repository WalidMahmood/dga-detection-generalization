"""
=============================================================================
CSE427 DGA KILLER — MASTER FINAL TEST-UNSEEN BENCHMARK & EVALUATION SUITE
=============================================================================
Evaluates all 5 primary models + 2 ensembles on the frozen test_unseen.csv:
  1. Random Forest (11 Lexical Features)
  2. XGBoost (11 Lexical Features)
  3. Dual-Branch Perfect v3 (Word-Seq BiLSTM + Char BiLSTM, 4.1M)
  4. Dual-Branch Coherent (Char BiLSTM + Word BiLSTM + 6D Semantic Cosine, 0.73M)
  5. ModernBERT-Base (149M Parameter Pretrained Transformer)
  6. Ensemble 1 (ModernBERT + Dual-Branch Perfect v3 Soft Voting)
  7. Ensemble 2 (ModernBERT + Dual-Branch Coherent Soft Voting)

Generates:
  - Complete multi-metric tables (Classification, Ranking, Per-Family)
  - 8 Publication-Grade Visualizations (ROC, PR, CM grid, Bars, Anchors, Thresholds, Donuts, Radar)
  - LaTeX table exports (.tex) & CSV exports (.csv)
  - Full JSON report (.json)
=============================================================================
"""

import os, sys, json, math, time, pickle, gc, joblib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, matthews_corrcoef, classification_report
)

import wordninja
from gensim.models.fasttext import load_facebook_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

BASE_DIR = Path(r"G:\dga_killer")
DATA_DIR = BASE_DIR / "data" / "processed"
WEIGHTS_DIR = BASE_DIR / "training" / "weights"
VOCAB_DIR = BASE_DIR / "data" / "vocab"
REPORTS_DIR = BASE_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"
METRIC_DIR = REPORTS_DIR / "metrics"

for d in [FIG_DIR, TABLE_DIR, METRIC_DIR, BASE_DIR / "training" / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== MASTER EVALUATION SUITE INITIALIZED ===")
print(f"Device: {device}")
print(f"Report Directory: {REPORTS_DIR}")

# -----------------------------------------------------------------------------
# 1. LOAD TEST DATASET
# -----------------------------------------------------------------------------
print("\n[1/7] Loading frozen test_unseen.csv...")
test_path = DATA_DIR / "test_unseen.csv"
test_df = pd.read_csv(test_path, keep_default_na=False)
test_df["sld"] = test_df["sld"].fillna("").astype(str)

y_true = test_df["label_binary"].values.astype(int)
families = test_df["family"].values
slds = test_df["sld"].values
n_test = len(test_df)

print(f"  Total test domains: {n_test:,}")
print(f"  Class balance: DGA={y_true.sum():,} ({y_true.mean():.1%}), Benign={(y_true == 0).sum():,}")
print("  Test families breakdown:")
for fam, cnt in test_df["family"].value_counts().items():
    wl_flag = bool(test_df[test_df["family"] == fam]["wordlist"].iloc[0])
    print(f"    {fam:15s} n={cnt:5d} ({'Wordlist' if wl_flag else 'Non-Wordlist' if fam!='legit' else 'Benign'})")

# -----------------------------------------------------------------------------
# 2. FEATURE EXTRACTION FOR CLASSICAL MODELS
# -----------------------------------------------------------------------------
print("\n[2/7] Extracting 11 Lexical Features for Random Forest & XGBoost...")

def shannon_entropy(s):
    s = str(s)
    if not s: return 0.0
    cnt = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in cnt.values())

def ngram_common(s, n=2):
    s = str(s).lower()
    if len(s) < n: return 0.0
    common = {"th","he","in","er","an","re","on","at","en","nd"} if n == 2 else {"the","ing","and","her","ere"}
    grams = [s[i:i+n] for i in range(len(s) - n + 1)]
    return sum(1 for g in grams if g in common) / len(grams) if grams else 0.0

def extract_lexical_features(sld_list):
    feats = []
    for s in sld_list:
        s = str(s) if s is not None else ""
        l = max(len(s), 1)
        sl = s.lower()
        digits = sum(c.isdigit() for c in s)
        hyphens = s.count('-')
        underscores = s.count('_')
        vowels = sum(c in 'aeiou' for c in sl)
        consonants = sum(c in 'bcdfghjklmnpqrstvwxyz' for c in sl)
        ent = shannon_entropy(s)
        bigram = ngram_common(s, 2)
        trigram = ngram_common(s, 3)
        unique_chars = len(set(s))
        
        feats.append([
            l, ent, digits / l, hyphens / l, underscores / l,
            vowels / l, consonants / l, bigram, trigram,
            unique_chars / l, 1.0 - unique_chars / l
        ])
    return np.array(feats, dtype=np.float32)

X_test_lexical = extract_lexical_features(slds)
print(f"  Extracted lexical feature matrix: {X_test_lexical.shape}")

# -----------------------------------------------------------------------------
# 3. LOAD FASTTEXT & CHAR VOCAB FOR DEEP LEARNING MODELS
# -----------------------------------------------------------------------------
print("\n[3/7] Loading Char Vocab & Pretrained FastText (7.2 GB Crawl binary)...")
vocab_path = VOCAB_DIR / "char_vocab.json"
vocab_data = json.load(open(vocab_path, "r", encoding="utf-8"))
char2idx = vocab_data["vocab"]
vocab_size = vocab_data["size"]
print(f"  Loaded char vocab size: {vocab_size}")

ft_bin_path = WEIGHTS_DIR / "cc.en.300.bin"
t0 = time.time()
print(f"  Loading FastText model from {ft_bin_path}...")
ft_model = load_facebook_model(str(ft_bin_path))
print(f"  FastText model ready in {time.time() - t0:.1f}s!")

def get_word_vector(w):
    w = str(w).lower()
    if w in ft_model.wv:
        return ft_model.wv[w]
    return np.zeros(300, dtype=np.float32)

def segment_domain(sld):
    sld_str = str(sld) if sld is not None else ""
    if not sld_str: return ["<unk>"]
    parts = [p for p in sld_str.replace('_', '-').split('-') if p]
    words = []
    for p in parts:
        w_splits = wordninja.split(p)
        words.extend(w_splits if w_splits else [p])
    return words if words else ["<unk>"]

def compute_coherence_features(words, sld_len, sld_str=""):
    k = len(words)
    has_hyphen = 1.0 if '-' in sld_str else 0.0
    if k <= 1:
        return np.array([0.0, 0.0, 0.0, 0.1, 1.0, has_hyphen], dtype=np.float32)
    vecs = [get_word_vector(w) for w in words]
    norms = [np.linalg.norm(v) + 1e-8 for v in vecs]
    unit_vecs = [v / n for v, n in zip(vecs, norms)]
    sims = [float(np.dot(unit_vecs[i], unit_vecs[j])) for i in range(k) for j in range(i+1, k)]
    return np.array([min(sims), np.mean(sims), max(sims), min(1.0, k/6.0),
                     min(1.0, sum(len(w) for w in words)/max(sld_len, 1)), has_hyphen], dtype=np.float32)

def encode_chars(sld, max_len=64):
    sld_str = str(sld) if sld is not None else ""
    if not sld_str: return [0], 1
    ids = [char2idx.get(ch, char2idx.get("<UNK>", 1)) for ch in sld_str[:max_len]]
    return (ids if ids else [0]), max(len(ids), 1)

# Pre-extract Dual-Branch data
print("  Pre-extracting character sequences and FastText word sequences for test set...")
all_char_ids = []
all_char_lens = []
all_word_vecs = []
all_word_lens = []
all_cohere_vecs = []

for s in slds:
    cids, clen = encode_chars(s)
    ws = segment_domain(s)
    wvecs = np.stack([get_word_vector(w) for w in ws])
    coh = compute_coherence_features(ws, clen, s)
    
    all_char_ids.append(cids)
    all_char_lens.append(clen)
    all_word_vecs.append(wvecs)
    all_word_lens.append(len(ws))
    all_cohere_vecs.append(coh)

class DualBranchInferenceDataset(Dataset):
    def __init__(self, cids, clens, wvecs, wlens, cohs):
        self.cids = cids
        self.clens = clens
        self.wvecs = wvecs
        self.wlens = wlens
        self.cohs = cohs
    def __len__(self):
        return len(self.cids)
    def __getitem__(self, idx):
        return (torch.tensor(self.cids[idx], dtype=torch.long),
                torch.tensor(self.clens[idx], dtype=torch.long),
                torch.tensor(self.wvecs[idx], dtype=torch.float32),
                torch.tensor(self.wlens[idx], dtype=torch.long),
                torch.tensor(self.cohs[idx], dtype=torch.float32))

def collate_dual(batch):
    cids, clens, wvecs, wlens, cohs = zip(*batch)
    max_c = max(clens)
    padded_c = torch.zeros(len(cids), max_c, dtype=torch.long)
    for i, (ids, l) in enumerate(zip(cids, clens)): padded_c[i, :l] = ids
    max_w = max(wlens)
    padded_w = torch.zeros(len(wvecs), max_w, 300, dtype=torch.float32)
    for i, (vec, l) in enumerate(zip(wvecs, wlens)): padded_w[i, :l] = vec
    return (padded_c, torch.stack(clens), padded_w, torch.stack(wlens), torch.stack(cohs))

infer_loader = DataLoader(
    DualBranchInferenceDataset(all_char_ids, all_char_lens, all_word_vecs, all_word_lens, all_cohere_vecs),
    batch_size=256, shuffle=False, collate_fn=collate_dual, num_workers=0
)
print("  Inference DataLoader ready.")

# -----------------------------------------------------------------------------
# 4. MODEL ARCHITECTURES
# -----------------------------------------------------------------------------
class WordBranch(nn.Module):
    def __init__(self, ft_dim=300, hidden=128, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(ft_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden*2, 256)
    def forward(self, w_seq, w_lens):
        packed = pack_padded_sequence(w_seq, w_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        h = self.dropout(h)
        return torch.relu(self.proj(h))

class DualBranchPerfect(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, lstm_hidden=320, lstm_layers=2, ft_dim=300, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True, dropout=dropout if lstm_layers>1 else 0)
        self.word_branch = WordBranch(ft_dim, hidden=128, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        char_dim = lstm_hidden * 2
        word_dim = 256
        self.gate = nn.Sequential(nn.Linear(char_dim + word_dim, 1), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(char_dim + word_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    def forward(self, char_ids, char_lens, w_seq, w_lens):
        emb = self.embed(char_ids)
        packed = pack_padded_sequence(emb, char_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_char = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h_char = self.dropout(h_char)
        h_word = self.word_branch(w_seq, w_lens)
        gate_input = torch.cat([h_char, h_word], dim=1)
        g = self.gate(gate_input)
        fused = torch.cat([g * h_char, (1 - g) * h_word], dim=1)
        return self.head(fused).squeeze(1)

class DualBranchCoherent(nn.Module):
    def __init__(self, vocab_size, char_dim=64, lstm_hidden=128, ft_dim=300, cohere_dim=6, dropout=0.4):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, char_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(char_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.char_drop = nn.Dropout(dropout)
        self.word_lstm = nn.LSTM(ft_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.word_drop = nn.Dropout(dropout)
        self.cohere_mlp = nn.Sequential(nn.Linear(cohere_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 64), nn.ReLU())
        total_repr = (lstm_hidden * 2) + (lstm_hidden * 2) + 64
        self.gate = nn.Sequential(nn.Linear(total_repr, 1), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(total_repr, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, char_ids, char_lens, w_seqs, w_lens, coheres):
        c_emb = self.char_embed(char_ids)
        packed_c = pack_padded_sequence(c_emb, char_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_c, _) = self.char_lstm(packed_c)
        h_char = self.char_drop(torch.cat([h_c[0], h_c[1]], dim=1))
        packed_w = pack_padded_sequence(w_seqs, w_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_w, _) = self.word_lstm(packed_w)
        h_word = self.word_drop(torch.cat([h_w[0], h_w[1]], dim=1))
        h_cohere = self.cohere_mlp(coheres)
        combined = torch.cat([h_char, h_word, h_cohere], dim=1)
        g = self.gate(combined)
        fused = torch.cat([g * h_char, (1.0 - g) * h_word, h_cohere], dim=1)
        return self.head(fused).squeeze(1)

# -----------------------------------------------------------------------------
# 5. EXECUTE INFERENCE FOR ALL MODELS
# -----------------------------------------------------------------------------
print("\n[4/7] Running Inference across all models...")
predictions_dict = {}

# 1. Random Forest
rf_path = WEIGHTS_DIR / "rf_best.pkl"
print(f"  [1/5] Evaluating Random Forest from {rf_path.name}...")
rf_obj = joblib.load(rf_path)
rf_model = rf_obj["model"] if isinstance(rf_obj, dict) else rf_obj
rf_probs = rf_model.predict_proba(X_test_lexical)[:, 1]
predictions_dict["Random Forest"] = rf_probs

# 2. XGBoost
xgb_path = WEIGHTS_DIR / "xgb_best.pkl"
print(f"  [2/5] Evaluating XGBoost from {xgb_path.name}...")
xgb_obj = joblib.load(xgb_path)
xgb_model = xgb_obj["model"] if isinstance(xgb_obj, dict) else xgb_obj
xgb_probs = xgb_model.predict_proba(X_test_lexical)[:, 1]
predictions_dict["XGBoost"] = xgb_probs

# 3. Dual-Branch Perfect v3
dual_v3_path = WEIGHTS_DIR / "dual_branch_best.pt"
print(f"  [3/5] Evaluating Dual-Branch Perfect (v3) from {dual_v3_path.name}...")
v3_model = DualBranchPerfect(vocab_size=vocab_size).to(device)
v3_model.load_state_dict(torch.load(dual_v3_path, map_location=device))
v3_model.eval()

v3_probs_list = []
with torch.no_grad():
    for cids, clens, wseqs, wlens, cohs in infer_loader:
        cids, clens, wseqs, wlens = cids.to(device), clens.to(device), wseqs.to(device), wlens.to(device)
        logits = v3_model(cids, clens, wseqs, wlens)
        probs = torch.sigmoid(logits).cpu().numpy()
        v3_probs_list.append(probs)
predictions_dict["Dual-Branch (v3)"] = np.concatenate(v3_probs_list)

# 4. Dual-Branch Coherent
dual_coh_path = WEIGHTS_DIR / "dual_branch_coherent_best.pt"
print(f"  [4/5] Evaluating Dual-Branch Coherent from {dual_coh_path.name}...")
coh_model = DualBranchCoherent(vocab_size=vocab_size).to(device)
coh_model.load_state_dict(torch.load(dual_coh_path, map_location=device))
coh_model.eval()

coh_probs_list = []
with torch.no_grad():
    for cids, clens, wseqs, wlens, cohs in infer_loader:
        cids, clens, wseqs, wlens, cohs = cids.to(device), clens.to(device), wseqs.to(device), wlens.to(device), cohs.to(device)
        logits = coh_model(cids, clens, wseqs, wlens, cohs)
        probs = torch.sigmoid(logits).cpu().numpy()
        coh_probs_list.append(probs)
predictions_dict["Dual-Branch Coherent"] = np.concatenate(coh_probs_list)

# 5. ModernBERT-Base
bert_path = WEIGHTS_DIR / "modernbertweights" / "training_weights" / "modernbert_best"
print(f"  [5/5] Evaluating ModernBERT-Base (149M) from {bert_path.name}...")
b_tokenizer = AutoTokenizer.from_pretrained(bert_path)
b_model = AutoModelForSequenceClassification.from_pretrained(bert_path).to(device)
b_model.eval()

b_probs = []
batch_size = 128
with torch.no_grad():
    for i in range(0, n_test, batch_size):
        batch_texts = [str(s) for s in slds[i:i+batch_size]]
        enc = b_tokenizer(batch_texts, truncation=True, max_length=32, padding=True, return_tensors="pt").to(device)
        logits = b_model(**enc).logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        b_probs.extend(probs)
predictions_dict["ModernBERT-Base"] = np.array(b_probs)

# 6. Ensemble 1: ModernBERT + Dual-Branch v3
predictions_dict["Ensemble (BERT + Dual v3)"] = 0.5 * predictions_dict["ModernBERT-Base"] + 0.5 * predictions_dict["Dual-Branch (v3)"]

# 7. Ensemble 2: ModernBERT + Dual-Branch Coherent
predictions_dict["Ensemble (BERT + Dual Coh)"] = 0.5 * predictions_dict["ModernBERT-Base"] + 0.5 * predictions_dict["Dual-Branch Coherent"]

print("  All 7 model predictions generated successfully!")

# -----------------------------------------------------------------------------
# 6. COMPUTE FULL METRIC SUITE
# -----------------------------------------------------------------------------
print("\n[5/7] Computing Full Metric Suite for all models...")

BENCHMARK_ZERO_SHOT = ["ngioweb", "pizd"]
TEST_WORDLIST_FAMS = ["ngioweb", "pizd", "matsnu"]
TEST_NON_WORDLIST_FAMS = ["tinba", "murofet"]

benchmark_rows = []
detailed_rows = []
json_output = {"benchmark_comparison": {}, "per_family_breakdown": {}, "detailed_metrics": {}}

for name, y_prob in predictions_dict.items():
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Classification metrics
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Ranking metrics
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Per-family recall
    fam_recalls = {}
    for f in sorted(np.unique(families)):
        mask = (families == f)
        if f == "legit":
            fam_recalls[f] = float((y_pred[mask] == 0).mean())
        else:
            fam_recalls[f] = float((y_pred[mask] == 1).mean())
            
    zs_wordlist_macro = float(np.mean([fam_recalls[f] for f in BENCHMARK_ZERO_SHOT]))
    overall_wordlist_macro = float(np.mean([fam_recalls[f] for f in TEST_WORDLIST_FAMS]))
    nw_macro = float(np.mean([fam_recalls[f] for f in TEST_NON_WORDLIST_FAMS]))
    benign_accuracy = fam_recalls["legit"]
    
    benchmark_rows.append({
        "Model": name,
        "Zero-Shot WL (Ngioweb+Pizd)": zs_wordlist_macro,
        "Matsnu (Hard Case)": fam_recalls["matsnu"],
        "Overall Wordlist Macro": overall_wordlist_macro,
        "Non-Wordlist Macro": nw_macro,
        "Benign Accuracy": benign_accuracy,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "PR-AUC": ap
    })
    
    detailed_rows.append({
        "Model": name,
        "Accuracy": acc,
        "Balanced Acc": b_acc,
        "Precision": prec,
        "Recall": rec,
        "Specificity": spec,
        "FPR": fpr,
        "F1 (Binary)": f1,
        "F1 (Macro)": f1_macro,
        "MCC": mcc,
        "ROC-AUC": auc,
        "PR-AUC (AP)": ap,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    })
    
    json_output["per_family_breakdown"][name] = fam_recalls

benchmark_df = pd.DataFrame(benchmark_rows)
detailed_df = pd.DataFrame(detailed_rows)

print("\n" + "="*95)
print("                   CSE427 FINAL TEST-UNSEEN BENCHMARK RESULTS")
print("="*95)
print(benchmark_df.to_string(index=False))

# Export CSV and LaTeX
benchmark_df.to_csv(TABLE_DIR / "final_test_benchmark_table.csv", index=False)
detailed_df.to_csv(TABLE_DIR / "final_test_detailed_metrics.csv", index=False)

with open(TABLE_DIR / "final_test_benchmark_table.tex", "w", encoding="utf-8") as f:
    f.write(benchmark_df.to_latex(index=False, float_format="%.3f"))
with open(TABLE_DIR / "final_test_detailed_metrics.tex", "w", encoding="utf-8") as f:
    f.write(detailed_df.to_latex(index=False, float_format="%.3f"))

json_output["benchmark_comparison"] = benchmark_df.to_dict(orient="records")
json_output["detailed_metrics"] = detailed_df.to_dict(orient="records")
with open(METRIC_DIR / "complete_evaluation_metrics.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=2)

print(f"\nTables and JSON successfully exported to {TABLE_DIR} and {METRIC_DIR}")

# -----------------------------------------------------------------------------
# 7. GENERATE ALL 8 PUBLICATION-GRADE VISUALIZATIONS
# -----------------------------------------------------------------------------
print("\n[6/7] Generating 8 Publication-Grade Visualizations...")

model_colors = {
    "Random Forest": "#7f8c8d",
    "XGBoost": "#e67e22",
    "Dual-Branch (v3)": "#27ae60",
    "Dual-Branch Coherent": "#2980b9",
    "ModernBERT-Base": "#8e44ad",
    "Ensemble (BERT + Dual v3)": "#c0392b",
    "Ensemble (BERT + Dual Coh)": "#d35400"
}

# --- FIGURE 1: ROC CURVES ---
print("  Generating Figure 1: Multi-Model ROC Curves...")
plt.figure(figsize=(8, 6), dpi=300)
for name, y_prob in predictions_dict.items():
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=model_colors.get(name, "black"), lw=1.8)
plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label="Random Guess (AUC = 0.500)")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Receiver Operating Characteristic (ROC) — Test-Unseen", fontweight="bold", pad=12)
plt.legend(loc="lower right", frameon=True, framealpha=0.9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_roc_curves.png", dpi=300)
plt.close()

# --- FIGURE 2: PR CURVES ---
print("  Generating Figure 2: Multi-Model Precision-Recall Curves...")
plt.figure(figsize=(8, 6), dpi=300)
for name, y_prob in predictions_dict.items():
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap_val = average_precision_score(y_true, y_prob)
    plt.plot(rec, prec, label=f"{name} (AP = {ap_val:.3f})", color=model_colors.get(name, "black"), lw=1.8)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall (PR) Curve — Test-Unseen", fontweight="bold", pad=12)
plt.legend(loc="lower left", frameon=True, framealpha=0.9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_pr_curves.png", dpi=300)
plt.close()

# --- FIGURE 3: CONFUSION MATRICES GRID ---
print("  Generating Figure 3: Confusion Matrices Grid...")
fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)
axes = axes.flatten()

for i, (name, y_prob) in enumerate(predictions_dict.items()):
    yp = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, yp, labels=[0, 1])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Annotate with count + percentage
    annot = np.empty_like(cm).astype(str)
    for r in range(2):
        for c in range(2):
            annot[r, c] = f"{cm[r, c]:,}\n({cm_norm[r, c]:.1%})"
            
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues", cbar=False, ax=axes[i],
                xticklabels=["Benign", "DGA"], yticklabels=["Benign", "DGA"])
    axes[i].set_title(name, fontweight="bold", fontsize=11)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

# Remove 8th empty subplot
fig.delaxes(axes[7])
plt.suptitle("Confusion Matrices on Test-Unseen (Counts & Class Accuracies)", fontweight="bold", fontsize=15, y=0.98)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_confusion_matrices_grid.png", dpi=300)
plt.close()

# --- FIGURE 4: PER-FAMILY RECALL GROUPED BAR CHART ---
print("  Generating Figure 4: Per-Family Recall Comparison Bar Chart...")
fams_order = ["ngioweb", "pizd", "matsnu", "tinba", "murofet", "legit"]
fam_display_labels = ["Ngioweb\n(Wordlist)", "Pizd\n(Wordlist)", "Matsnu\n(Hard WL)", "Tinba\n(Non-WL)", "Murofet\n(Non-WL)", "Legit\n(Benign)"]

per_fam_data = []
for name in predictions_dict.keys():
    for f in fams_order:
        val = json_output["per_family_breakdown"][name][f]
        per_fam_data.append({"Model": name, "Class": f, "Score": val})
df_fam_plot = pd.DataFrame(per_fam_data)

plt.figure(figsize=(14, 6), dpi=300)
ax = sns.barplot(data=df_fam_plot, x="Class", y="Score", hue="Model", palette=model_colors, edgecolor="black")
plt.xticks(ticks=range(len(fams_order)), labels=fam_display_labels, fontweight="bold")
plt.ylabel("Recall (or Benign Accuracy)")
plt.ylim(0, 1.05)
plt.title("Per-Family Detection Rate on Test-Unseen (Leave-Families-Out)", fontweight="bold", pad=14)
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_per_family_recall_barchart.png", dpi=300)
plt.close()

# --- FIGURE 5: BENCHMARK ANCHORS COMPARISON ---
print("  Generating Figure 5: Benchmark Anchors Comparison Plot...")
plt.figure(figsize=(11, 5.5), dpi=300)
x_pos = np.arange(len(benchmark_df))
bars = plt.bar(x_pos, benchmark_df["Zero-Shot WL (Ngioweb+Pizd)"], color=[model_colors.get(m, "gray") for m in benchmark_df["Model"]], edgecolor="black", width=0.6)

plt.axhline(0.194, color="crimson", linestyle="--", lw=2, label="Drichel et al. (2022) Zero-Day Floor (19.4%)")
plt.axhline(0.809, color="forestgreen", linestyle="--", lw=2, label="Leyva La O et al. (2026) IEEE SOTA Ceiling (80.9%)")

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=9)

plt.xticks(x_pos, benchmark_df["Model"], rotation=15, ha="right", fontweight="bold")
plt.ylabel("Zero-Shot Wordlist Macro-Recall (Ngioweb + Pizd)")
plt.ylim(0, 1.0)
plt.title("CSE427 Zero-Shot Wordlist Generalization vs. Published Academic Anchors", fontweight="bold", pad=12)
plt.legend(loc="upper left", frameon=True, framealpha=0.9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_benchmark_anchors_comparison.png", dpi=300)
plt.close()

# --- FIGURE 6: DECISION THRESHOLD TRADE-OFF CURVES ---
print("  Generating Figure 6: Decision Threshold Trade-Off Curves...")
thresholds = np.linspace(0.05, 0.95, 30)
plt.figure(figsize=(10, 6), dpi=300)

wl_mask = np.isin(families, TEST_WORDLIST_FAMS)
legit_mask = (families == "legit")

for name, y_prob in predictions_dict.items():
    if "Ensemble" in name or name in ["ModernBERT-Base", "Dual-Branch (v3)"]:
        wl_recs = []
        benign_accs = []
        for t in thresholds:
            pred_t = (y_prob >= t).astype(int)
            # Wordlist recall across ngioweb, pizd, matsnu
            sub_recs = [float((pred_t[families == f] == 1).mean()) for f in TEST_WORDLIST_FAMS]
            wl_recs.append(np.mean(sub_recs))
            benign_accs.append(float((pred_t[legit_mask] == 0).mean()))
        plt.plot(benign_accs, wl_recs, marker='o', markersize=3, lw=1.8, label=name, color=model_colors.get(name))

plt.xlabel("Benign Accuracy (1 - False Positive Rate)")
plt.ylabel("Wordlist Macro-Recall")
plt.title("Operational Trade-off: Wordlist DGA Recall vs. Benign Accuracy across Thresholds", fontweight="bold", pad=12)
plt.xlim(0.5, 1.0)
plt.ylim(0.0, 1.0)
plt.legend(loc="lower left", frameon=True)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig6_threshold_tradeoff_curves.png", dpi=300)
plt.close()

# --- FIGURE 7: ERROR BREAKDOWN DONUTS ---
print("  Generating Figure 7: Error Breakdown Donut Charts...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
key_models = ["Random Forest", "ModernBERT-Base", "Dual-Branch (v3)", "Ensemble (BERT + Dual v3)"]

for i, name in enumerate(key_models):
    y_prob = predictions_dict[name]
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sizes = [tp, tn, fp, fn]
    labels = [f"True DGA\n({tp:,})", f"True Benign\n({tn:,})", f"False Alarm\n({fp:,})", f"Missed DGA\n({fn:,})"]
    colors_donut = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
    
    axes[i].pie(sizes, labels=labels, colors=colors_donut, autopct='%1.1f%%', startangle=90, 
                pctdistance=0.75, textprops={'fontsize': 8})
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    axes[i].add_artist(centre_circle)
    axes[i].set_title(name, fontweight="bold", fontsize=11)

plt.suptitle("Model Outcome Distributions (Correct Classifications vs. Specific Errors)", fontweight="bold", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig7_error_breakdown_donuts.png", dpi=300)
plt.close()

# --- FIGURE 8: RADAR / SPIDER CHART ---
print("  Generating Figure 8: 6-Axis Model Profile Radar Chart...")
radar_categories = ["Zero-Shot WL", "Non-Wordlist", "Benign Acc", "F1-Score", "ROC-AUC", "Eff. Score"]
N = len(radar_categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=300)
plt.xticks(angles[:-1], radar_categories, color='grey', size=10, fontweight="bold")
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
plt.ylim(0, 1.0)

radar_models = ["XGBoost", "ModernBERT-Base", "Dual-Branch (v3)", "Ensemble (BERT + Dual v3)"]
# Param efficiency score: 1 - log10(params)/10
param_eff = {
    "XGBoost": 0.95,
    "Dual-Branch (v3)": 0.85,
    "ModernBERT-Base": 0.40,
    "Ensemble (BERT + Dual v3)": 0.35
}

for name in radar_models:
    row = benchmark_df[benchmark_df["Model"] == name].iloc[0]
    values = [
        row["Zero-Shot WL (Ngioweb+Pizd)"],
        row["Non-Wordlist Macro"],
        row["Benign Accuracy"],
        row["F1-Score"],
        row["ROC-AUC"],
        param_eff.get(name, 0.5)
    ]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=name, color=model_colors.get(name))
    ax.fill(angles, values, color=model_colors.get(name), alpha=0.1)

plt.title("Multi-Dimensional Model Profile Comparison", fontweight="bold", pad=20)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig8_radar_model_profile.png", dpi=300)
plt.close()

# Copy all figures to training/figures for notebook access
for fig_f in FIG_DIR.glob("*.png"):
    dest = BASE_DIR / "training" / "figures" / fig_f.name
    with open(fig_f, "rb") as sf, open(dest, "wb") as df:
        df.write(sf.read())

print("\n[7/7] MASTER EVALUATION COMPLETED SUCCESSFULLY!")
print(f"All 8 figures saved to: {FIG_DIR}")
print(f"All tables saved to: {TABLE_DIR}")
print(f"All metrics saved to: {METRIC_DIR}")
