#!/usr/bin/env python3
"""
RF & XGBoost Baselines for DGA Killer — Lexical Features
- Master Plan §5 baseline #1 (hand-crafted: entropy, n-gram, length, digit/vowel/hyphen)
- Train on TRAIN (32 fam), validate on VAL-UNSEEN (6 new fam), save best on val wordlist macro-F1
- RTX 3050: RF CPU, XGBoost gpu_hist (CUDA) — fits 8GB easily, domains are short
- Outputs: best_weights/*.pkl, metrics.json, detailed plots (confusion, ROC, PR, per-family F1, feature importance/SHAP)

Usage:
  python scripts/train_rf_xgboost.py --model rf          # RandomForest
  python scripts/train_rf_xgboost.py --model xgboost     # XGBoost (GPU if available)
  python scripts/train_rf_xgboost.py --model both        # both sequentially
  python scripts/train_rf_xgboost.py --model xgboost --use-gpu

Reference: senior-data-scientist skill — build_feature_pipeline, evaluate_model, SHAP
"""
import argparse, json, pathlib, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi":150, "savefig.dpi":300})

BASE = Path(r"G:\dga_killer")
DATA_PROC = BASE / "data" / "processed"
RESULTS = BASE / "results" / "baselines"
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------- Lexical feature helpers (same as EDA) ----------
def shannon(s: str) -> float:
    if not s:
        return 0.0
    cnt = Counter(str(s))
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in cnt.values())

def ngram_freq(s: str, n: int = 2) -> float:
    """Mean frequency of char n-grams vs English? Simplified: count of common bigrams (th, he, in, er) / len"""
    s = str(s).lower()
    if len(s) < n:
        return 0.0
    common = {"th","he","in","er","an","re","on","at","en","nd"}
    grams = [s[i:i+n] for i in range(len(s)-n+1)]
    if not grams:
        return 0.0
    return sum(1 for g in grams if g in common) / len(grams)

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """From SLD strings to numeric matrix. Uses SLD only (TLD already stripped)."""
    s = df["sld"].fillna("").astype(str)
    s_lower = s.str.lower()
    feats = pd.DataFrame({
        "sld_len": s.str.len(),
        "entropy": s.apply(shannon),
        "digit_ratio": s.apply(lambda x: sum(c.isdigit() for c in x)/len(x) if x else 0),
        "hyphen_ratio": s.str.count("-") / s.str.len().replace(0, 1),
        "underscore_ratio": s.str.count("_") / s.str.len().replace(0, 1),
        "vowel_ratio": s_lower.apply(lambda x: sum(c in "aeiou" for c in x)/len(x) if x else 0),
        "consonant_ratio": s_lower.apply(lambda x: sum(c in "bcdfghjklmnpqrstvwxyz" for c in x)/len(x) if x else 0),
        "bigram_common_ratio": s.apply(lambda x: ngram_freq(x, 2)),
        "trigram_common_ratio": s.apply(lambda x: ngram_freq(x, 3)),
        "unique_char_ratio": s.apply(lambda x: len(set(x))/len(x) if x else 0),
        "repeated_char_ratio": s.apply(lambda x: 1 - len(set(x))/len(x) if x else 0),
    })
    # NaN guard
    feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
    return feats

FEATURE_COLS = ["sld_len","entropy","digit_ratio","hyphen_ratio","underscore_ratio","vowel_ratio","consonant_ratio","bigram_common_ratio","trigram_common_ratio","unique_char_ratio","repeated_char_ratio"]

# ---------- Evaluation + plotting ----------
def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["benign","dga"], yticklabels=["benign","dga"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(title, fontweight="bold")
    # annotate rates
    tn, fp, fn, tp = cm.ravel()
    plt.text(1.5, -0.3, f"Acc { (tp+tn)/cm.sum():.3f} | F1 {f1_score(y_true,y_pred):.3f}", ha="center", fontsize=8, transform=plt.gca().transData)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_roc_pr(y_true, y_proba, title_prefix, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    axes[0].plot(fpr, tpr, color="#e74c3c", label=f"AUC={auc:.3f}")
    axes[0].plot([0,1],[0,1], color="gray", linestyle="--")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title(f"{title_prefix} ROC", fontweight="bold"); axes[0].legend()
    axes[1].plot(rec, prec, color="#3498db", label=f"AP={ap:.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title(f"{title_prefix} PR", fontweight="bold"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return {"auc": float(auc), "ap": float(ap)}

def per_family_table(df_val, y_true, y_pred):
    """F1 per family (for DGA families) + benign as group. df_val is val slice with family column."""
    rows = []
    for fam in sorted(df_val["family"].unique()):
        mask = df_val["family"] == fam
        yt = y_true[mask]
        yp = y_pred[mask]
        # For legit, positive is benign (0), but we measure detection of that family vs legit?
        # Simpler: compute F1 for that family as binary (if fam==legit, treat benign as positive)
        # For DGA families, y_true is 1 for DGA rows of that fam, 0 for legit rows? No, legit rows have different fam, so per-family we isolate.
        # So y_true here is all 1 for DGA families, all 0 for legit — accuracy is just correct rate for that family.
        # We'll report recall for DGA families (how many of that family's domains were caught) and precision via pooled.
        # For simplicity, report recall and F1 with respect to pooled val.
        # For per-family F1, we need to consider that family's DGA vs all benign in val.
        # We'll compute recall for DGA families, and for legit we compute specificity.
        if fam == "legit":
            # benign: true label 0, predicted 0 is correct
            acc = (yt == yp).mean() if len(yt) else 0
            rows.append((fam, len(yt), float(acc), None, None))
        else:
            # DGA family: yt all 1, so recall = mean(yp==1), precision requires pooled but we approximate via overall
            recall = (yp == 1).mean() if len(yp) else 0
            # F1 for that family vs all benign: we can compute F1 on subset that includes this family's DGA + all benign val
            # To avoid per-family precision noise, we report recall as main metric (per Master Plan)
            rows.append((fam, len(yt), float(recall), None, None))
    return rows

def feature_importance_report(model, X, y, feature_names, save_prefix):
    """RF impurity + permutation, plus SHAP if available. Returns dict."""
    # Impurity
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        order = np.argsort(imp)[::-1]
        plt.figure(figsize=(8,4))
        plt.barh([feature_names[i] for i in order], imp[order], color="#2980b9", edgecolor="black", linewidth=0.4)
        plt.xlabel("Impurity importance")
        plt.title("Feature Importance (RF impurity)", fontweight="bold")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_impurity.png", bbox_inches="tight")
        plt.close()
    # Permutation (more reliable)
    try:
        perm = permutation_importance(model, X, y, n_repeats=10, random_state=SEED, n_jobs=-1)
        perm_imp = perm.importances_mean
        order = np.argsort(perm_imp)[::-1]
        plt.figure(figsize=(8,4))
        plt.barh([feature_names[i] for i in order], perm_imp[order], color="#27ae60", edgecolor="black", linewidth=0.4)
        plt.xlabel("Permutation importance (drop in accuracy)")
        plt.title("Feature Importance (permutation, val)", fontweight="bold")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_permutation.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Permutation failed: {e}")
    # SHAP (optional, for RF)
    shap_ok = False
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X[:500])  # sample 500 for speed
        # shap for binary RF returns list [class0, class1]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        plt.figure()
        shap.summary_plot(shap_vals, X[:500], feature_names=feature_names, show=False, plot_size=(8,4))
        plt.title("SHAP Summary (RF, sample 500)")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_shap.png", bbox_inches="tight")
        plt.close()
        shap_ok = True
    except Exception as e:
        print(f"SHAP skipped: {e}")
    # Correlation heatmap (requested)
    corr = pd.DataFrame(X, columns=feature_names).corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, cbar_kws={"label":"Pearson"})
    plt.title("Lexical Feature Correlation", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_correlation.png", bbox_inches="tight")
    plt.close()
    return {"shap": shap_ok}

# ---------- Main train ----------
def train_one(model_name: str, use_gpu: bool = False):
    print(f"\n{'='*60}\nTraining {model_name} {'(GPU)' if use_gpu else ''}\n{'='*60}")
    # Load
    train = pd.read_csv(DATA_PROC / "train.csv")
    val = pd.read_csv(DATA_PROC / "val_unseen.csv")
    # For RF/XGBoost we train on TRAIN only, validate on VAL-UNSEEN (no train val split)
    X_train = extract_features(train)
    y_train = train["label_binary"].values
    X_val = extract_features(val)
    y_val = val["label_binary"].values
    print(f"Train {X_train.shape} (benign {(y_train==0).sum():,} / dga {(y_train==1).sum():,})")
    print(f"Val   {X_val.shape} (benign {(y_val==0).sum():,} / dga {(y_val==1).sum():,})")
    print(f"Features: {FEATURE_COLS}")
    print(f"Feature means train:\n{X_train.mean().round(3).to_dict()}")
    # Correlation heatmap on train (pre-training)
    # Scale for XGBoost? Not needed for trees, but we log it
    scaler = StandardScaler()
    # Fit scaler on train for potential later use (RF doesn't need, but we save)
    scaler.fit(X_train)

    # Model
    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=2,
            n_jobs=-1, random_state=SEED, class_weight="balanced", verbose=0
        )
    elif model_name == "xgboost":
        try:
            import xgboost as xgb
            if use_gpu:
                # New XGBoost uses device param
                try:
                    model = xgb.XGBClassifier(
                        n_estimators=400, max_depth=8, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9,
                        eval_metric="logloss", random_state=SEED,
                        tree_method="hist", device="cuda", verbosity=1
                    )
                    print("XGBoost will use CUDA (device='cuda')")
                except:
                    model = xgb.XGBClassifier(
                        n_estimators=400, max_depth=8, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9,
                        eval_metric="logloss", random_state=SEED,
                        tree_method="gpu_hist", predictor="gpu_predictor", verbosity=1
                    )
                    print("XGBoost fallback to gpu_hist")
            else:
                model = xgb.XGBClassifier(
                    n_estimators=400, max_depth=8, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9,
                    eval_metric="logloss", random_state=SEED,
                    tree_method="hist", n_jobs=-1, verbosity=1
                )
        except ImportError:
            print("xgboost not installed, pip install xgboost")
            return
    else:
        raise ValueError(model_name)

    # Fit
    print(f"Fitting {model_name}...")
    model.fit(X_train, y_train)
    # Predict
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:,1] if hasattr(model, "predict_proba") else np.zeros_like(y_train, dtype=float)
    y_pred_val = model.predict(X_val)
    y_proba_val = model.predict_proba(X_val)[:,1] if hasattr(model, "predict_proba") else np.zeros_like(y_val, dtype=float)

    # Metrics — overall + per split
    for split_name, yt, yp, ypr in [("train", y_train, y_pred_train, y_proba_train), ("val_unseen", y_val, y_pred_val, y_proba_val)]:
        print(f"\n--- {split_name} ---")
        print(classification_report(yt, yp, target_names=["benign","dga"], digits=3))
        print(f"Acc {accuracy_score(yt, yp):.4f} | F1 {f1_score(yt, yp):.4f} | AUC {roc_auc_score(yt, ypr):.4f} | AP {average_precision_score(yt, ypr):.4f}")

    # Per-family recall on val (key for wordlist generalization)
    print("\n--- Per-family val recall (wordlist vs non-wordlist) ---")
    val_with_pred = val.copy()
    val_with_pred["y_true"] = y_val
    val_with_pred["y_pred"] = y_pred_val
    for fam in sorted(val["family"].unique()):
        mask = val["family"] == fam
        yt = y_val[mask]
        yp = y_pred_val[mask]
        if fam == "legit":
            acc = (yt == yp).mean()
            print(f"{fam:15s} n={len(yt):4d}  benign-accuracy {acc:.3f}")
        else:
            rec = (yp == 1).mean()  # yt all 1 for DGA families
            print(f"{fam:15s} n={len(yt):4d}  wordlist={val.loc[mask,'wordlist'].iloc[0] if len(val.loc[mask]) else '?'}  recall {rec:.3f}")

    # Val wordlist macro-F1 (checkpoint criterion per Master Plan §6)
    val_wordlist_mask = val["wordlist"] & (val["label_binary"]==1)
    # For wordlist families, we need macro across families, but simpler: F1 on wordlist subset vs benign?
    # We'll compute wordlist macro as mean recall across wordlist families (suppobox, bigviktor, manuelita)
    wordlist_fams = ["suppobox","bigviktor","manuelita"]
    recalls = []
    for fam in wordlist_fams:
        mask = val["family"] == fam
        if mask.sum():
            recalls.append((y_pred_val[mask]==1).mean())
    wordlist_macro_recall = float(np.mean(recalls)) if recalls else 0.0
    # Also pooled F1 on val
    val_f1 = f1_score(y_val, y_pred_val)
    print(f"\nCheckpoint criterion — val wordlist macro-recall {wordlist_macro_recall:.4f} (mean of {wordlist_fams}) | val pooled F1 {val_f1:.4f}")

    # Plots
    out_dir = RESULTS / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(y_train, y_pred_train, f"{model_name} — Confusion Train", out_dir / "confusion_train.png")
    plot_confusion(y_val, y_pred_val, f"{model_name} — Confusion Val-Unseen", out_dir / "confusion_val.png")
    plot_roc_pr(y_train, y_proba_train, f"{model_name} Train", out_dir / "roc_pr_train.png")
    plot_roc_pr(y_val, y_proba_val, f"{model_name} Val-Unseen", out_dir / "roc_pr_val.png")

    # Per-family F1 bar on val
    fams = sorted([f for f in val["family"].unique() if f != "legit"])
    recs = [(y_pred_val[val["family"]==f]==1).mean() for f in fams]
    is_wl = [val[val["family"]==f]["wordlist"].iloc[0] for f in fams]
    colors = ["#3498db" if wl else "#e67e22" for wl in is_wl]
    plt.figure(figsize=(10,4))
    plt.bar(fams, recs, color=colors, edgecolor="black", linewidth=0.4)
    plt.axhline(0.5, color="gray", linestyle="--")
    plt.title(f"{model_name} — Val-Unseen Recall per Family (blue=wordlist)", fontweight="bold")
    plt.ylabel("Recall (DGA caught)")
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(recs):
        plt.text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=7)
    plt.ylim(0,1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "per_family_recall_val.png", bbox_inches="tight")
    plt.close()

    # Feature importance + correlation
    feat_info = feature_importance_report(model, X_val, y_val, FEATURE_COLS, str(out_dir / "feature"))
    # Also correlation on train
    corr_train = X_train.corr()
    # Save metrics
    metrics = {
        "model": model_name,
        "train": {"acc": float(accuracy_score(y_train, y_pred_train)), "f1": float(f1_score(y_train, y_pred_train)), "auc": float(roc_auc_score(y_train, y_proba_train)), "ap": float(average_precision_score(y_train, y_proba_train))},
        "val": {"acc": float(accuracy_score(y_val, y_pred_val)), "f1": float(f1_score(y_val, y_pred_val)), "auc": float(roc_auc_score(y_val, y_proba_val)), "ap": float(average_precision_score(y_val, y_proba_val)), "wordlist_macro_recall": wordlist_macro_recall},
        "per_family_val_recall": {f: float((y_pred_val[val["family"]==f]==1).mean()) for f in sorted(val["family"].unique()) if f!="legit"},
        "feature_cols": FEATURE_COLS,
        "use_gpu": use_gpu,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    # Save model (best = this single fit, since we validate on unseen families, no further tuning — matches Master Plan)
    joblib.dump({"model": model, "scaler": scaler, "feature_cols": FEATURE_COLS}, out_dir / "best_weights.pkl")
    print(f"\nSaved {model_name} best to {out_dir / 'best_weights.pkl'} and metrics to {out_dir / 'metrics.json'}")
    print(f"Figures in {out_dir}/")
    return metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rf","xgboost","both"], default="both")
    ap.add_argument("--use-gpu", action="store_true", help="force XGBoost CUDA")
    args = ap.parse_args()
    # Auto-detect GPU for xgboost if not forced (torch optional)
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        print(f"RTX 3050 CUDA available: {has_cuda} (torch {torch.__version__})")
    except ImportError:
        has_cuda = False
        print(" torch not installed — XGBoost will try GPU via tree_method='gpu_hist' if --use-gpu is set")
    if args.model in ("rf","both"):
        train_one("rf", use_gpu=False)
    if args.model in ("xgboost","both"):
        train_one("xgboost", use_gpu=args.use_gpu or has_cuda)
    print("\nAll baselines done. Next: python scripts/train_rf_xgboost.py --model both --use-gpu  (RTX 3050 will use gpu_hist)")
