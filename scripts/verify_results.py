#!/usr/bin/env python3
"""
Verify that every number published in README.md is consistent with the
committed evaluation artifacts (reports/metrics/complete_evaluation_metrics.json).

Checks:
  1. The three README result tables (master, per-family, diagnostics) against
     the metrics JSON, cell by cell.
  2. Every derived headline claim (204x, +21.4% rel, +24.4% rel, 88.6%,
     recall-per-million-parameter table, ensemble operating point).
  3. Internal consistency of the confusion counts (TP+FP+TN+FN == 19,998).

Run from the repository root:  python scripts/verify_results.py
Exits non-zero on any mismatch. No dataset, weights, or GPU required.
"""
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METRICS = REPO / "reports" / "metrics" / "complete_evaluation_metrics.json"
README = REPO / "README.md"

N_TEST = 19998
TOL = 5e-3  # half a displayed rounding unit

failures = []
checks = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        failures.append(f"{name}: {detail}")


def close(a: float, b: float, tol: float = TOL) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


# README model label -> JSON model label
ALIAS = {
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
    "Dual-Branch v3": "Dual-Branch (v3)",
    "Dual-Branch Coherent": "Dual-Branch Coherent",
    "ModernBERT-Base": "ModernBERT-Base",
    "Ensemble 1 (BERT + v3)": "Ensemble (BERT + Dual v3)",
    "Ensemble 2 (BERT + Coh)": "Ensemble (BERT + Dual Coh)",
}


def parse_table(block: str) -> list[list[str]]:
    """Extract markdown table rows (split on |) from a section block."""
    rows = []
    for line in block.splitlines():
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and not set(cells[0]) <= set("- :"):  # skip separator rows
                rows.append(cells)
    return rows


def numeric(cell: str) -> float:
    return float(re.sub(r"[^0-9.\-]", "", cell))


def verify_markdown_table(readme: str, json_rows: list[dict], section: str,
                          json_keys: list[str], is_percent: list[bool],
                          row_name_re: str) -> None:
    print(f"\n{section}")
    pattern = re.escape(section) + r"(.*?)(?=^#{2,3} |\Z)"
    block = re.search(pattern, readme, re.S | re.M)
    rows = parse_table(block.group(0) if block else "")
    table = [r for r in rows if re.search(row_name_re, r[0])]
    check(f"{section}: all 7 model rows present", len(table) == 7, f"found {len(table)}")
    if len(table) != 7:
        return
    by_json = {r["Model"]: r for r in json_rows}
    for cells in table:
        m = re.search(row_name_re, cells[0])
        if m is None:  # header or non-model row
            continue
        readme_name = m.group(1).strip()
        json_name = ALIAS.get(readme_name)
        if json_name is None:
            check(f"{section}: '{readme_name}' known", False, "no JSON alias")
            continue
        row = by_json[json_name]
        ok_all, bad = True, []
        for i, key in enumerate(json_keys):
            col = cells[i + 1]
            if is_percent[i]:
                ok = close(numeric(col), row[key] * 100, tol=0.05)
            else:
                ok = close(numeric(col), row[key], tol=0.005)
            if not ok:
                ok_all = False
                bad.append(f"{key}: README={col} JSON={row[key]:.6f}")
        check(f"{section}: {readme_name}", ok_all, "; ".join(bad))


def main() -> int:
    if not METRICS.exists():
        print(f"metrics file missing: {METRICS}")
        return 2
    if not README.exists():
        print(f"README missing: {README}")
        return 2
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    readme = README.read_text(encoding="utf-8")

    bench = data["benchmark_comparison"]
    per_family = data["per_family_breakdown"]
    detailed = data["detailed_metrics"]

    print("== README result tables vs reports/metrics/complete_evaluation_metrics.json ==")

    # 1. Master benchmark table
    verify_markdown_table(
        readme, bench, "### Master table",
        ["Zero-Shot WL (Ngioweb+Pizd)", "Matsnu (Hard Case)", "Overall Wordlist Macro",
         "Non-Wordlist Macro", "Benign Accuracy", "F1-Score", "ROC-AUC", "PR-AUC"],
        [True, True, True, True, True, False, False, False],
        r"\*\*(.+?)\*\*",
    )

    # 2. Per-family table
    fam_keys = ["ngioweb", "pizd", "matsnu", "tinba", "murofet", "legit"]
    fam_rows = [{"Model": m, **{k: v for k, v in vals.items()}} for m, vals in per_family.items()]
    verify_markdown_table(
        readme, fam_rows, "### Per-family recall / benign specificity",
        fam_keys, [True] * 6, r"\*\*(.+?)\*\*",
    )

    # 3. Diagnostics table
    det_keys = ["Accuracy", "Precision", "Recall", "Specificity", "FPR",
                "F1 (Binary)", "MCC", "TP", "FP", "TN", "FN"]
    verify_markdown_table(
        readme, detailed, "### Full diagnostics",
        det_keys, [True] * 5 + [False] * 6, r"\*\*(.+?)\*\*",
    )

    # 4. Confusion-count internal consistency + cross-section agreement
    print("\n== Internal consistency ==")
    bench_by = {r["Model"]: r for r in bench}
    for row in detailed:
        total = row["TP"] + row["FP"] + row["TN"] + row["FN"]
        check(f"confusion counts sum to {N_TEST}: {row['Model']}", total == N_TEST, f"got {total}")
        b = bench_by.get(row["Model"])
        if b:
            check(f"benign specificity agrees across tables: {row['Model']}",
                  close(b["Benign Accuracy"], row["Specificity"], tol=0.01),
                  f"bench={b['Benign Accuracy']:.6f} detailed={row['Specificity']:.6f}")

    # 5. Derived headline claims
    print("\n== Derived headline claims ==")
    b = {r["Model"]: r for r in bench}
    coh = b["Dual-Branch Coherent"]; v3 = b["Dual-Branch (v3)"]; bert = b["ModernBERT-Base"]
    ens2 = b["Ensemble (BERT + Dual Coh)"]

    param_ratio = 149.0 / 0.73
    check("204x parameter ratio (149.0M / 0.73M)", close(param_ratio, 204.1, tol=0.15),
          f"computed {param_ratio:.2f}")

    rel_wl = (coh["Overall Wordlist Macro"] - v3["Overall Wordlist Macro"]) / v3["Overall Wordlist Macro"]
    check("coherence ablation +21.4% relative WL macro (Coherent vs v3)",
          close(rel_wl * 100, 21.4, tol=0.1), f"computed {rel_wl*100:.2f}%")
    check("ablation +5.46 absolute WL macro points",
          close((coh["Overall Wordlist Macro"] - v3["Overall Wordlist Macro"]) * 100, 5.46, tol=0.05))

    rel_matsnu = (coh["Matsnu (Hard Case)"] - bert["Matsnu (Hard Case)"]) / bert["Matsnu (Hard Case)"]
    check("matsnu inversion +24.4% relative (Coherent vs ModernBERT)",
          close(rel_matsnu * 100, 24.4, tol=0.1), f"computed {rel_matsnu*100:.2f}%")
    check("matsnu detections 352 vs 283 of 2,000",
          close(coh["Matsnu (Hard Case)"] * 2000, 352, tol=0.5)
          and close(bert["Matsnu (Hard Case)"] * 2000, 283, tol=0.5))

    frac = coh["Overall Wordlist Macro"] / bert["Overall Wordlist Macro"]
    check("Coherent reaches 88.6% of ModernBERT WL macro",
          close(frac * 100, 88.6, tol=0.1), f"computed {frac*100:.2f}%")

    print("  -- efficiency table (recall per million parameters) --")
    for name, params, expected in [("Dual-Branch Coherent", 0.73, 42.51),
                                   ("Dual-Branch (v3)", 4.10, 6.24),
                                   ("ModernBERT-Base", 149.0, 0.24)]:
        rpm = b[name]["Overall Wordlist Macro"] * 100 / params
        check(f"{name}: {expected} recall/M-param", close(rpm, expected, tol=0.05),
              f"computed {rpm:.3f}")

    print("  -- ensemble operating point (Ensemble 2) --")
    check("Ensemble 2 non-wordlist macro = 100.00%", close(ens2["Non-Wordlist Macro"] * 100, 100.0, tol=0.01))
    check("Ensemble 2 benign specificity = 94.88%", close(ens2["Benign Accuracy"] * 100, 94.88, tol=0.05))
    check("Ensemble 2 WL macro = 34.25%", close(ens2["Overall Wordlist Macro"] * 100, 34.25, tol=0.05))
    check("Ensemble 2 ROC-AUC = 0.8085", close(ens2["ROC-AUC"], 0.8085, tol=0.0005))

    d = {r["Model"]: r for r in detailed}
    print("  -- SOC false-positive counts --")
    check("ModernBERT 444 FP", d["ModernBERT-Base"]["FP"] == 444)
    check("Coherent 1,377 FP", d["Dual-Branch Coherent"]["FP"] == 1377)
    check("Ensemble 2 512 FP", d["Ensemble (BERT + Dual Coh)"]["FP"] == 512)

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED: {len(failures)} of {checks} checks")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {checks} CHECKS PASSED — README numbers match the committed artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
