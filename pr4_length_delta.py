# Progress Report #4: per-length-bin effect of the merged training data.
#
# evaluate_pr3.py already writes each arm's length table, but nothing differences
# the two arms bin by bin. Because both arms were tested on the same holdout, the
# per-bin delta answers "which peptide lengths did the extra data actually help?".
#
# Works on the unzipped pr4_results folder or on a live results/ directory:
#   python pr4_length_delta.py path/to/pr4_results

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef

# Copied from evaluate_pr3.py rather than imported: that module pulls in torch,
# PyG and deepchem at import time, and this script only needs pandas, so it stays
# runnable on a downloaded results folder without the training environment.
PAPER_BINS = [
    ("11-16", 11, 16),
    ("16-22", 16, 22),
    ("22-27", 22, 27),
    ("27-33", 27, 33),
    ("33-39", 33, 39),
    ("39-44", 39, 45),
]
PAPER_ACCURACY = {"11-16": 0.769, "16-22": 0.781, "22-27": 0.875,
                  "27-33": 0.765, "33-39": 0.583, "39-44": 0.500}
LABELS = {
    "graphcpp_sage2": "GraphSAGE-2",
    "graphcpp_sage4": "GraphSAGE-4",
    "graphcpp_sage6": "GraphSAGE-6",
    "graphcpp_transformer2": "TransformerConv-2",
    "graphcpp_transformer4": "TransformerConv-4",
}

ARMS = ("original", "merged")


def load(root, arm, model):
    path = root / f"pr4_{arm}" / f"{model}_test_predictions.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return pd.read_csv(path)


def bin_rows(frame):
    """Per-bin accuracy/MCC, plus an explicit row for peptides no paper bin covers."""
    rows, covered = {}, np.zeros(len(frame), dtype=bool)
    for label, lower, upper in PAPER_BINS:
        mask = (frame["length"] >= lower) & (frame["length"] < upper)
        covered |= mask.to_numpy()
        sub = frame[mask]
        y, p = sub["label"].to_numpy(int), sub["prediction"].to_numpy(int)
        rows[label] = {
            "N": len(sub),
            "CPP": int((y == 1).sum()),
            "Accuracy": accuracy_score(y, p) if len(sub) else np.nan,
            "MCC": matthews_corrcoef(y, p) if len(np.unique(y)) == 2 else np.nan,
        }
    outside = frame[~covered]
    y, p = outside["label"].to_numpy(int), outside["prediction"].to_numpy(int)
    rows["outside 11-44"] = {
        "N": len(outside),
        "CPP": int((y == 1).sum()),
        "Accuracy": accuracy_score(y, p) if len(outside) else np.nan,
        "MCC": matthews_corrcoef(y, p) if len(np.unique(y)) == 2 else np.nan,
    }
    y, p = frame["label"].to_numpy(int), frame["prediction"].to_numpy(int)
    rows["Overall"] = {
        "N": len(frame),
        "CPP": int((y == 1).sum()),
        "Accuracy": accuracy_score(y, p),
        "MCC": matthews_corrcoef(y, p),
    }
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="results",
                        help="folder containing pr4_original/ and pr4_merged/")
    parser.add_argument("--models", nargs="+", default=list(LABELS))
    args = parser.parse_args()
    root = Path(args.root)

    long_rows, delta_rows = [], []
    for model in args.models:
        per_arm = {arm: load(root, arm, model) for arm in ARMS}

        # The whole comparison rests on the holdout being identical; check, don't assume.
        a, b = per_arm["original"], per_arm["merged"]
        for col in ("row_id", "name", "label", "length"):
            if not a[col].equals(b[col]):
                raise SystemExit(f"{model}: arms disagree on test column {col!r} -- "
                                 "the arms did not share a holdout, deltas are meaningless")

        binned = {arm: bin_rows(frame) for arm, frame in per_arm.items()}
        for bin_label in binned["original"]:
            o, m = binned["original"][bin_label], binned["merged"][bin_label]
            for arm, vals in (("original", o), ("merged", m)):
                long_rows.append({"Model": LABELS.get(model, model), "arm": arm,
                                  "Length range": bin_label, **vals})
            delta_rows.append({
                "Model": LABELS.get(model, model),
                "Length range": bin_label,
                "N": o["N"],
                "CPP": o["CPP"],
                "Paper Acc": PAPER_ACCURACY.get(bin_label, np.nan),
                "Acc original": o["Accuracy"],
                "Acc merged": m["Accuracy"],
                "Acc delta": m["Accuracy"] - o["Accuracy"],
                "MCC original": o["MCC"],
                "MCC merged": m["MCC"],
                "MCC delta": m["MCC"] - o["MCC"],
            })

    delta = pd.DataFrame(delta_rows)
    pd.DataFrame(long_rows).to_csv(root / "pr4_length_per_arm.csv", index=False)
    delta.to_csv(root / "pr4_length_delta.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)
    print("PER-LENGTH-BIN EFFECT OF THE MERGED TRAINING DATA (shared holdout, threshold 0.5)")
    print("N/CPP describe the holdout, so they are identical across arms by construction.\n")
    print(delta.to_string(index=False, float_format="%.4f", na_rep="-"))

    pivot = delta.pivot(index="Length range", columns="Model", values="Acc delta")
    order = [b[0] for b in PAPER_BINS] + ["outside 11-44", "Overall"]
    print("\nACCURACY DELTA (merged - original) BY BIN")
    print(pivot.reindex(order).to_string(float_format="%+.4f", na_rep="-"))
    print(f"\nSaved {root/'pr4_length_delta.csv'} and {root/'pr4_length_per_arm.csv'}")
