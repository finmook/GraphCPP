# Progress Report #3 evaluation.
# Re-evaluates each experiment's best checkpoint (selected by val_mcc in main.py) on the unshuffled
# independent test.csv, using the same alignment, threshold (0.5) and length bins as the PR2 notebook.
# Reads results/<experiment>.json written by main.py and writes CSV/JSON tables to results/.

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score, f1_score
from torch_geometric.loader import DataLoader

from config import BATCH_SIZE
from graphcpp.dataset import CPPDataset
from graphcpp.lightning import GraphCPPModule

warnings.filterwarnings("ignore")

MODELS = [
    ('graphcpp_sage4', 'GraphSAGE-4'),
    ('graphcpp_sage6', 'GraphSAGE-6'),
    ('graphcpp_transformer2', 'TransformerConv-2'),
    ('graphcpp_transformer4', 'TransformerConv-4'),
]
REFERENCE = 'graphcpp_sage4'

# Non-overlapping interpretation of the paper's adjacent bin labels (identical to PR2):
# [11,16), [16,22), [22,27), [27,33), [33,39), [39,45)
PAPER_BINS = [
    ("11-16", 11, 16),
    ("16-22", 16, 22),
    ("22-27", 22, 27),
    ("27-33", 27, 33),
    ("33-39", 33, 39),
    ("39-44", 39, 45),
]
PAPER_ACCURACY = {"11-16": 0.769, "16-22": 0.781, "22-27": 0.875, "27-33": 0.765, "33-39": 0.583, "39-44": 0.500}


def predict(checkpoint, dataset_dir, fp_type, device):
    model = GraphCPPModule.load_from_checkpoint(checkpoint_path=str(checkpoint), map_location=device)
    model.eval()
    model.freeze()
    model.to(device)

    test_dataset = CPPDataset(root=str(dataset_dir), _split="test", fp_type=fp_type)
    names, model_labels, probs_all = [], [], []
    with torch.no_grad():
        for batch in DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False):
            batch = batch.to(device)
            logits, labels, _ = model(batch)
            batch_names = [batch.name] if isinstance(batch.name, str) else batch.name
            names.extend([str(x) for x in batch_names])
            model_labels.extend(labels.reshape(-1).cpu().numpy().astype(int).tolist())
            probs_all.extend(torch.sigmoid(logits.reshape(-1)).cpu().numpy().tolist())

    pred = pd.DataFrame({"name": names, "model_label": model_labels, "probability": probs_all})
    pred["prediction"] = (pred["probability"] >= 0.5).astype(int)
    pred["_occ"] = pred.groupby("name").cumcount()

    # Align predictions to test.csv rows by peptide name (+ occurrence index for duplicate names)
    meta = pd.read_csv(Path(dataset_dir) / "test.csv")
    meta["length"] = pd.to_numeric(meta["len"], errors="raise").astype(int)
    if int((meta["sequence"].astype(str).str.len() != meta["length"]).sum()) != 0:
        raise RuntimeError("Sequence length does not agree with the test.csv length column.")
    meta["_occ"] = meta.groupby("name").cumcount()
    meta["row_id"] = np.arange(len(meta))

    result = meta.merge(pred, on=["name", "_occ"], how="left", validate="one_to_one")
    if result["probability"].isna().any():
        raise RuntimeError("Some test peptides are missing predictions.")
    result["label"] = result["label"].astype(int)
    if int((result["label"] != result["model_label"]).sum()) != 0:
        raise RuntimeError("Prediction/test label alignment failed.")
    result["correct"] = (result["prediction"] == result["label"]).astype(int)
    return result.drop(columns=["_occ"])


def overall_metrics(result):
    y, p, prob = result["label"], result["prediction"], result["probability"]
    return {
        "Accuracy": accuracy_score(y, p),
        "MCC": matthews_corrcoef(y, p),
        "ROC-AUC": roc_auc_score(y, prob),
        "Precision": precision_score(y, p, zero_division=0),
        "Recall": recall_score(y, p, zero_division=0),
        "F1": f1_score(y, p, zero_division=0),
    }


def length_metrics(result):
    rows = []
    for label, lower, upper in PAPER_BINS:
        frame = result[(result["length"] >= lower) & (result["length"] < upper)]
        y = frame["label"].to_numpy(dtype=int)
        p = frame["prediction"].to_numpy(dtype=int)
        rows.append({
            "Length range": label,
            "N": len(frame),
            "CPP": int((y == 1).sum()),
            "Non-CPP": int((y == 0).sum()),
            "Accuracy": accuracy_score(y, p) if len(frame) else np.nan,
            # MCC is only meaningful when both classes are present in the bin
            "MCC": matthews_corrcoef(y, p) if len(np.unique(y)) == 2 else np.nan,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    missing = [name for name, _ in MODELS if not (results_dir / f"{name}.json").is_file()]
    if missing:
        raise SystemExit(f"Missing training results for: {missing}. Run main.py --config <name> for each first.")

    preds, overall_rows, length_tables = {}, [], {}
    for name, label in MODELS:
        run = json.loads((results_dir / f"{name}.json").read_text())
        params = run["parameters"]
        print(f"{label}: {params['layer_type']} x{params['mp_layers']}  checkpoint={run['best_model_path']}")

        result = predict(run["best_model_path"], run["dataset"], params["fingerprint_type"], device)
        result.to_csv(results_dir / f"{name}_test_predictions.csv", index=False)
        preds[name] = result

        metrics = overall_metrics(result)
        overall_rows.append({"Model": label, **metrics, "Parameter Count": run["parameter_count"],
                             "layer_type": params["layer_type"], "mp_layers": params["mp_layers"]})

        lm = length_metrics(result)
        lm.to_csv(results_dir / f"{name}_length_metrics.csv", index=False)
        length_tables[name] = lm

    # Every model must have been evaluated on exactly the same test peptides, in the same order
    ref = preds[REFERENCE]
    for name, result in preds.items():
        for col in ["row_id", "name", "sequence", "label", "length"]:
            assert ref[col].equals(result[col]), f"Test-set mismatch for {name} in column {col}"

    # Task 8: overall comparison
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(results_dir / "pr3_overall_comparison.csv", index=False)
    (results_dir / "pr3_overall_comparison.json").write_text(overall.to_json(orient="records", indent=2))

    # Task 9: length-range comparison (accuracy primary, MCC supporting)
    comparison = length_tables[REFERENCE][["Length range", "N", "CPP", "Non-CPP"]].copy()
    comparison["Paper Accuracy"] = comparison["Length range"].map(PAPER_ACCURACY)
    for name, label in MODELS:
        comparison[f"{label} Accuracy"] = length_tables[name]["Accuracy"]
    for name, label in MODELS:
        comparison[f"{label} MCC"] = length_tables[name]["MCC"]
    comparison.to_csv(results_dir / "pr3_length_comparison.csv", index=False)

    # Task 10: paired same-peptide analysis against GraphSAGE-4
    paired_rows = []
    for name, label in MODELS:
        if name == REFERENCE:
            continue
        for bin_label, lower, upper in PAPER_BINS + [("Overall", 0, 10**6)]:
            mask = (ref["length"] >= lower) & (ref["length"] < upper)
            r, m = ref.loc[mask, "correct"], preds[name].loc[mask, "correct"]
            fixed, broken = int(((r == 0) & (m == 1)).sum()), int(((r == 1) & (m == 0)).sum())
            paired_rows.append({"Model": label, "Length range": bin_label, "N": int(mask.sum()),
                                "SAGE-4 wrong -> new correct": fixed, "SAGE-4 correct -> new wrong": broken,
                                "Net corrected by new model": fixed - broken})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(results_dir / "pr3_paired_vs_sage4.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\nOVERALL INDEPENDENT-TEST COMPARISON (threshold 0.5)")
    print(overall.drop(columns=["layer_type", "mp_layers"]).to_string(index=False, float_format="%.4f"))
    print("\nSAME LENGTH-RANGE COMPARISON (accuracy primary; MCC NaN = single-class bin)")
    print(comparison.to_string(index=False, float_format="%.4f"))
    print("\nPAIRED SAME-PEPTIDE ANALYSIS vs GraphSAGE-4")
    print(paired.to_string(index=False))
    print(f"\nSaved tables to {results_dir}/")
