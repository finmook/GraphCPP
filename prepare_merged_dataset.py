"""Merge the original GraphCPP splits with additional MLCPP data into new splits.

Pools every source, removes duplicate peptides by canonical SMILES, then writes
train/val/test.csv plus a manifest describing exactly what went where.

Validation and test are built class-balanced (50/50) so their MCC/AUC stay
directly comparable to the PR2/PR3 numbers in results/. Every remaining peptide
goes to train, which therefore keeps its natural class imbalance -- train with
`pos_weight` (printed below) rather than discarding negatives.

Two dataset folders are written, sharing byte-identical val/test splits:

    <out>/merged/    train = every pooled peptide outside the holdout
    <out>/original/  train = only peptides from the original splits, with any
                     peptide that landed in the holdout removed

Sharing the holdout is the whole point: it is what makes "more data" the only
difference between the two arms, so the comparison measures the data and not a
change of test set.

Sequence-only inputs (FASTA, or a CSV with a `sequence` but no `smiles` column)
are converted with Chem.MolFromFASTA, the same path graphcpp.dataset uses.

Usage:
    python prepare_merged_dataset.py \
        --old dataset/raw/2024-07-14-15-40-08 \
        --extra dataset/mlcpp2_independent.csv \
        --out dataset/raw/merged
"""

import argparse
import json
import os
import os.path as osp
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:  # pragma: no cover - lets --dry-run work outside the env
    HAS_RDKIT = False

OLD_SPLITS = ("train", "val", "test")
# Columns every pooled frame is normalised to before dedup.
COLUMNS = ["name", "source", "sequence", "smiles", "label"]


def canonicalize(smiles):
    """Canonical SMILES for dedup, or None if RDKit cannot parse it."""
    if not HAS_RDKIT:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def sequence_from_smiles(smiles):
    """Recover the amino-acid sequence from SMILES, or None when the peptide is
    cyclic/modified and RDKit cannot express it as a plain sequence."""
    if not HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        sequence = Chem.MolToSequence(mol)
    except Exception:
        return None
    return sequence.strip().upper() or None


def smiles_from_sequence(sequence):
    """SMILES for a plain amino-acid sequence, or None if it cannot be built."""
    if not HAS_RDKIT:
        raise RuntimeError("RDKit is required to convert sequences to SMILES")
    mol = Chem.MolFromFASTA(sequence)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def read_fasta(path, default_label=None):
    """Read a FASTA file into a frame. Labels come from the header when it
    contains `positive`/`negative` (or CPP/non-CPP), else from default_label."""
    records = []
    header, chunks = None, []

    def flush():
        if header is None:
            return
        sequence = "".join(chunks).strip().upper()
        if not sequence:
            return
        low = header.lower()
        if "non-cpp" in low or "noncpp" in low or "negative" in low:
            label = 0
        elif "cpp" in low or "positive" in low:
            label = 1
        else:
            label = default_label
        records.append({"name": header, "sequence": sequence, "label": label})

    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                flush()
                header, chunks = line[1:].strip(), []
            elif line:
                chunks.append(line)
    flush()

    frame = pd.DataFrame(records)
    if len(frame) and frame["label"].isna().any():
        raise ValueError(
            f"{path}: cannot infer labels from headers; pass --extra {path}:0 "
            "or :1 to set the label for the whole file"
        )
    return frame


def load_source(spec, label_override=None):
    """Load one input file (CSV or FASTA) into the normalised column set."""
    if spec.lower().endswith((".fasta", ".fa", ".txt")):
        frame = read_fasta(spec, default_label=label_override)
    else:
        frame = pd.read_csv(spec)
        if label_override is not None:
            frame["label"] = label_override

    if "label" not in frame:
        raise ValueError(f"{spec}: no label column and no label override given")

    if "smiles" not in frame:
        if "sequence" not in frame:
            raise ValueError(f"{spec}: needs a smiles or sequence column")
        frame["smiles"] = [smiles_from_sequence(s) for s in frame["sequence"]]

    if "source" not in frame:
        frame["source"] = osp.splitext(osp.basename(spec))[0]
    if "sequence" not in frame:
        frame["sequence"] = None
    if "name" not in frame:
        frame["name"] = [f"{frame['source'].iloc[0]}_{i}" for i in range(len(frame))]

    frame["origin_file"] = spec
    return frame[COLUMNS + ["origin_file"]]


def load_old_splits(folder):
    frames = []
    for split in OLD_SPLITS:
        frame = pd.read_csv(osp.join(folder, f"{split}.csv"))
        frame["origin_file"] = osp.join(folder, f"{split}.csv")
        frame["old_split"] = split
        frames.append(frame[COLUMNS + ["origin_file", "old_split"]])
    return pd.concat(frames, ignore_index=True)


def dedup(pooled):
    """Drop unparseable rows and duplicate peptides, keeping the first
    occurrence (old splits are pooled first, so they win ties)."""
    pooled = pooled.copy()
    pooled["canonical_smiles"] = [canonicalize(s) for s in pooled["smiles"]]

    unparseable = pooled[pooled["canonical_smiles"].isna()]
    pooled = pooled[pooled["canonical_smiles"].notna()]

    # Flag peptides that appear twice with conflicting labels -- a silent
    # relabel would quietly corrupt the benchmark, so refuse instead.
    conflicts = (
        pooled.groupby("canonical_smiles")["label"].nunique().loc[lambda s: s > 1]
    )
    if len(conflicts):
        examples = pooled[pooled["canonical_smiles"].isin(conflicts.index[:5])]
        raise ValueError(
            f"{len(conflicts)} peptide(s) appear with conflicting labels across "
            f"sources; resolve before merging. Examples:\n"
            f"{examples[['name', 'source', 'label']].to_string()}"
        )

    duplicates = pooled[pooled.duplicated("canonical_smiles", keep="first")]
    deduped = pooled.drop_duplicates("canonical_smiles", keep="first")
    return deduped.reset_index(drop=True), duplicates, unparseable


def add_sequence_and_length(frame):
    """Ensure every row has a `sequence` and a matching `len`.

    evaluate_pr3.py asserts that test.csv carries both and that they agree, so
    rows whose sequence cannot be recovered are marked ineligible for the
    val/test splits (they are still perfectly usable for training).
    """
    frame = frame.copy()
    sequences = []
    for existing, smiles in zip(frame["sequence"], frame["smiles"]):
        if isinstance(existing, str) and existing.strip():
            sequences.append(existing.strip().upper())
        else:
            sequences.append(sequence_from_smiles(smiles))
    frame["sequence"] = sequences
    frame["len"] = [len(s) if isinstance(s, str) else pd.NA for s in sequences]
    frame["holdout_eligible"] = frame["sequence"].notna()
    return frame


def balanced_sample(frame, n_per_class, rng):
    """Take n_per_class rows of each label; returns (sampled, remainder)."""
    picked = []
    for label in (0, 1):
        pool = frame[frame["label"] == label]
        if len(pool) < n_per_class:
            raise ValueError(
                f"only {len(pool)} rows with label={label} available, "
                f"need {n_per_class}"
            )
        picked.append(pool.sample(n=n_per_class, random_state=rng.integers(2**32)))
    sampled = pd.concat(picked)
    remainder = frame.drop(sampled.index)
    return sampled, remainder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old",
        default="dataset/raw/2024-07-14-15-40-08",
        help="Folder holding the original train/val/test.csv",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Additional CSV/FASTA to merge in. Append :0 or :1 to force the "
        "label for the whole file. Repeatable.",
    )
    parser.add_argument("--out", default="dataset/raw/merged", help="Output folder")
    parser.add_argument(
        "--test-per-class", type=int, default=150, help="Balanced test rows per class"
    )
    parser.add_argument(
        "--val-per-class", type=int, default=150, help="Balanced val rows per class"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the merge and the resulting splits without writing files",
    )
    args = parser.parse_args()

    if not HAS_RDKIT:
        print(
            "WARNING: RDKit not importable -- dedup falls back to exact SMILES "
            "string matching and sequence inputs are unsupported.\n"
        )

    rng = np.random.default_rng(args.seed)

    old = load_old_splits(args.old)
    frames = [old]
    for spec in args.extra:
        path, _, override = spec.rpartition(":")
        if override in ("0", "1") and path:
            frames.append(load_source(path, label_override=int(override)))
        else:
            frames.append(load_source(spec))

    pooled = pd.concat(frames, ignore_index=True)
    deduped, duplicates, unparseable = dedup(pooled)

    print(f"Pooled rows:       {len(pooled)}")
    for path, count in pooled["origin_file"].value_counts().items():
        print(f"  {count:>6}  {path}")
    if len(unparseable):
        print(f"Dropped (unparseable SMILES): {len(unparseable)}")
    print(f"Duplicates removed: {len(duplicates)}")
    print(f"Unique peptides:    {len(deduped)}")

    deduped = add_sequence_and_length(deduped)
    eligible = deduped[deduped["holdout_eligible"]]
    ineligible = deduped[~deduped["holdout_eligible"]]
    print(
        f"Sequence recovered:  {len(eligible)} "
        f"({len(ineligible)} cyclic/modified -> train only)"
    )

    # Balanced val/test are drawn only from sequence-known peptides so the
    # length-range analysis in evaluate_pr3.py stays valid.
    test, rest = balanced_sample(eligible, args.test_per_class, rng)
    val, rest = balanced_sample(rest, args.val_per_class, rng)
    merged_train = pd.concat([rest, ineligible])

    # The paired arm: same holdout, but trained only on the original peptides.
    original_train = merged_train[merged_train["old_split"].notna()]

    arms = {
        "merged": {"train": merged_train, "val": val, "test": test},
        "original": {"train": original_train, "val": val, "test": test},
    }

    pos_weights = {}
    for arm, splits in arms.items():
        print(f"\n{arm}/")
        for name, frame in splits.items():
            counts = frame["label"].value_counts().to_dict()
            pos = counts.get(1, 0)
            print(
                f"  {name:>5}: {len(frame):>5} rows  "
                f"{pos} pos / {counts.get(0, 0)} neg  ({pos / len(frame):.1%} positive)"
            )
        train = splits["train"]
        pos_weights[arm] = (train["label"] == 0).sum() / (train["label"] == 1).sum()
        print(f"  train pos_weight: {pos_weights[arm]:.4f}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    for arm, splits in arms.items():
        arm_dir = osp.join(args.out, arm)
        os.makedirs(arm_dir, exist_ok=True)
        for name, frame in splits.items():
            frame = frame.drop(columns=["holdout_eligible"]).reset_index(drop=True)
            frame.insert(0, "id", frame.index)
            frame.to_csv(osp.join(arm_dir, f"{name}.csv"), index=False)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "old_split_folder": args.old,
        "extra_sources": args.extra,
        "rdkit_canonicalization": HAS_RDKIT,
        "pooled_rows": int(len(pooled)),
        "rows_per_origin": {
            str(k): int(v) for k, v in pooled["origin_file"].value_counts().items()
        },
        "duplicates_removed": int(len(duplicates)),
        "unparseable_dropped": int(len(unparseable)),
        "unique_peptides": int(len(deduped)),
        "test_per_class": args.test_per_class,
        "val_per_class": args.val_per_class,
        "sequence_recovered": int(len(eligible)),
        "train_only_no_sequence": int(len(ineligible)),
        "pos_weight": {arm: float(w) for arm, w in pos_weights.items()},
        "arms": {
            arm: {
                name: {
                    "rows": int(len(frame)),
                    "positive": int((frame["label"] == 1).sum()),
                    "negative": int((frame["label"] == 0).sum()),
                }
                for name, frame in splits.items()
            }
            for arm, splits in arms.items()
        },
    }
    for arm in arms:
        with open(osp.join(args.out, arm, "manifest.json"), "w") as handle:
            json.dump({**manifest, "arm": arm, "pos_weight": float(pos_weights[arm])}, handle, indent=2)

    duplicates[["name", "source", "label", "origin_file"]].to_csv(
        osp.join(args.out, "removed_duplicates.csv"), index=False
    )
    print(f"\nWrote {', '.join(arms)} splits + manifest.json under {args.out}/")


if __name__ == "__main__":
    main()
