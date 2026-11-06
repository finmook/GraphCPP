#!/usr/bin/env bash
# Progress Report #4: does merging the extra MLCPP data help, and does the
# answer differ per architecture?
#
# Trains every architecture in config.EXPERIMENTS twice -- once on the original
# peptides, once on the merged set -- against a shared, byte-identical holdout,
# so the only difference between the two arms is the training data.
#
# Run prepare_merged_dataset.py first; it writes both dataset arms and the
# pos_weight each one needs.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-dataset/raw/pr4}"
SEED="${SEED:-42}"
CONFIGS=(graphcpp_sage2 graphcpp_sage4 graphcpp_sage6 graphcpp_transformer2 graphcpp_transformer4)

for arm in original merged; do
    if [[ ! -f "$DATA_ROOT/$arm/train.csv" ]]; then
        echo "error: $DATA_ROOT/$arm/train.csv not found -- run prepare_merged_dataset.py first" >&2
        exit 1
    fi
done

# Guard the premise of the whole comparison: both arms must test on the same peptides.
if ! cmp -s "$DATA_ROOT/original/test.csv" "$DATA_ROOT/merged/test.csv"; then
    echo "error: the two arms have different test.csv files -- the comparison would be confounded" >&2
    exit 1
fi

for arm in original merged; do
    pos_weight=$(python -c "import json;print(json.load(open('$DATA_ROOT/$arm/manifest.json'))['pos_weight'])")
    echo "=== arm=$arm pos_weight=$pos_weight"
    for config in "${CONFIGS[@]}"; do
        echo "--- $config on $arm data"
        python main.py --config "$config" --dataset "$DATA_ROOT/$arm" \
            --results_dir "results/pr4_$arm" --pos_weight "$pos_weight" --seed "$SEED"
    done
done

for arm in original merged; do
    echo "=== evaluating results/pr4_$arm"
    python evaluate_pr3.py --results_dir "results/pr4_$arm" --models "${CONFIGS[@]}"
done

echo
echo "Done. Both arms were tested on the same $DATA_ROOT/merged/test.csv, so"
echo "results/pr4_original/pr3_overall_comparison.csv and results/pr4_merged/"
echo "are directly comparable row by row."
