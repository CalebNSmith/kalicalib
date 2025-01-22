#!/usr/bin/env bash

# Example shell script to automate runs with various hyperparameters

CONFIG_FILE="configs/default.yaml"
DATA_DIR="../data/kalicalib_v3/"  # Update this to your actual data dir
OUTPUT_DIR="outputs/train_runs_sweeps"

# Make sure the script is executable:
#   chmod +x run_experiments.sh

# Sweeping over different batch sizes and learning rates:
for KEYPOINT_WEIGHT in 1 10; do
    echo "============================================================"
    echo "Running experiment with keypoint_weight=${KEYPOINT_WEIGHT}"
    echo "============================================================"

    python scripts/train.py \
      --config "${CONFIG_FILE}" \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --keypoint-weight "${KEYPOINT_WEIGHT}" \
      --n-epochs 200 \
      --num-workers 4

    # You can add more args if needed, e.g.:
    # --keypoint-weight 20 \
    # --background-weight 2
done
