#!/bin/bash

# Script to compute CALVIN EEF statistics for ABC and D datasets
# This script applies unwrap to action and state angles, and doesn't cross episodes for delta computation
# Uses parallel processing for faster computation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Activate conda environment (optional, uncomment if needed)
# conda activate genie_envisioner

# Configuration
NUM_WORKERS=32  # Number of parallel workers
OUTPUT_DIR="$SCRIPT_DIR/statistics_output"
mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "Computing CALVIN ABC EEF Statistics (with $NUM_WORKERS workers)..."
echo "================================================"
python "$SCRIPT_DIR/compute_calvin_eef_statistics.py" \
    --data_root /work/hdd/behe/calvin/ABC_lerobot \
    --data_name ABC_lerobot \
    --data_type eef \
    --action_key action \
    --state_key observation.state \
    --save_path "$OUTPUT_DIR/ABC_lerobot_eef_statistics.json" \
    --nrnd 0 \
    --filter \
    --is_calvin_eef \
    --num_workers $NUM_WORKERS

echo ""
echo "================================================"
echo "Computing CALVIN D EEF Statistics (with $NUM_WORKERS workers)..."
echo "================================================"
python "$SCRIPT_DIR/compute_calvin_eef_statistics.py" \
    --data_root /work/hdd/behe/calvin/D_lerobot \
    --data_name D_lerobot \
    --data_type eef \
    --action_key action \
    --state_key observation.state \
    --save_path "$OUTPUT_DIR/D_lerobot_eef_statistics.json" \
    --nrnd 0 \
    --filter \
    --is_calvin_eef \
    --num_workers $NUM_WORKERS

echo ""
echo "================================================"
echo "Statistics computed successfully!"
echo "================================================"
echo "ABC statistics: $OUTPUT_DIR/ABC_lerobot_eef_statistics.json"
echo "D statistics: $OUTPUT_DIR/D_lerobot_eef_statistics.json"
echo ""
echo "Next steps:"
echo "1. Review the generated JSON files"
echo "2. Copy the statistics to data/utils/statistics.py"
echo "3. Update your config to use the new statistics"

