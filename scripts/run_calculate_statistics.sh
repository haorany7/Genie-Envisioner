#!/bin/bash

# Default values
DATA_ROOT=${1:-"/work/nvme/behe/WORLD-MODEL-TOUCH/task_usb_expert_lerobot"}
DATA_NAME=${2:-"task_usb_expert"}
DATA_TYPE=${3:-"joint"}
SAVE_DIR="/projects/behe/haorany7/WORLD-MODEL-TOUCH/Reimplementation/Genie-Envisioner/configs/ltx_model/${DATA_NAME}"
SAVE_PATH="${SAVE_DIR}/${DATA_NAME}_stats.json"

# Environment
# Try to find conda or use absolute path if known
source ~/.bashrc || echo "Warning: could not source .bashrc"

# Check if conda command exists, if not try common paths
if ! command -v conda &> /dev/null; then
    export PATH="/projects/bfxb/haorany7/miniconda3/bin:$PATH"
fi

conda activate genie_envisioner || { echo "Failed to activate conda environment 'genie_envisioner'"; exit 1; }

# Ensure save directory exists
mkdir -p "${SAVE_DIR}"

echo "🚀 Computing statistics for ${DATA_NAME}..."
echo "📂 Data Root: ${DATA_ROOT}"
echo "💾 Save Path: ${SAVE_PATH}"
echo "🔧 Data Type: ${DATA_TYPE}"

# Run the calculation script
python /projects/behe/haorany7/WORLD-MODEL-TOUCH/Reimplementation/Genie-Envisioner/scripts/calculate_statistics.py \
    --data_root "${DATA_ROOT}" \
    --data_name "${DATA_NAME}" \
    --data_type "${DATA_TYPE}" \
    --action_key actions \
    --state_key state \
    --save_path "${SAVE_PATH}" \
    --num_workers 8 \
    --check_jumps

echo "✅ Statistics generation completed!"
