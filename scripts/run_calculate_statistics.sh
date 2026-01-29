#!/bin/bash

# Default values
DATA_ROOT=${1:-"/workspace/vtam/data/combined_peel_usb_wipe_joint_lerobot"}
DATA_NAME=${2:-"combined_peel_usb_wipe_joint"}
DATA_TYPE=${3:-"joint"}
SAVE_DIR="/workspace/vtam/Genie-Envisioner/configs/ltx_model/${DATA_NAME}"
SAVE_PATH="${SAVE_DIR}/${DATA_NAME}_stats.json"

# Environment
source /workspace/vtam/Genie-Envisioner/venv_ge/bin/activate

# Ensure save directory exists
mkdir -p "${SAVE_DIR}"

echo "🚀 Computing statistics for ${DATA_NAME}..."
echo "📂 Data Root: ${DATA_ROOT}"
echo "💾 Save Path: ${SAVE_PATH}"
echo "🔧 Data Type: ${DATA_TYPE}"

# Run the calculation script
python /workspace/vtam/Genie-Envisioner/scripts/calculate_statistics.py \
    --data_root "${DATA_ROOT}" \
    --data_name "${DATA_NAME}" \
    --data_type "${DATA_TYPE}" \
    --action_key actions \
    --state_key state \
    --save_path "${SAVE_PATH}" \
    --num_workers 8 \
    --check_jumps \
    --gripper_threshold 50

echo "✅ Statistics generation completed!"
