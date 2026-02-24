#!/usr/bin/env bash
set -euo pipefail

# Usage: bash experiments/eval_xarm_gelsight_reguralized.sh [PROMPT_ID] [VARIANT]
# PROMPT_ID: 0="pick up the chip and place it on the plate" (default)
# VARIANT:   chip_reguralized (default)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genie_envisioner

VARIANT="${2:-chip_reguralized}"

if [[ "${VARIANT}" == "chip_reguralized" ]]; then
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/task_chip_reguralized/action_model_task_chip_reguralized_action_full_gelsight_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_chip_reguralized_action_full_gelsight/step_20000/diffusion_pytorch_model.safetensors"
else
  echo "Unknown VARIANT='${VARIANT}'. Use: chip_reguralized"
  exit 2
fi
DEVICE="cuda:0"

# Prompt dictionary (select via PROMPT_ID)
declare -A PROMPTS=(
  [0]="pick up the chip and place it on the plate"
)
PROMPT_ID="${1:-0}"
PROMPT="${PROMPTS[$PROMPT_ID]:-${PROMPTS[0]}}"

python experiments/eval_xarm.py \
  --config_file "${CONFIG_FILE}" \
  --ckpt_path "${CKPT_PATH}" \
  --prompt "${PROMPT}" \
  --action_type "absolute" \
  --device "${DEVICE}" \
  --base_cam_index 4 \
  --third_cam_index 10 \
  --gelsight_cam_index 12 \
  --use_third \
  --use_gelsight \
  --exec_step 54 \
  --num_inference_steps 10 \
  --smoothing 0.5
