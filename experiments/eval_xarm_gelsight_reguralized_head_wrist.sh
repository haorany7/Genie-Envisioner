#!/usr/bin/env bash
set -euo pipefail

# Usage: bash experiments/eval_xarm_gelsight_reguralized_head_wrist.sh [PROMPT_ID] [VARIANT]
# PROMPT_ID: 0="pick up the chip and place it on the plate" (default)
# VARIANT:   gelsight (default) — GelSight as visual + force in state
#             vision  — head+wrist only visual, GelSight force only in state

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genie_envisioner

VARIANT="${2:-gelsight}"

if [[ "${VARIANT}" == "gelsight" ]]; then
  # head + wrist + GelSight visual + force in state
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/task_chip_reguralized/action_model_task_chip_reguralized_head_wrist_action_full_gelsight_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_chip_reguralized_head_wrist_action_full_gelsight/step_20000/diffusion_pytorch_model.safetensors"
  GELSIGHT_FLAGS="--use_gelsight --gelsight_cam_index 6"
elif [[ "${VARIANT}" == "vision" ]]; then
  # head + wrist only visual, GelSight force only in state (no GelSight image input)
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/task_chip_reguralized/action_model_task_chip_reguralized_head_wrist_action_full_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_chip_reguralized_head_wrist_action_full/step_20000/diffusion_pytorch_model.safetensors"
  GELSIGHT_FLAGS="--use_gelsight --gelsight_force_only --gelsight_cam_index 6"
else
  echo "Unknown VARIANT='${VARIANT}'. Use: gelsight | vision"
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
  --base_cam_index 12 \
  --third_cam_index 18 \
  --use_third \
  --exec_step 54 \
  --num_inference_steps 10 \
  --smoothing 0.5 \
  ${GELSIGHT_FLAGS}
