#!/usr/bin/env bash
set -euo pipefail

# Usage: bash experiments/eval_xarm_relative.sh [PROMPT_ID]
# PROMPT_ID: 0="wipe the plate", 1="peel the cucumber" (default), ...

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genie_envisioner

PROMPT_ID="${1:-1}"

echo "🔄 Running in RELATIVE mode"
CONFIG_FILE="configs/ltx_model/combined_peel_usb_wipe_joint/action_model_combined_peel_usb_wipe_joint_relative_deployment.yaml"
CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_combined_peel_usb_wipe_joint_action_relative_xinzhuo/2026_02_04_11_05_04/step_10000/diffusion_pytorch_model.safetensors"
ACTION_TYPE="relative_base"

# Prompt dictionary
declare -A PROMPTS=(
  [0]="wipe the plate"
  [1]="peel the cucumber"
  [2]="plug in the USB"
  [3]="unplug the USB"
)
PROMPT="${PROMPTS[$PROMPT_ID]:-${PROMPTS[1]}}"

python experiments/eval_xarm.py \
  --config_file "${CONFIG_FILE}" \
  --ckpt_path "${CKPT_PATH}" \
  --prompt "${PROMPT}" \
  --action_type "${ACTION_TYPE}" \
  --device "cuda:0" \
  --base_cam_index 4 \
  --third_cam_index 10 \
  --use_third \
  --num_inference_steps 20 \
  --smoothing 0.5
