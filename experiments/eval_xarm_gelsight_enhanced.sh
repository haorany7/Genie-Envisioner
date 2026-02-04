#!/usr/bin/env bash
set -euo pipefail

# Usage: bash experiments/eval_xarm_gelsight.sh [PROMPT_ID]
# PROMPT_ID: 0="wipe the plate", 1="peel the cucumber" (default), 2="plug in the USB", 3="unplug the USB"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genie_envisioner

CONFIG_FILE="configs/ltx_model/combined_peel_usb_wipe_joint/action_model_combined_peel_usb_wipe_joint_gelsight_enhanced_deployment.yaml"
CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_combined_peel_usb_wipe_joint_gelsight_enhanced_action/2026_01_30_09_31_47/step_30000/diffusion_pytorch_model.safetensors"
DEVICE="cuda:0"

# Prompt dictionary (select via PROMPT_ID)
declare -A PROMPTS=(
  [0]="wipe the plate"
  [1]="peel the cucumber"
  [2]="plug in the USB"
  [3]="unplug the USB"
)
PROMPT_ID="${1:-1}"
 PROMPT="${PROMPTS[$PROMPT_ID]:-${PROMPTS[0]}}"
 
 python experiments/eval_xarm.py \
   --config_file "${CONFIG_FILE}" \
   --ckpt_path "${CKPT_PATH}" \
   --prompt "${PROMPT}" \
   --device "${DEVICE}" \
   --base_cam_index 4 \
   --third_cam_index 10 \
   --gelsight_cam_index 12 \
   --use_third \
   --use_gelsight \
   --num_inference_steps 20 \
   --smoothing 0.5
