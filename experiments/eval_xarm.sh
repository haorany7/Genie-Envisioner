#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genie_envisioner

CONFIG_FILE="configs/ltx_model/combined_peel_usb_wipe_joint/action_model_combined_peel_usb_wipe_joint_no_state_deployment.yaml"
CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_combined_peel_usb_wipe_joint_action_no_state_xinzhuo/2026_02_04_11_18_06/step_10000/diffusion_pytorch_model.safetensors"
DEVICE="cuda:0"

# Prompt dictionary (select via PROMPT_ID)
declare -A PROMPTS=(
  [0]="wipe the plate"
  [1]="peel the cucumber"
  [2]="plug in the USB"
  [3]="unplug the USB"
)
PROMPT_ID="${1:-1}"
PROMPT="${PROMPTS[$PROMPT_ID]:-${PROMPTS[1]}}"

python experiments/eval_xarm.py \
  --config_file "${CONFIG_FILE}" \
  --ckpt_path "${CKPT_PATH}" \
  --prompt "${PROMPT}" \
  --action_type "absolute" \
  --device "${DEVICE}" \
  --base_cam_index 4 \
  --third_cam_index 10 \
  --use_third \
  --exec_step 10 \
  --num_inference_steps 20 \
  --smoothing 0.5
