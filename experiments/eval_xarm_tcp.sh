#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genie_envisioner

CONFIG_FILE="configs/ltx_model/combined_usb_wipe_tcp/action_model_combined_usb_wipe_tcp_gelsight_deployment.yaml"
CKPT_PATH="checkpoints/task_combined_usb_wipe_tcp_gelsight_enhanced_action/step_5000/diffusion_pytorch_model.safetensors"
DEVICE="cuda:0"

# Prompt dictionary (select via PROMPT_ID)
declare -A PROMPTS=(
  [0]="wipe the plate"
  [1]="plug in the USB"
  [2]="unplug the USB"
)
PROMPT_ID="${1:-0}"
PROMPT="${PROMPTS[$PROMPT_ID]:-${PROMPTS[0]}}"

python experiments/eval_xarm_tcp.py \
  --config_file "${CONFIG_FILE}" \
  --ckpt_path "${CKPT_PATH}" \
  --prompt "${PROMPT}" \
  --device "${DEVICE}" \
  --base_cam_index 4 \
  --third_cam_index 10 \
  --gelsight_cam_index 16 \
  --use_third \
  --use_gelsight \
  --num_inference_steps 10 \
  --smoothing 0.15
