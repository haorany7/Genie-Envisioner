#!/usr/bin/env bash
set -euo pipefail

# Usage: bash experiments/eval_xarm_gelsight_enhanced.sh [PROMPT_ID] [VARIANT]
# PROMPT_ID: 0 wipe | 1 cucumber_peel | 2 carrot_peel | 3 chip
# VARIANT:   all_tasks (default) | fancy_wipe

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genie_envisioner

VARIANT="${2:-all_tasks}"

if [[ "${VARIANT}" == "all_tasks" ]]; then
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/all_tasks/action_model_all_tasks_action_full_gelsight_enhanced_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/all_tasks_action_full_gelsight_enhanced/2026_02_18_14_25_19/step_5000/diffusion_pytorch_model.safetensors"
elif [[ "${VARIANT}" == "fancy_wipe" ]]; then
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/task_fancy_wipe_clean/action_model_task_fancy_wipe_clean_action_full_gelsight_enhanced_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_fancy_wipe_clean_action_full_gelsight_enhanced/2026_02_20_07_29_12/step_10000/diffusion_pytorch_model.safetensors"
else
  echo "Unknown VARIANT='${VARIANT}'. Use: all_tasks | fancy_wipe"
  exit 2
fi
DEVICE="cuda:0"

# Prompt dictionary (select via PROMPT_ID)
declare -A PROMPTS=(
  [0]="wipe the plate until clean"
  [1]="peel the cucumber into strips"
  [2]="peel the carrot into strips"
  [3]="pick up the chip and place it on the plate"
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
  --third_cam_index 12 \
  --gelsight_cam_index 6 \
  --use_third \
  --use_gelsight \
  --exec_step 54 \
  --num_inference_steps 10 \
  --smoothing 0.5
