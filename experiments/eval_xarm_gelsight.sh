#!/usr/bin/env bash
set -euo pipefail

# Usage: bash experiments/eval_xarm_gelsight.sh [PROMPT_ID] [VARIANT]
# PROMPT_ID: see per-variant prompt dictionaries below
# VARIANT:   combined (default) | chip_task_xinzhuo

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genie_envisioner

VARIANT="${2:-combined}"
DEVICE="cuda:0"

if [[ "${VARIANT}" == "combined" ]]; then
  # Original combined peel/usb/wipe model
  CONFIG_FILE="configs/ltx_model/combined_peel_usb_wipe_joint/action_model_combined_peel_usb_wipe_joint_gelsight_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_combined_peel_usb_wipe_joint_action/2026_01_30_22_45_50/step_20000/diffusion_pytorch_model.safetensors"
  ACTION_TYPE="absolute"
  BASE_CAM_INDEX=4
  THIRD_CAM_INDEX=10
  GELSIGHT_CAM_INDEX=12
  NUM_INFERENCE_STEPS=20
  SMOOTHING=0.3
  EXEC_STEP=""

  declare -A PROMPTS=(
    [0]="wipe the plate"
    [1]="peel the cucumber"
    [2]="plug in the USB"
    [3]="unplug the USB"
  )
  PROMPT_ID="${1:-1}"

elif [[ "${VARIANT}" == "chip_task_xinzhuo" ]]; then
  # Chip pick and place model trained on xinzhuo (7D state, GelSight visual only)
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/chip_task/action_model_chip_task_action_full_gelsight_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/chip_task_action_full_gelsight/step_20000/diffusion_pytorch_model.safetensors"
  ACTION_TYPE="absolute"
  BASE_CAM_INDEX=4
  THIRD_CAM_INDEX=10
  GELSIGHT_CAM_INDEX=12
  NUM_INFERENCE_STEPS=10
  SMOOTHING=0.5
  EXEC_STEP="--exec_step 54"

  declare -A PROMPTS=(
    [0]="pick up the chip and place it on the plate"
  )
  PROMPT_ID="${1:-0}"

else
  echo "Unknown VARIANT='${VARIANT}'. Use: combined | chip_task_xinzhuo"
  exit 2
fi

PROMPT="${PROMPTS[$PROMPT_ID]:-${PROMPTS[0]}}"

python experiments/eval_xarm.py \
  --config_file "${CONFIG_FILE}" \
  --ckpt_path "${CKPT_PATH}" \
  --prompt "${PROMPT}" \
  --action_type "${ACTION_TYPE}" \
  --device "${DEVICE}" \
  --base_cam_index "${BASE_CAM_INDEX}" \
  --third_cam_index "${THIRD_CAM_INDEX}" \
  --gelsight_cam_index "${GELSIGHT_CAM_INDEX}" \
  --use_third \
  --use_gelsight \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --smoothing "${SMOOTHING}" \
  ${EXEC_STEP}
