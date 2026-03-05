#!/usr/bin/env bash
set -euo pipefail

# Usage: bash experiments/eval_xarm_gelsight_reguralized.sh [PROMPT_ID] [VARIANT]
# PROMPT_ID: 0="pick up the chip and place it on the plate" (default)
# VARIANT:   chip_reguralized (default) | force_in_action | force_in_action_relative | cucumber_peel | hard_wipe

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genie_envisioner

VARIANT="${2:-chip_reguralized}"

if [[ "${VARIANT}" == "chip_reguralized" ]]; then
  # force in state, GelSight as visual input
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/task_chip_reguralized/action_model_task_chip_reguralized_action_full_gelsight_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_chip_reguralized_action_full_gelsight/step_20000/diffusion_pytorch_model.safetensors"
elif [[ "${VARIANT}" == "force_in_action" ]]; then
  # force in action (absolute), GelSight as visual input, 26D = action(10) + state(16)
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/task_chip_force_in_action/action_model_task_chip_force_in_action_action_full_gelsight_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_chip_force_in_action_action_full_gelsight/step_20000/diffusion_pytorch_model.safetensors"
elif [[ "${VARIANT}" == "force_in_action_relative" ]]; then
  # force in action (relative), GelSight as visual input, 26D = action(10) + state(16)
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/task_chip_force_in_action/action_model_task_chip_force_in_action_action_full_gelsight_relative_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_chip_force_in_action_action_full_gelsight_relative/step_20000/diffusion_pytorch_model.safetensors"
  ACTION_TYPE_OVERRIDE="relative_base"
elif [[ "${VARIANT}" == "cucumber_peel" ]]; then
  # cucumber peel, force in action (absolute), GelSight as visual input, 26D
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/task_cucumber_peel_force_in_action/action_model_task_cucumber_peel_force_in_action_action_full_gelsight_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_cucumber_peel_force_in_action_action_full_gelsight/step_20000/diffusion_pytorch_model.safetensors"
  INITIAL_GRIPPER_POS_OVERRIDE=240
elif [[ "${VARIANT}" == "hard_wipe" ]]; then
  # hard wipe, force in action (absolute), GelSight as visual input, 26D
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/task_hard_wipe_force_in_action/action_model_task_hard_wipe_force_in_action_action_full_gelsight_runpod_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_hard_wipe_force_in_action_action_full_gelsight/step_20000/diffusion_pytorch_model.safetensors"
  INITIAL_GRIPPER_POS_OVERRIDE=700
else
  echo "Unknown VARIANT='${VARIANT}'. Use: chip_reguralized | force_in_action | force_in_action_relative | cucumber_peel | hard_wipe"
  exit 2
fi
DEVICE="cuda:0"

# Prompt dictionary (select via PROMPT_ID)
declare -A PROMPTS=(
  [0]="pick up the chip and place it on the plate"
  [1]="peel the cucumber into strips"
  [2]="wipe the whiteboard with the eraser until clean"
)
PROMPT_ID="${1:-0}"
 PROMPT="${PROMPTS[$PROMPT_ID]:-${PROMPTS[0]}}"
 
 ACTION_TYPE="${ACTION_TYPE_OVERRIDE:-absolute}"
 GRIPPER_ARG=""
 if [[ -n "${INITIAL_GRIPPER_POS_OVERRIDE:-}" ]]; then
   GRIPPER_ARG="--initial_gripper_pos ${INITIAL_GRIPPER_POS_OVERRIDE}"
 fi
 python experiments/eval_xarm.py \
   --config_file "${CONFIG_FILE}" \
   --ckpt_path "${CKPT_PATH}" \
   --prompt "${PROMPT}" \
  --action_type "${ACTION_TYPE}" \
   --device "${DEVICE}" \
   --base_cam_index 12 \
   --third_cam_index 4 \
   --gelsight_cam_index 6 \
   --use_third \
   --use_gelsight \
  --exec_step 54 \
  --num_inference_steps 10 \
  --smoothing 0.5 \
  ${GRIPPER_ARG}
