#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genie_envisioner

# Usage:
#   bash experiments/eval_xarm.sh [PROMPT_ID] [VARIANT]
#   - PROMPT_ID: 0 wipe | 1 cucumber_peel | 2 carrot_peel | 3 chip
#   - VARIANT:   all_tasks (default) | all_tasks_gelsight | no_state | state | wipe_clean | wipe_tcp | peel_tcp | chip_task | chip_reguralized | chip_reguralized_head_wrist_gelsight

VARIANT="${2:-all_tasks}"

if [[ "${VARIANT}" == "all_tasks" ]]; then
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/all_tasks/action_model_all_tasks_action_full_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/all_tasks_action_full/2026_02_18_06_11_07/step_10000/diffusion_pytorch_model.safetensors"
elif [[ "${VARIANT}" == "all_tasks_gelsight" ]]; then
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/all_tasks/action_model_all_tasks_action_full_gelsight_enhanced_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/all_tasks_action_full_gelsight_enhanced/2026_02_18_14_25_19/step_5000/diffusion_pytorch_model.safetensors"
  USE_GELSIGHT="--use_gelsight --gelsight_cam_index 6"
elif [[ "${VARIANT}" == "no_state" ]]; then
  CONFIG_FILE="configs/ltx_model/combined_peel_usb_wipe_joint/action_model_combined_peel_usb_wipe_joint_no_state_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_combined_peel_usb_wipe_joint_action_no_state_xinzhuo/2026_02_04_11_18_06/step_10000/diffusion_pytorch_model.safetensors"
elif [[ "${VARIANT}" == "state" ]]; then
  CONFIG_FILE="configs/ltx_model/combined_peel_usb_wipe_joint/action_model_combined_peel_usb_wipe_joint_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_combined_peel_usb_wipe_joint_action/2026_02_06_12_06_59/step_10000/diffusion_pytorch_model.safetensors"
elif [[ "${VARIANT}" == "wipe_clean" ]]; then
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/task_wipe_clean/action_model_task_wipe_clean_action_full_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_wipe_clean_action_full/2026_02_10_05_19_00/step_10000/diffusion_pytorch_model.safetensors"
  INITIAL_GRIPPER_POS_OVERRIDE=700
elif [[ "${VARIANT}" == "fancy_wipe" ]]; then
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/task_fancy_wipe_clean/action_model_task_fancy_wipe_clean_action_full_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_fancy_wipe_clean_action_full/2026_02_19_10_17_05/step_10000/diffusion_pytorch_model.safetensors"
  INITIAL_GRIPPER_POS_OVERRIDE=700
elif [[ "${VARIANT}" == "wipe_tcp" ]]; then
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/wipe_tcp_in_state/action_model_wipe_tcp_in_state_action_full_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/wipe_tcp_in_state_action_full/step_20000/diffusion_pytorch_model.safetensors"
  INITIAL_GRIPPER_POS_OVERRIDE=700
elif [[ "${VARIANT}" == "peel_tcp" ]]; then
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/peel_tcp_in_state/action_model_peel_tcp_in_state_action_full_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/peel_tcp_in_state_action_full/step_20000/diffusion_pytorch_model.safetensors"
  INITIAL_GRIPPER_POS_OVERRIDE=240
elif [[ "${VARIANT}" == "chip_task" ]]; then
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/chip_task/action_model_chip_task_action_full_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/chip_task_action_full/step_4000/diffusion_pytorch_model.safetensors"
elif [[ "${VARIANT}" == "chip_reguralized" ]]; then
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/task_chip_reguralized/action_model_task_chip_reguralized_action_full_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_chip_reguralized_action_full/step_20000/diffusion_pytorch_model.safetensors"
  USE_GELSIGHT="--use_gelsight --gelsight_force_only --gelsight_cam_index 6"
elif [[ "${VARIANT}" == "chip_reguralized_head_wrist_gelsight" ]]; then
  CONFIG_FILE="/home/yuchenmo/Desktop/VLA/VTAM/configs/ltx_model/task_chip_reguralized/action_model_task_chip_reguralized_head_wrist_action_full_gelsight_deployment.yaml"
  CKPT_PATH="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_chip_reguralized_head_wrist_action_full_gelsight/step_20000/diffusion_pytorch_model.safetensors"
  USE_GELSIGHT="--use_gelsight --gelsight_cam_index 6"
else
  echo "Unknown VARIANT='${VARIANT}'. Use: all_tasks | all_tasks_gelsight | no_state | state | wipe_clean | chip_reguralized | chip_reguralized_head_wrist_gelsight"
  exit 2
fi
DEVICE="cuda:0"
USE_GELSIGHT="${USE_GELSIGHT:-}"
THIRD_CAM_INDEX="${THIRD_CAM_INDEX:-4}"

# Prompt dictionary (select via PROMPT_ID)
declare -A PROMPTS=(
  [0]="wipe the plate until clean"
  [1]="peel the cucumber into strips"
  [2]="peel the carrot into strips"
  [3]="pick up the chip and place it on the plate"
)
# Default gripper width per task (aligned with gelsight_reguralized.sh)
declare -A GRIPPER_PER_TASK=(
  [0]=700   # wipe
  [1]=240   # cucumber peel
  [2]=240   # carrot peel
  [3]=800   # chip
)
PROMPT_ID="${1:-0}"
PROMPT="${PROMPTS[$PROMPT_ID]:-${PROMPTS[0]}}"

# Per-variant override takes priority, then per-task default
GRIPPER_POS="${INITIAL_GRIPPER_POS_OVERRIDE:-${GRIPPER_PER_TASK[$PROMPT_ID]:-800}}"
GRIPPER_ARG="--initial_gripper_pos ${GRIPPER_POS}"

python experiments/eval_xarm.py \
  --config_file "${CONFIG_FILE}" \
  --ckpt_path "${CKPT_PATH}" \
  --prompt "${PROMPT}" \
  --action_type "absolute" \
  --device "${DEVICE}" \
  --base_cam_index 12 \
  --third_cam_index "${THIRD_CAM_INDEX}" \
  --use_third \
  --exec_step 54 \
  --num_inference_steps 10 \
  --smoothing 0.5 \
  ${USE_GELSIGHT} \
  ${GRIPPER_ARG}
