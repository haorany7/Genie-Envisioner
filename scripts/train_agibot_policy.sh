#!/usr/bin/env bash
set -euo pipefail

# Optional: activate env if provided
if [ "${1:-}" = "--activate" ]; then
  shift
  eval "$(conda shell.bash hook)"
  conda activate genie_envisioner
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MAIN_PY="${PROJ_ROOT}/main.py"
BASE_CONFIG="${PROJ_ROOT}/configs/ltx_model/policy_model_lerobot.yaml"

DEFAULT_WORK_ROOT="/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH"
WORK_ROOT="${WM_WORK_ROOT:-${DEFAULT_WORK_ROOT}}"

# Paths (customize if needed)
DATA_ROOT="/work/hdd/behe/AgiBotWorld-Alpha_lerobot"
DOMAIN="agibotworld"
CACHE_DIR_TRAIN="/projects/behe/haorany7/WORLD-MODEL-TOUCH/cache/ge_eval_cache_train"
CACHE_DIR_VAL="/projects/behe/haorany7/WORLD-MODEL-TOUCH/cache/ge_eval_cache_val"
CACHE_TRAIN="${CACHE_DIR_TRAIN}/agibotworld_train.json"
CACHE_VAL="${CACHE_DIR_VAL}/agibotworld_val.json"
PRETRAIN_VAE_TOK="/projects/behe/haorany7/WORLD-MODEL-TOUCH/pretrained_models/ltx_video"
MODEL_CKPT="/projects/behe/haorany7/WORLD-MODEL-TOUCH/pretrained_models/genie_envisioner/GE_base_fast_v0.1.safetensors"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/outputs/agibot_policy/${TIMESTAMP}}"
RUN_CFG_DIR="${OUTPUT_DIR}/used_configs"
GEN_CONFIG="${RUN_CFG_DIR}/policy_model_lerobot.generated.yaml"

mkdir -p "${OUTPUT_DIR}" "${RUN_CFG_DIR}" "${CACHE_DIR_TRAIN}" "${CACHE_DIR_VAL}"

echo "Preparing run-specific config at: ${GEN_CONFIG}"

# Build a generated config by replacing placeholders and dataset fields
sed -E \
  -e "s|^output_dir: .*|output_dir: ${OUTPUT_DIR}|" \
  -e "s|^pretrained_model_name_or_path: .*|pretrained_model_name_or_path: ${PRETRAIN_VAE_TOK}|" \
  -e "s|^([[:space:]]*)model_path: .*|\1model_path: ${MODEL_CKPT}|" \
  -e "s|^([[:space:]]*)data_roots:.*|\1data_roots: [\"${DATA_ROOT}\"]|g" \
  -e "s|^([[:space:]]*)domains:.*|\1domains: [\"${DOMAIN}\"]|g" \
  -e "s|^([[:space:]]*)valid_cam[[:space:]]*:.*|\1valid_cam :  ['observation.images.head','observation.images.hand_left','observation.images.hand_right']|g" \
  -e "0,/dataset_info_cache_path:.*/ s|dataset_info_cache_path: .*|dataset_info_cache_path: \"${CACHE_TRAIN}\"|" \
  -e "1,/dataset_info_cache_path:.*/ s|dataset_info_cache_path: .*|dataset_info_cache_path: \"${CACHE_VAL}\"|" \
  -e "s|^([[:space:]]*)action_type:.*|\1action_type: \"absolute\"|g" \
  -e "s|^([[:space:]]*)action_space:.*|\1action_space: \"joint\"|g" \
  "${BASE_CONFIG}" > "${GEN_CONFIG}"

echo "Config prepared:"
echo "  Base: ${BASE_CONFIG}"
echo "  Gen : ${GEN_CONFIG}"
echo "  Out : ${OUTPUT_DIR}"

# Launch training on all visible GPUs
NGPU=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
echo "Launching training on ${NGPU} GPU(s)"

torchrun --nnodes=1 \
  --nproc_per_node="${NGPU}" \
  --node_rank=0 \
  "${MAIN_PY}" \
  --config_file "${GEN_CONFIG}" \
  --runner_class_path runner/ge_trainer.py \
  --runner_class Trainer \
  --mode train

echo "Training started. Logs/checkpoints under: ${OUTPUT_DIR}"


