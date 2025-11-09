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

# Paths
MAIN_PY="${PROJ_ROOT}/main.py"
CONFIG_PATH="${PROJ_ROOT}/configs/ltx_model/video_model_infer_fast_lerobot.yaml"
CKPT_PATH="/projects/behe/haorany7/WORLD-MODEL-TOUCH/pretrained_models/genie_envisioner/GE_base_fast_v0.1.safetensors"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/behe/haorany7/WORLD-MODEL-TOUCH/outputs/agibot_infer/$(date +%Y%m%d_%H%M%S)}"
DOMAIN_NAME="${DOMAIN_NAME:-Agibot_lerobot}"

mkdir -p "${OUTPUT_DIR}"
echo "Config:      ${CONFIG_PATH}"
echo "Checkpoint:  ${CKPT_PATH}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Domain:      ${DOMAIN_NAME}"

# Single-GPU inference
torchrun --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  "${MAIN_PY}" \
  --runner_class_path runner/ge_inferencer.py \
  --runner_class Inferencer \
  --config_file "${CONFIG_PATH}" \
  --mode infer \
  --checkpoint_path "${CKPT_PATH}" \
  --output_path "${OUTPUT_DIR}" \
  --n_validation 10 \
  --n_chunk_action 10 \
  --domain_name "${DOMAIN_NAME}"

echo "Done. Results at: ${OUTPUT_DIR}"


