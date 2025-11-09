#!/usr/bin/env bash
set -euo pipefail

# Optional: activate env if provided
if [ "${1:-}" = "--activate" ]; then
  shift
  eval "$(conda shell.bash hook)"
  conda activate genie_envisioner
fi

# Writable caches and disable bitsandbytes (avoid GLIBC_2.34 issue)
# Force to writable paths (override any pre-set values from the cluster env)
export HF_HOME="/work/hdd/bche/haorany7/.cache/huggingface"
export TRANSFORMERS_CACHE="/work/hdd/bche/haorany7/.cache/transformers"
export TRITON_CACHE_DIR="/work/hdd/bche/haorany7/.triton"
export USE_BITSANDBYTES=0
export BITSANDBYTES_NOWELCOME=1
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TRITON_CACHE_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Paths
MAIN_PY="${PROJ_ROOT}/main.py"
CONFIG_PATH="${PROJ_ROOT}/configs/ltx_model/video_model_infer_fast_calvin_tactile.yaml"
CKPT_PATH="${CKPT_PATH:-/projects/behe/haorany7/WORLD-MODEL-TOUCH/pretrained_models/genie_envisioner/GE_base_fast_v0.1.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/behe/haorany7/WORLD-MODEL-TOUCH/outputs/calvin_infer_tactile/$(date +%Y%m%d_%H%M%S)}"
DOMAIN_NAME="${DOMAIN_NAME:-ABC_lerobot_joint_action}"
TASKS_PER_RUN="${TASKS_PER_RUN:-34}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
BASE_N_VALIDATION="${N_VALIDATION:-34}"
N_CHUNK_ACTION="${N_CHUNK_ACTION:-30}"

if [[ -n "${TASKS_PER_RUN}" && "${TASKS_PER_RUN}" -gt 0 ]]; then
  if [[ -z "${EPISODES_PER_TASK}" || "${EPISODES_PER_TASK}" -le 0 ]]; then
    echo "EPISODES_PER_TASK must be a positive integer when TASKS_PER_RUN is set." >&2
    exit 1
  fi
  N_VALIDATION=$(( TASKS_PER_RUN * EPISODES_PER_TASK ))
else
  N_VALIDATION="${BASE_N_VALIDATION}"
  TASKS_PER_RUN=""
fi

mkdir -p "${OUTPUT_DIR}"
echo "Config:      ${CONFIG_PATH}"
echo "Checkpoint:  ${CKPT_PATH}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Domain:      ${DOMAIN_NAME}"
if [[ -n "${TASKS_PER_RUN}" ]]; then
  echo "Tasks/run:   ${TASKS_PER_RUN}"
  echo "Episodes/task: ${EPISODES_PER_TASK}"
fi

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
  --n_validation "${N_VALIDATION}" \
  --n_chunk_action "${N_CHUNK_ACTION}" \
  --domain_name "${DOMAIN_NAME}" \
  ${TASKS_PER_RUN:+--tasks_per_run "${TASKS_PER_RUN}"} \
  --episodes_per_task "${EPISODES_PER_TASK}"

echo "Done. Results at: ${OUTPUT_DIR}"


