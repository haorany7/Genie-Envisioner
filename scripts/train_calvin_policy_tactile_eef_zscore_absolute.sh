#!/usr/bin/env bash
set -euo pipefail

# Optional: conda activation
if [ "${1:-}" = "--activate" ]; then
  shift
  eval "$(conda shell.bash hook)"
  conda activate genie_envisioner
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MAIN_PY="${PROJ_ROOT}/main.py"
CONFIG_FILE="${PROJ_ROOT}/configs/ltx_model/policy_model_calvin_tactile_eef_zscore_absolute.yaml"

echo "Project root   : ${PROJ_ROOT}"
echo "Main script    : ${MAIN_PY}"
echo "Config file    : ${CONFIG_FILE}"

if [ ! -f "${MAIN_PY}" ]; then
  echo "ERROR: main.py not found at ${MAIN_PY}" >&2
  exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "ERROR: config file not found at ${CONFIG_FILE}" >&2
  exit 1
fi

# Detect number of GPUs
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
else
  NGPU=1
fi

echo "Launching training on ${NGPU} GPU(s)..."

torchrun --nnodes=1 \
  --nproc_per_node="${NGPU}" \
  --node_rank=0 \
  "${MAIN_PY}" \
  --config_file "${CONFIG_FILE}" \
  --runner_class_path runner/ge_trainer.py \
  --runner_class Trainer \
  --mode train

echo "Training launched. Check logs and checkpoints under the configured output_dir."


