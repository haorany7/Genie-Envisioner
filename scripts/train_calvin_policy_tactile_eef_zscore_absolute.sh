#!/bin/bash
#SBATCH --job-name="calvin_tactile_eef_zscore_abs"
#SBATCH --output="/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/slurm_outputs/calvin_tactile_eef_zscore_abs/slurm-%j.out"
#SBATCH --error="/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/slurm_outputs/calvin_tactile_eef_zscore_abs/slurm-%j.err"
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=behe-delta-gpu
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH -t 24:00:00

# ============================================================================
# 📌 Environment Setup
# ============================================================================
echo "🚀 Starting CALVIN Tactile EEF Z-Score Absolute Policy Training at $(date)"
echo "============================================================================"
echo "📍 Job ID: $SLURM_JOB_ID"
echo "📍 Node: $SLURM_NODELIST"
echo "📍 Working Directory: $(pwd)"
echo "============================================================================"

# Activate conda environment (avoid sourcing user .bashrc to prevent module issues)
if [ -f "/projects/behe/haorany7/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/projects/behe/haorany7/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "❌ ERROR: conda.sh not found. Please update the path in the script." >&2
  exit 1
fi
conda activate genie_envisioner

# Paths (avoid relying on BASH_SOURCE which changes under SLURM spool directories)
PROJ_ROOT="/projects/behe/haorany7/WORLD-MODEL-TOUCH/WM-Touch-Evaluation/GE-official/Genie-Envisioner"
SCRIPT_DIR="${PROJ_ROOT}/scripts"

MAIN_PY="${PROJ_ROOT}/main.py"
CONFIG_FILE="${PROJ_ROOT}/configs/ltx_model/policy_model_calvin_tactile_eef_zscore_absolute.yaml"

echo "📂 Project root   : ${PROJ_ROOT}"
echo "📄 Main script    : ${MAIN_PY}"
echo "📄 Config file    : ${CONFIG_FILE}"

if [ ! -f "${MAIN_PY}" ]; then
  echo "❌ ERROR: main.py not found at ${MAIN_PY}" >&2
  exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "❌ ERROR: config file not found at ${CONFIG_FILE}" >&2
  exit 1
fi

# Detect number of GPUs
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
else
  NGPU=1
fi

# Create slurm output directory if it doesn't exist
SLURM_OUT_DIR="/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/slurm_outputs/calvin_tactile_eef_zscore_abs"
mkdir -p "${SLURM_OUT_DIR}"

echo "============================================================================"
echo "🔥 Launching training on ${NGPU} GPU(s)..."
echo "============================================================================"

# Navigate to project root to ensure imports work correctly
cd "${PROJ_ROOT}"

torchrun --nnodes=1 \
  --nproc_per_node="${NGPU}" \
  --node_rank=0 \
  "${MAIN_PY}" \
  --config_file "${CONFIG_FILE}" \
  --runner_class_path runner/ge_trainer.py \
  --runner_class Trainer \
  --mode train

echo "============================================================================"
echo "✅ Training launched. Check logs and checkpoints under the configured output_dir."
echo "✅ Training completed at $(date)"
echo "============================================================================"
