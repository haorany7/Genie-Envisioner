#!/bin/bash
#SBATCH --job-name="libero_eef_zscore_abs_ac36"
#SBATCH --output="/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/slurm_outputs/libero_eef_zscore_abs_action_chunk_36/slurm-%j.out"
#SBATCH --error="/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/slurm_outputs/libero_eef_zscore_abs_action_chunk_36/slurm-%j.err"
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
#SBATCH -t 48:00:00

# ============================================================================
# 📌 Environment Setup
# ============================================================================
echo "🚀 Starting Libero EEF Z-Score Absolute Policy Training (Action Chunk 36) at $(date)"
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
CONFIG_FILE="${PROJ_ROOT}/configs/ltx_model/policy_model_libero_eef_zscore_absolute.yaml"

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
echo "----------------------------------------------------------------------------"
echo "🔎 Debugging GPU Environment"
echo "----------------------------------------------------------------------------"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  NGPU=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
else
  echo "⚠️ nvidia-smi not found! Assuming 1 GPU."
  NGPU=1
fi
echo "Detected ${NGPU} GPUs."

# Ensure CUDA_VISIBLE_DEVICES is set correctly if it's empty but we have GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ] && [ "$NGPU" -gt 0 ]; then
    # Generate sequence 0,1,2...N-1
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($NGPU - 1)))
    echo "Set CUDA_VISIBLE_DEVICES to $CUDA_VISIBLE_DEVICES"
fi

# Create slurm output directory if it doesn't exist
SLURM_OUT_DIR="/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/slurm_outputs/libero_eef_zscore_abs_action_chunk_36"
mkdir -p "${SLURM_OUT_DIR}"

echo "============================================================================"
echo "🔥 Launching Libero policy training on ${NGPU} GPU(s)..."
echo "============================================================================"

# Navigate to project root to ensure imports work correctly
cd "${PROJ_ROOT}"

# Resume from latest checkpoint in existing run folder
torchrun --nnodes=1 \
  --nproc_per_node="${NGPU}" \
  --node_rank=0 \
  "${MAIN_PY}" \
  --config_file "${CONFIG_FILE}" \
  --runner_class_path runner/ge_trainer.py \
  --runner_class Trainer \
  --mode train \
  --resume \
  --sub_folder "2025_12_10_19_30_59"

echo "============================================================================"
echo "✅ Libero policy training launched. Check logs and checkpoints under the configured output_dir."
echo "✅ Training completed at $(date)"
echo "============================================================================"


