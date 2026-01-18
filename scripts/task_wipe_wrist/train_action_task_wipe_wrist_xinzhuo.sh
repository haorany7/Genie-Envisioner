#!/bin/bash

# 创建必要的目录
mkdir -p /data/vtam/slurm_outputs/train_action_task_wipe_wrist

echo "🚀 Starting WM-Touch Action Task Wipe Wrist Training (Manual Mode)"
echo "============================================================="
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
CONDA_SH=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null)"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
  fi
fi
if [ -z "$CONDA_SH" ]; then
  if [ -n "$CONDA_EXE" ]; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
      CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
    fi
  fi
fi
if [ -z "$CONDA_SH" ]; then
  if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/miniconda/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
fi
if [ -n "$CONDA_SH" ]; then
  source "$CONDA_SH"
else
  echo "⚠️ Warning: conda.sh not found; conda activate may fail."
fi
conda activate genie_envisioner

# 指定 GPU 并启动训练，同时记录日志
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export MASTER_PORT=${MASTER_PORT:-29501}
# Work around NCCL collective hang on this node by disabling P2P.
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
nvidia-smi -L || true
bash scripts/train_xz.sh main.py \
    configs/ltx_model/task_wipe_wrist/action_model_task_wipe_wrist_xinzhuo.yaml \
    2>&1 | tee /data/vtam/slurm_outputs/train_action_task_wipe_wrist/manual_run_$(date +%Y%m%d_%H%M%S).log
