#!/bin/bash
#SBATCH --job-name="infer_action_task_wipe_wrist"
#SBATCH --output="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/infer_action_task_wipe_wrist/slurm-%j.out"
#SBATCH --error="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/infer_action_task_wipe_wrist/slurm-%j.err"
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --account=behe-delta-gpu
#SBATCH -t 02:00:00

echo "🚀 Starting WM-Touch Action Task Wipe Wrist Inference"
echo "============================================================="
echo "Time: $(date)"

# 环境设置
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate genie_envisioner

# 修复环境路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=121 
export BITSANDBYTES_NOWELCOME=1

# 运行推理
# 确保反斜杠后面没有任何空格，且最后一行没有反斜杠
bash scripts/infer.sh \
  main.py \
  configs/ltx_model/task_wipe_wrist/action_model_task_wipe_wrist_deployment.yaml \
  checkpoints/task_wipe_wrist_action/2026_01_18_08_51_42/step_20000/diffusion_pytorch_model.safetensors \
  eval_results_ge_20000 \
  task_wipe_wrist_lerobot
