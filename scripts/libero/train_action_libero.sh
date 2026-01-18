#!/bin/bash
#SBATCH --job-name="action_libero"
#SBATCH --output="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_action_libero/slurm-%j.out"
#SBATCH --error="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_action_libero/slurm-%j.err"
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --mem=240G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=behe-delta-gpu
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH -t 48:00:00  # Action Wipe Training Time

echo "🚀 Starting WM-Touch Action Libero Training"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner

# 确保目录存在
mkdir -p /work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_action_libero
mkdir -p /work/nvme/behe/WORLD-MODEL-TOUCH/outputs/libero_action

# 切换到项目根目录
cd /projects/behe/haorany7/WORLD-MODEL-TOUCH/Reimplementation/Genie-Envisioner

bash scripts/train.sh main.py configs/ltx_model/libero/action_model_libero.yaml
