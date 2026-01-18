#!/bin/bash
#SBATCH --job-name="action_task_wipe_wrist"
#SBATCH --output="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_action_task_wipe_wrist/slurm-%j.out"
#SBATCH --error="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_action_task_wipe_wrist/slurm-%j.err"
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --mem=240G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=bdpp-delta-gpu
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH -t 48:00:00  # Action Task Wipe Wrist Training Time

# 创建必要的目录
mkdir -p /work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_action_task_wipe_wrist
mkdir -p /work/nvme/behe/WORLD-MODEL-TOUCH/outputs/task_wipe_wrist_action

echo "🚀 Starting WM-Touch Action Task Wipe Wrist Training"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner
bash scripts/train.sh main.py configs/ltx_model/task_wipe_wrist/action_model_task_wipe_wrist.yaml