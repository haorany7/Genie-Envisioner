#!/bin/bash
#SBATCH --job-name="video_task_wipe"
#SBATCH --output="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_video_task_wipe/slurm-%j.out"
#SBATCH --error="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_video_task_wipe/slurm-%j.err"
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --mem=480G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=closest
#SBATCH --account=bfxb-delta-gpu
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH -t 24:00:00

echo "🚀 Starting WM-Touch Single-Node Video Task Wipe Training (H200x8)"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES ($SLURM_NODELIST)"
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner

# 单节点不需要额外设置 WORLD_SIZE 和 RANK
# scripts/train.sh 会自动检测并配置单机 8-GPU 训练

# 启动训练
bash scripts/train.sh main.py configs/ltx_model/task_wipe/video_model_task_wipe.yaml
