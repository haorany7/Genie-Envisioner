#!/bin/bash
#SBATCH --job-name="video_task_wipe"
#SBATCH --output="/projects/bfxb/haorany7/WORLD-MODEL-TOUCH/slurm_outputs/train_video_task_wipe/slurm-%j.out"
#SBATCH --error="/projects/bfxb/haorany7/WORLD-MODEL-TOUCH/slurm_outputs/train_video_task_wipe/slurm-%j.err"
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --mem=480G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=behe-delta-gpu
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH -t 48:00:00  # Video Task Wipe Training时间

echo "🚀 Starting WM-Touch Video Task Wipe Training"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner
bash scripts/train.sh main.py configs/ltx_model/task_wipe/video_model_task_wipe.yaml