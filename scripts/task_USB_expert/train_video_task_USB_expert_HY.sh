#!/bin/bash
#SBATCH --job-name="video_task_usb_expert"
#SBATCH --output="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_video_task_usb_expert/slurm-%j.out"
#SBATCH --error="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_video_task_usb_expert/slurm-%j.err"
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
#SBATCH -t 48:00:00  # Video USB Expert Training时间

echo "🚀 Starting WM-Touch Video Task USB Expert Training"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner

# Make sure log dir exists
mkdir -p /work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_video_task_usb_expert

# Make sure we run from the GE repo root
cd /projects/behe/haorany7/WORLD-MODEL-TOUCH/Reimplementation/Genie-Envisioner
bash scripts/train.sh main.py configs/ltx_model/task_USB_expert/video_model_task_USB_expert.yaml