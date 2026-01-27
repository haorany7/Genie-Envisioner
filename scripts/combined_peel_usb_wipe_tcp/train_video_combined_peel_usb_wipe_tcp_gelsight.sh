#!/bin/bash
#SBATCH --job-name="video_combined_peel_usb_wipe_tcp_gelsight"
#SBATCH --output="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_video_combined_peel_usb_wipe_tcp_gelsight/slurm-%j.out"
#SBATCH --error="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_video_combined_peel_usb_wipe_tcp_gelsight/slurm-%j.err"
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
#SBATCH -t 48:00:00

echo "🚀 Starting WM-Touch Video Combined Peel USB Wipe TCP Gelsight Training"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner

# Make sure log dir exists
mkdir -p /work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_video_combined_peel_usb_wipe_tcp_gelsight

# Make sure we run from the GE repo root
cd /projects/behe/haorany7/WORLD-MODEL-TOUCH/Reimplementation/Genie-Envisioner
bash scripts/train.sh main.py configs/ltx_model/combined_peel_usb_wipe_tcp/video_model_combined_peel_usb_wipe_tcp_gelsight.yaml
