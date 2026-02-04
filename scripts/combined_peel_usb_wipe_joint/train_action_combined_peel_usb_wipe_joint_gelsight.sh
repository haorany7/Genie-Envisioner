#!/bin/bash
#SBATCH --job-name="action_combined_peel_usb_wipe_joint_gelsight"
#SBATCH --output="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_action_combined_peel_usb_wipe_joint_gelsight/slurm-%j.out"
#SBATCH --error="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_action_combined_peel_usb_wipe_joint_gelsight/slurm-%j.err"
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --mem=240G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=bekg-delta-gpu
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH -t 48:00:00  # Action Combined Peel USB Wipe Joint Gelsight Training Time

# 创建必要的目录
mkdir -p /work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_action_combined_peel_usb_wipe_joint_gelsight
mkdir -p /work/hdd/behe/WORLD-MODEL-TOUCH/outputs/task_combined_peel_usb_wipe_joint_gelsight_action

echo "🚀 Starting WM-Touch Action Combined Peel USB Wipe Joint Gelsight Training"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner
bash scripts/train.sh main.py configs/ltx_model/combined_peel_usb_wipe_joint/action_model_combined_peel_usb_wipe_joint_gelsight.yaml