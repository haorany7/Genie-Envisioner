#!/bin/bash
#SBATCH --job-name="infer_action_wipe_gelsight"
#SBATCH --output="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/infer_action_wipe_gelsight/slurm-%j.out"
#SBATCH --error="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/infer_action_wipe_gelsight/slurm-%j.err"
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --account=behe-delta-gpu
#SBATCH -t 02:00:00

echo "🚀 Starting WM-Touch Action Wipe Gelsight Inference"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner

# 创建输出目录
mkdir -p /work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/infer_action_wipe_gelsight
mkdir -p /work/hdd/behe/WORLD-MODEL-TOUCH/outputs/wipe_gelsight_action/2026_01_11_03_24_56/infer_step_10000

# 运行推理
# 参数说明: scripts/infer.sh <script_path> <config_path> <ckp_path> <output_path> <domain_name>
bash scripts/infer.sh \
  main.py \
  configs/ltx_model/wipe/action_model_wipe_gelsight.yaml \
  /work/hdd/behe/WORLD-MODEL-TOUCH/outputs/wipe_gelsight_action/2026_01_11_03_24_56/step_10000/diffusion_pytorch_model.safetensors \
  /work/hdd/behe/WORLD-MODEL-TOUCH/outputs/wipe_gelsight_action/2026_01_11_03_24_56/infer_step_10000 \
  wipe_lerobot
