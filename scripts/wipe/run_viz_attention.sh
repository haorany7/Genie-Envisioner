#!/bin/bash
#SBATCH --job-name="viz_attn_wipe"
#SBATCH --output="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/viz_attn/slurm-%j.out"
#SBATCH --error="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/viz_attn/slurm-%j.err"
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --account=behe-delta-gpu
#SBATCH -t 02:00:00

echo "🚀 Starting Attention Visualization Video Generation"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner

cd /projects/behe/haorany7/WORLD-MODEL-TOUCH/Reimplementation/Genie-Envisioner
export PYTHONPATH=$PYTHONPATH:.

# 创建日志目录
mkdir -p /work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/viz_attn

# 运行可视化脚本
# n_validation: 处理 1 个 episode
# n_chunk_action: 连续运行 20 个推理块（长程推理）
python scripts/visualize_attention.py \
    --config_file configs/ltx_model/wipe/action_model_wipe_gelsight.yaml \
    --checkpoint_path /work/hdd/behe/WORLD-MODEL-TOUCH/outputs/wipe_gelsight_action/2026_01_11_03_24_56/step_10000/diffusion_pytorch_model.safetensors \
    --n_validation 100 \
    --n_chunk_action 20 \
    --output_path /work/hdd/behe/WORLD-MODEL-TOUCH/outputs/attention_viz/wipe_gelsight_step10000

echo "✅ Visualization complete. Check results in /work/hdd/behe/WORLD-MODEL-TOUCH/outputs/attention_viz/wipe_gelsight_step10000"
