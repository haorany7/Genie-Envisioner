#!/bin/bash

# 创建必要的目录
mkdir -p /data/vtam/slurm_outputs/train_action_task_wipe_wrist_gelsight

echo "🚀 Starting WM-Touch Action Task Wipe Wrist Gelsight Training (Manual Mode)"
echo "============================================================="
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner

# 指定 GPU 并启动训练，同时记录日志
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh main.py \
    configs/ltx_model/task_wipe_wrist/action_model_task_wipe_wrist_gelsight_xinzhuo.yaml \
    2>&1 | tee /data/vtam/slurm_outputs/train_action_task_wipe_wrist_gelsight/manual_run_$(date +%Y%m%d_%H%M%S).log
