#!/bin/bash
#SBATCH --job-name="video_task_wipe"
#SBATCH --output="/projects/bfxb/haorany7/WORLD-MODEL-TOUCH/slurm_outputs/train_video_task_wipe/slurm-%j.out"
#SBATCH --error="/projects/bfxb/haorany7/WORLD-MODEL-TOUCH/slurm_outputs/train_video_task_wipe/slurm-%j.err"
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
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

echo "🚀 Starting WM-Touch 2-Node Video Task Wipe Training"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES ($SLURM_NODELIST)"
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner

# 多节点分布式配置
export WORLD_SIZE=$SLURM_JOB_NUM_NODES
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# 使用 srun 在两个节点上启动训练
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
    bash -c "export RANK=\$SLURM_NODEID; bash scripts/train.sh main.py configs/ltx_model/task_wipe/video_model_task_wipe.yaml"
