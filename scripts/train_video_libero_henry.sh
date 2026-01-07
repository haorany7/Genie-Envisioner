#!/bin/bash
#SBATCH --job-name="video_libero"
#SBATCH --output="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_video_libero/henry/slurm-%j.out"
#SBATCH --error="/work/hdd/behe/WORLD-MODEL-TOUCH/slurm_outputs/train_video_libero/henry/slurm-%j.err"
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --mem=480G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=closest
#SBATCH --account=bdpp-delta-gpu
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH -t 00:30:00  # Video Libero Training Time

echo "🚀 Starting WM-Touch Video Libero Training"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Time: $(date)"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1


# Environment Setup
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner
bash scripts/train_henry.sh main.py configs/ltx_model/libero/video_model_libero_henry.yaml