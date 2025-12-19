#!/bin/bash
#SBATCH --job-name="calvin_tactile_eef_zscore_abs_infer_ac18"
#SBATCH --output="/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/slurm_outputs/calvin_tactile_eef_zscore_abs_infer_ac18/slurm-%j.out"
#SBATCH --error="/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/slurm_outputs/calvin_tactile_eef_zscore_abs_infer_ac18/slurm-%j.err"
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=behe-delta-gpu
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH -t 24:00:00

# ----------------------------------------------------------------------------
# 1. Environment Setup (Critical for SLURM)
# ----------------------------------------------------------------------------
# Hardcode project root to avoid issues with BASH_SOURCE in spool directories
PROJ_ROOT="/projects/behe/haorany7/WORLD-MODEL-TOUCH/WM-Touch-Evaluation/GE-official/Genie-Envisioner"

# Initialize Conda (robust method)
if [ -f "/anaconda/etc/profile.d/conda.sh" ]; then
    source /anaconda/etc/profile.d/conda.sh
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
else
    # Fallback, though .bashrc often causes Lmod issues on compute nodes
    source ~/.bashrc
fi

# Activate environment
conda activate genie_envisioner
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'genie_envisioner'"
    exit 1
fi

# Go to project root
cd "$PROJ_ROOT" || { echo "❌ Could not cd to $PROJ_ROOT"; exit 1; }
echo "📍 Working Directory: $(pwd)"

# ----------------------------------------------------------------------------
# 2. Arguments & Execution
# ----------------------------------------------------------------------------
# Defaults; override via positional args if needed.
script_path=${1:-${PROJ_ROOT}/main.py}
echo "main.py: ${script_path}"

config_path=${2:-/projects/behe/haorany7/WORLD-MODEL-TOUCH/WM-Touch-Evaluation/GE-official/Genie-Envisioner/configs/ltx_model/policy_model_calvin_tactile_eef_zscore_absolute_action_chunk_18.yaml}
echo "config : ${config_path}"

ckp_path=${3:-/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/outputs/calvin_tactile_eef_zscore_abs_policy_action_chunk_18/2025_12_05_06_08_15/step_6000/diffusion_pytorch_model.safetensors}
echo "ckpt   : ${ckp_path}"

DEFAULT_WORK_ROOT="/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH"
WORK_ROOT="${WM_WORK_ROOT:-${DEFAULT_WORK_ROOT}}"

output_path=${4:-${WORK_ROOT}/outputs/calvin_policy_tactile_eef_infer_action_chunk_18/$(date +%Y%m%d_%H%M%S)}
domain_name=${5:-ABC_lerobot}
statistics_domain=${6:-ABC_lerobot}
n_validation=${7:-34}
n_chunk_action=${8:-18}
tasks_per_run=${9:-34}
episodes_per_task=${10:-10}
rollout_steps=${11:-52}
echo "output : ${output_path}"
echo "domain : ${domain_name}"
echo "stat_dom: ${statistics_domain}"
echo "n_val  : ${n_validation}"
echo "n_chunk: ${n_chunk_action}"
echo "tasks  : ${tasks_per_run}"
echo "epis   : ${episodes_per_task}"
echo "rollout: ${rollout_steps}"

mkdir -p "$output_path"

echo "Inference on 1 Nodes, 1 GPUs"
torchrun --nnodes=1 \
    --nproc_per_node=1 \
    --node_rank=0 \
    "$script_path" \
    --runner_class_path runner/ge_inferencer.py \
    --runner_class Inferencer \
    --config_file "$config_path" \
    --mode infer \
    --checkpoint_path "$ckp_path" \
    --output_path "$output_path" \
    --n_validation "$n_validation" \
    --n_chunk_action "$n_chunk_action" \
    --domain_name "$domain_name" \
    --statistics_domain "$statistics_domain" \
    --tasks_per_run "$tasks_per_run" \
    --episodes_per_task "$episodes_per_task" \
    --rollout_steps "$rollout_steps"


