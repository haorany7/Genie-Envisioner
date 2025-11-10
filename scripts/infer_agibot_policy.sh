#!/usr/bin/bash

# Defaults; can be overridden by positional args
script_path=${1:-/projects/behe/haorany7/WORLD-MODEL-TOUCH/WM-Touch-Evaluation/GE-official/Genie-Envisioner/main.py}
echo "main.py: ${script_path}"

config_path=${2:-/projects/behe/haorany7/WORLD-MODEL-TOUCH/WM-Touch-Evaluation/GE-official/Genie-Envisioner/configs/ltx_model/policy_model_lerobot.yaml}
echo "config : ${config_path}"
ckp_path=${3:-/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/outputs/agibot_policy/20251109_053633/2025_11_09_05_38_52/step_3000/diffusion_pytorch_model.safetensors}
echo "ckpt   : ${ckp_path}"

DEFAULT_WORK_ROOT="/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH"
WORK_ROOT="${WM_WORK_ROOT:-${DEFAULT_WORK_ROOT}}"

output_path=${4:-${WORK_ROOT}/outputs/agibot_policy_infer/$(date +%Y%m%d_%H%M%S)}
domain_name=${5:-agibotworld}
n_validation=${6:-10}
n_chunk_action=${7:-10}
tasks_per_run=${8:-0}
episodes_per_task=${9:-30}
echo "output : ${output_path}"
echo "domain : ${domain_name}"
echo "n_val  : ${n_validation}"
echo "n_chunk: ${n_chunk_action}"
echo "tasks  : ${tasks_per_run}"
echo "epis   : ${episodes_per_task}"

mkdir -p "$output_path"

echo "Inference on 1 Nodes, 1 GPUs"
torchrun --nnodes=1 \
    --nproc_per_node=1 \
    --node_rank=0 \
    $script_path \
    --runner_class_path runner/ge_inferencer.py \
    --runner_class Inferencer \
    --config_file $config_path \
    --mode infer \
    --checkpoint_path $ckp_path \
    --output_path $output_path \
    --n_validation $n_validation \
    --n_chunk_action $n_chunk_action \
    --domain_name $domain_name \
    --tasks_per_run $tasks_per_run \
    --episodes_per_task $episodes_per_task


