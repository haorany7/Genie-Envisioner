#!/usr/bin/bash
# Defaults; can be overridden by positional args
script_path=${1:-/projects/behe/haorany7/WORLD-MODEL-TOUCH/WM-Touch-Evaluation/GE-official/Genie-Envisioner/main.py}
echo "main.py: ${script_path}"

config_path=${2:-/projects/behe/haorany7/WORLD-MODEL-TOUCH/WM-Touch-Evaluation/GE-official/Genie-Envisioner/configs/ltx_model/policy_model_calvin_tactile_joint.yaml}
echo "config : ${config_path}"
ckp_path=${3:-/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/ckpt_from_zzy/calvin_pure_vision_policy_12k_action_chunk_9/diffusion_pytorch_model_policy.safetensors}
echo "ckpt   : ${ckp_path}"

DEFAULT_WORK_ROOT="/work/hdd/bche/haorany7/WORLD-MODEL-TOUCH"
WORK_ROOT="${WM_WORK_ROOT:-${DEFAULT_WORK_ROOT}}"

output_path=${4:-${WORK_ROOT}/outputs/calvin_policy_pure_vision_joint_infer/$(date +%Y%m%d_%H%M%S)}
domain_name=${5:-ABC_lerobot_joint_action}
n_validation=${6:-34}
n_chunk_action=${7:-9}
tasks_per_run=${8:-34}
episodes_per_task=${9:-1}
rollout_steps=${10:-65}
echo "output : ${output_path}"
echo "domain : ${domain_name}"
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
    $script_path \
    --runner_class_path runner/ge_inferencer.py \
    --runner_class Inferencer \
    --config_file ${config_path:-/projects/behe/haorany7/WORLD-MODEL-TOUCH/WM-Touch-Evaluation/GE-official/Genie-Envisioner/configs/ltx_model/policy_model_calvin_pure_vision_joint.yaml} \
    --mode infer \
    --checkpoint_path $ckp_path \
    --output_path $output_path \
    --n_validation $n_validation \
    --n_chunk_action $n_chunk_action \
    --domain_name $domain_name \
    --tasks_per_run $tasks_per_run \
    --episodes_per_task $episodes_per_task \
    --rollout_steps $rollout_steps


