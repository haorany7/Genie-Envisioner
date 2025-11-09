#!/usr/bin/bash

# Defaults; can be overridden by positional args
script_path=${1:-/projects/behe/haorany7/WORLD-MODEL-TOUCH/WM-Touch-Evaluation/GE-official/Genie-Envisioner/main.py}
echo $script_path

config_path=${2:-/projects/behe/haorany7/WORLD-MODEL-TOUCH/WM-Touch-Evaluation/GE-official/Genie-Envisioner/configs/ltx_model/video_model_infer_fast_lerobot.yaml}
echo $config_path
ckp_path=${3:-/projects/behe/haorany7/WORLD-MODEL-TOUCH/pretrained_models/genie_envisioner/GE_base_fast_v0.1.safetensors}
output_path=${4:-/projects/behe/haorany7/WORLD-MODEL-TOUCH/outputs/agibot_infer/$(date +%Y%m%d_%H%M%S)}
domain_name=${5:-Agibot_lerobot}

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
    --n_validation 1 \
    --n_chunk_action 10 \
    --domain_name $domain_name


