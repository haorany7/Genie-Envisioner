#!/usr/bin/bash

script_path=${1}
echo $script_path

config_path=${2}
echo $config_path
ckp_path=${3}
output_path=${4}
domain_name=${5}
extra_args=("${@:6}")

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
    --n_validation 100 \
    --n_chunk_action 10 \
    --domain_name $domain_name \
    "${extra_args[@]}"

