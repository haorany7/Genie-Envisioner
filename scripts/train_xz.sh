#!/usr/bin/bash


script_path=${1}
echo $script_path

config_path=${2}
echo $config_path

# Use environment MASTER_PORT or default to 29500
MASTER_PORT=${MASTER_PORT:-29500}

if [ -z $WORLD_SIZE ]; then
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
# Count visible GPUs from CUDA_VISIBLE_DEVICES (e.g. "0,1,2,3")
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
NGPU=`nvidia-smi --list-gpus | wc -l`
fi
echo "Training on 1 Nodes, $NGPU GPUs using MASTER_PORT=$MASTER_PORT"
torchrun --nnodes=1 \
    --nproc_per_node=$NGPU \
    --master-port=$MASTER_PORT \
    --node_rank=0 \
    $script_path \
    --config_file $config_path
else
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
NGPU=`nvidia-smi --list-gpus | wc -l`
fi
echo "Training on $WORLD_SIZE Nodes, $NGPU GPU per Node"
torchrun --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NGPU \
    --node_rank=$RANK \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    $script_path \
    --config_file $config_path
fi
