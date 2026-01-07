#!/usr/bin/bash


script_path=${1}
echo $script_path

config_path=${2}
echo $config_path

NGPU=$(nvidia-smi --list-gpus | wc -l)
echo "Training on 1 Nodes, $NGPU GPUs"

torchrun --standalone \
  --nproc_per_node=$NGPU \
  --tee 3 \
  --log_dir torchrun_logs_${SLURM_JOB_ID:-manual} \
  $script_path \
  --config_file $config_path
