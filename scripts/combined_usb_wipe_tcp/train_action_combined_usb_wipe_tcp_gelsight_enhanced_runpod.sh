# 创建必要的目录
mkdir -p /workspace/vtam/slurm_outputs/train_action_combined_usb_wipe_tcp_gelsight_enhanced
mkdir -p /workspace/vtam/outputs/task_combined_usb_wipe_tcp_gelsight_enhanced_action

echo "🚀 Starting WM-Touch Action Combined USB Wipe TCP Gelsight Enhanced Training"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Time: $(date)"

# 环境设置
source /workspace/venv_ge/bin/activate
export HF_HOME=/workspace/tmp/huggingface_cache
export TMPDIR=/workspace/tmp/pip_tmp
bash scripts/train.sh main.py configs/ltx_model/combined_usb_wipe_tcp/action_model_combined_usb_wipe_tcp_gelsight_enhanced_runpod.yaml