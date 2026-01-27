# 创建必要的目录
mkdir -p /workspace/vtam/slurm_outputs/train_action_combined_usb_wipe_tcp_gelsight
mkdir -p /workspace/vtam/outputs/task_combined_usb_wipe_tcp_gelsight_action

echo "🚀 Starting WM-Touch Action Combined USB Wipe TCP Gelsight Training"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner
bash scripts/train.sh main.py configs/ltx_model/combined_usb_wipe_tcp/action_model_combined_usb_wipe_tcp_gelsight_runpod.yaml