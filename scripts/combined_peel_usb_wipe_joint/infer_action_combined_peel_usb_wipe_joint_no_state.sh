#!/bin/bash
#SBATCH --job-name="infer_action_combined_peel_usb_wipe_joint_no_state"
#SBATCH --output="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/slurm_outputs/infer_action_combined_peel_usb_wipe_joint_no_state/slurm-%j.out"
#SBATCH --error="/home/yuchenmo/Desktop/VLA/Genie-Envisioner/slurm_outputs/infer_action_combined_peel_usb_wipe_joint_no_state/slurm-%j.err"
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --account=behe-delta-gpu
#SBATCH -t 02:00:00

echo "🚀 Starting WM-Touch Action Combined Peel+USB+Wipe Joint Inference (no state)"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Time: $(date)"

# 环境设置
source ~/.bashrc || echo "⚠️ Warning: bashrc loading had issues, continuing..."
conda activate genie_envisioner

# 运行推理
# 参数说明: scripts/infer.sh <script_path> <config_path> <ckp_path> <output_path> <domain_name>
bash scripts/infer.sh \
  main.py \
  configs/ltx_model/combined_peel_usb_wipe_joint/action_model_combined_peel_usb_wipe_joint_no_state_deployment.yaml \
  /home/yuchenmo/Desktop/VLA/Genie-Envisioner/checkpoints/task_combined_peel_usb_wipe_joint_action_no_state_xinzhuo/2026_02_04_11_18_06/step_10000/diffusion_pytorch_model.safetensors \
  /home/yuchenmo/Desktop/VLA/Genie-Envisioner/eval_results/task_combined_peel_usb_wipe_joint_action_no_state_xinzhuo/2026_02_04_11_18_06/infer_step_10000 \
  combined_peel_usb_wipe_joint \
  --random_n_validation 20
