import pandas as pd
import numpy as np
import PIL.Image
import io
import os
import matplotlib.pyplot as plt
import argparse
import imageio
import glob
import random
import json

def load_task_map(data_root):
    """Load task_index to task_string mapping from meta/tasks.jsonl"""
    task_map = {}
    tasks_path = os.path.join(data_root, "meta", "tasks.jsonl")
    if os.path.exists(tasks_path):
        with open(tasks_path, "r") as f:
            for line in f:
                item = json.loads(line)
                task_map[item["task_index"]] = item["task"]
    return task_map

def process_single_episode(parquet_path, output_dir, fps, task_map):
    ep_id = os.path.basename(parquet_path).split(".")[0]
    print(f"\n>>> Processing {ep_id} from {parquet_path}...")
    
    # 1. Load Data
    df = pd.read_parquet(parquet_path)
    
    # Get Task Instruction
    instruction = "Unknown"
    if 'task_index' in df.columns:
        t_idx = df['task_index'].iloc[0]
        instruction = task_map.get(t_idx, f"Index {t_idx}")
    elif 'task' in df.columns:
        instruction = df['task'].iloc[0]
    
    print(f"  Task Instruction: {instruction}")

    # 2. Video Visualization
    cams = ["camera1", "camera2", "gelsight"]
    available_cams = [c for c in cams if c in df.columns]
    
    if available_cams:
        video_path = os.path.join(output_dir, f"{ep_id}_video.mp4")
        first_cam_data = df[available_cams[0]].iloc[0]
        first_img_bytes = first_cam_data['bytes'] if isinstance(first_cam_data, dict) else first_cam_data
        first_img = PIL.Image.open(io.BytesIO(first_img_bytes))
        target_h = first_img.height
        
        print(f"  Generating H.264 video (fps={fps})...")
        try:
            writer = imageio.get_writer(video_path, fps=fps, codec='libx264', pixelformat='yuv420p')
            for i in range(len(df)):
                frame_imgs = []
                for cam in available_cams:
                    img_data = df[cam].iloc[i]
                    img_bytes = img_data['bytes'] if isinstance(img_data, dict) else img_data
                    img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    if img.height != target_h:
                        aspect_ratio = img.width / img.height
                        new_w = int(target_h * aspect_ratio)
                        img = img.resize((new_w, target_h), PIL.Image.Resampling.LANCZOS)
                    frame_imgs.append(np.array(img))
                combined_frame = np.hstack(frame_imgs)
                writer.append_data(combined_frame)
            writer.close()
            print(f"  Video saved: {video_path}")
        except Exception as e:
            print(f"  Error generating video: {e}")

    # 3. Action/State Curves Visualization
    print("  Plotting action and state curves...")
    def extract_array(col_data):
        return np.stack([np.array(x).flatten() for x in col_data])

    actions = extract_array(df['actions'])
    state = extract_array(df['state'])
    is_identical = np.array_equal(actions, state)
    status_str = "(IDENTICAL)" if is_identical else "(DIFFERENT)"
    
    n_dims = min(8, actions.shape[1], state.shape[1])
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    # 将 Instruction 放在主标题中
    fig.suptitle(f"Trajectory Check: {ep_id} {status_str}\nInstruction: {instruction}\nBlue: Action (GT) | Red Dash: State (Input)", fontsize=16)
    
    dim_names = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Joint 6", "Joint 7", "Gripper"]
    for i in range(n_dims):
        ax = axes[i//2, i%2]
        ax.plot(actions[:, i], label='Action', color='blue', alpha=0.8, linewidth=2.5)
        ax.plot(state[:, i], label='State', color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.set_title(dim_names[i])
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        
        diff = np.abs(np.diff(actions[:, i]))
        if len(diff) > 0 and np.max(diff) > 1.0:
            ax.set_facecolor('#ffeeee')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, f"{ep_id}_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  Comparison plot saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--parquet_path", type=str, help="Path to a single episode parquet file")
    group.add_argument("--data_root", type=str, help="Root directory of the dataset to sample from")
    
    parser.add_argument("--output_dir", type=str, default="visualization_results")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random episodes to sample if data_root is used")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定数据根目录以便加载 meta 数据
    if args.data_root:
        data_root = args.data_root
    else:
        # 如果只传了 parquet_path，尝试向上找两级
        data_root = os.path.dirname(os.path.dirname(os.path.dirname(args.parquet_path)))
    
    task_map = load_task_map(data_root)
    
    if args.parquet_path:
        process_single_episode(args.parquet_path, args.output_dir, args.fps, task_map)
    else:
        search_pattern = os.path.join(args.data_root, "**", "*.parquet")
        all_files = glob.glob(search_pattern, recursive=True)
        all_files = sorted(all_files)
        
        if not all_files:
            print(f"Error: No .parquet files found in {args.data_root}")
            return
            
        print(f"Found {len(all_files)} total episodes. Sampling {args.num_samples}...")
        sampled_files = random.sample(all_files, min(len(all_files), args.num_samples))
        
        for pq in sorted(sampled_files):
            process_single_episode(pq, args.output_dir, args.fps, task_map)
            
    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()
