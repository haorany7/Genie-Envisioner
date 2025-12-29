import pandas as pd
import numpy as np
import cv2
import PIL.Image
import io
import os
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to the episode parquet file")
    parser.add_argument("--output_dir", type=str, default="visualization_results")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ep_id = os.path.basename(args.parquet_path).split(".")[0]
    
    # 1. Load Data
    print(f"Loading {args.parquet_path}...")
    df = pd.read_parquet(args.parquet_path)
    
    # 2. Video Visualization (Image -> Video)
    cams = ["camera1", "camera2", "gelsight"]
    available_cams = [c for c in cams if c in df.columns]
    
    if available_cams:
        video_path = os.path.join(args.output_dir, f"{ep_id}_video.mp4")
        # Get first frame for dimensions
        sample_img_data = df[available_cams[0]].iloc[0]
        img_bytes = sample_img_data['bytes'] if isinstance(sample_img_data, dict) else sample_img_data
        sample_img = PIL.Image.open(io.BytesIO(img_bytes))
        w, h = sample_img.size
        combined_w = w * len(available_cams)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, args.fps, (combined_w, h))
        
        print(f"Generating video to {video_path}...")
        for i in range(len(df)):
            frame_imgs = []
            for cam in available_cams:
                img_data = df[cam].iloc[i]
                img_bytes = img_data['bytes'] if isinstance(img_data, dict) else img_data
                img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_np = np.array(img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                frame_imgs.append(img_bgr)
            
            combined_frame = np.hstack(frame_imgs)
            out.write(combined_frame)
        out.release()
        print(f"Video saved: {video_path}")

    # 3. Action/State Curves Visualization (Mutation Check)
    print("Plotting action/state curves...")
    # Extract arrays
    actions = np.stack([np.array(x).flatten() for x in df['actions']])
    state = np.stack([np.array(x).flatten() for x in df['state']])
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle(f"Trajectory Check: {ep_id}\nBlue: Action | Orange Dash: State", fontsize=16)
    
    dim_names = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Joint 6", "Joint 7", "Gripper"]
    
    for i in range(min(8, actions.shape[1])):
        ax = axes[i//2, i%2]
        ax.plot(actions[:, i], label='Action', color='#1f77b4', alpha=0.8, linewidth=2)
        ax.plot(state[:, i], label='State', color='#ff7f0e', linestyle='--', alpha=0.8, linewidth=2)
        ax.set_title(dim_names[i])
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Check for mutation/jumps
        diff = np.abs(np.diff(actions[:, i]))
        if len(diff) > 0 and np.max(diff) > 1.0:
            ax.set_facecolor('#ffeeee')
            print(f"  ⚠️ Large jump detected in {dim_names[i]} (max diff: {np.max(diff):.4f})")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(args.output_dir, f"{ep_id}_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Curves plot saved: {plot_path}")

if __name__ == "__main__":
    main()

