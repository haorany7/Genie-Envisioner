import cv2
import numpy as np
import argparse
import os
import sys
import imageio

# 确保能找到 utils
sys.path.append(os.getcwd())
from utils.tactile_utils import TactileForceExtractor

def run_extraction(video_path, output_path="gelsight_force_xyz_viz.mp4"):
    print(f"🚀 启动力场提取流程...")
    print(f"📂 输入文件: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: 无法打开视频 {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 视频信息: {w}x{h}, {fps} FPS, 共 {total_frames} 帧")

    ret, first_frame = cap.read()
    if not ret:
        print("❌ 无法读取第一帧进行初始化")
        return
    
    ref_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    print("⚙️ 正在初始化 Delaunay 拓扑结构...")
    try:
        extractor = TactileForceExtractor(ref_gray, margin=15)
        print(f"✅ 初始化成功: 找到 {len(extractor.ref_dots)} 个标志点")
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 使用 imageio + libx264 以确保 Cursor 预览兼容性
    print(f"🎬 准备写入 H.264 编码视频: {output_path}")
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')

    frame_idx = 0
    print("⏳ 开始逐帧处理 (使用 libx264 编码)...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 提取力场核心逻辑
            ref_dots, force_xy, force_z = extractor.process_frame(curr_gray)
            
            # 可视化增强
            viz_frame = frame.copy()
            for i in range(len(ref_dots)):
                # XY 剪切力箭头 (放大 8 倍)
                start = (int(ref_dots[i, 0]), int(ref_dots[i, 1]))
                end = (int(ref_dots[i, 0] + force_xy[i, 0] * 8), 
                       int(ref_dots[i, 1] + force_xy[i, 1] * 8))
                
                z_val = force_z[i]
                # 过滤背景抖动阈值
                if abs(z_val) > 0.5 or np.linalg.norm(force_xy[i]) > 1.0:
                    # 红色代表压缩(Z+)，蓝色代表拉伸(Z-)
                    color = (0, 0, 255) if z_val < 0 else (255, 0, 0)
                    thickness = int(np.clip(abs(z_val) * 1.5, 1, 6))
                    cv2.arrowedLine(viz_frame, start, end, color, thickness, tipLength=0.3)
            
            # 写入帧 (BGR -> RGB)
            writer.append_data(cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB))
            
            frame_idx += 1
            if frame_idx % 20 == 0:
                print(f"▓进度: {frame_idx}/{total_frames} ({(frame_idx/total_frames)*100:.1f}%)", end='\r')

    except KeyboardInterrupt:
        print("\n🛑 用户终止了进程")
    except Exception as e:
        print(f"\n❌ 运行时发生异常: {e}")
    finally:
        cap.release()
        writer.close()
        print(f"\n✨ 处理完成! 结果文件: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="gelsight_xyz_force_viz.mp4")
    args = parser.parse_args()
    run_extraction(args.input, args.output)
