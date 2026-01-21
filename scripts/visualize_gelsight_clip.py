import cv2
import numpy as np
import argparse
import imageio

def process_gelsight_robust(video_path, output_path="tactile_pure_viz.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 强制进行 2x2 裁剪：只取左上角 (Top-Left)
    tile_w, tile_h = full_w // 2, full_h // 2
    print(f"模式：左上角处理 + 灵敏热力图 ({tile_w}x{tile_h})")

    ret, first_frame = cap.read()
    if not ret: return
    
    # 裁剪第一帧作为参考背景
    ref_tile = first_frame[0:tile_h, 0:tile_w].copy()
    ref_gray = cv2.cvtColor(ref_tile, cv2.COLOR_BGR2GRAY)
    
    def find_dots_robust(img_gray):
        # 1. 阈值处理并应用形态学清理
        _, thresh = cv2.threshold(img_gray, 65, 255, cv2.THRESH_BINARY_INV)
        # 开运算：先腐蚀后膨胀，消除细小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 200: # 限制面积范围，过滤极小噪点和极大色块
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    dots.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
        return np.array(dots)

    ref_dots = find_dots_robust(ref_gray)
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 严格裁剪左上角
        curr_tile = frame[0:tile_h, 0:tile_w].copy()
        curr_gray = cv2.cvtColor(curr_tile, cv2.COLOR_BGR2GRAY)
        
        # --- 1. 压力热力图优化 (提高灵敏度) ---
        diff = cv2.absdiff(curr_tile, ref_tile)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # 降低降噪阈值：15 -> 7
        _, diff_mask = cv2.threshold(diff_gray, 7, 255, cv2.THRESH_TOZERO)
        
        # 平滑处理
        diff_blur = cv2.GaussianBlur(diff_mask, (15, 15), 0)
        
        # 只要最大变动值超过 15，就认为有有效接触
        has_contact = np.max(diff_blur) > 15
        if has_contact:
            heatmap_norm = cv2.normalize(diff_blur, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
        
        # --- 2. 向量场优化 (防抖) ---
        curr_dots = find_dots_robust(curr_gray)
        viz_tile = curr_tile.copy()
        
        max_match_dist_sq = 100  # 仅允许 10px 内的点进行匹配
        min_disp_sq = 4          # 仅显示明显位移
        for ref_p in ref_dots:
            if len(curr_dots) > 0:
                dists = np.sum((curr_dots - ref_p)**2, axis=1)
                idx = np.argmin(dists)
                dist_sq = dists[idx]
                
                # 只在 dot 上绘制：必须是近邻匹配且位移明显
                if min_disp_sq < dist_sq < max_match_dist_sq:
                    curr_p = curr_dots[idx]
                    dx = int((curr_p[0] - ref_p[0]) * 6)
                    dy = int((curr_p[1] - ref_p[1]) * 6)
                    start = tuple(ref_p)
                    end = (ref_p[0] + dx, ref_p[1] + dy)
                    # 力越大箭头越粗：用位移强度映射线宽
                    disp = np.sqrt(dist_sq)
                    thickness = int(np.clip(1 + disp * 0.4, 1, 6))
                    cv2.circle(viz_tile, start, 1, (0, 0, 255), -1)
                    cv2.arrowedLine(viz_tile, start, end, (0, 0, 255), thickness, tipLength=0.3)

        # 3. 合成输出
        if has_contact:
            output_clip_bgr = cv2.addWeighted(heatmap, 0.45, viz_tile, 0.55, 0)
        else:
            output_clip_bgr = viz_tile
            
        output_clip_rgb = cv2.cvtColor(output_clip_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(output_clip_rgb)
        
        frame_idx += 1
        if frame_idx % 100 == 0: print(f"已处理 {frame_idx} 帧...")

    cap.release()
    writer.close()
    print(f"处理完成: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="gelsight_robust_viz.mp4")
    args = parser.parse_args()
    process_gelsight_robust(args.input, args.output)
