import os
import json
import argparse
import numpy as np
import torch
import torchvision
from pathlib import Path
from yaml import load, Loader
from tqdm import tqdm
from einops import rearrange

# 导入项目原有的组件
from runner.ge_inferencer import Inferencer
from utils.attention_viz import AttentionStore, VizAttentionProcessor, visualize_attention_on_images
from models.pipeline.custom_pipeline import CustomPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="attention_viz_results")
    parser.add_argument("--n_validation", type=int, default=1, help="跑多少个 Episode")
    parser.add_argument("--n_chunk_action", type=int, default=10, help="每个 Episode 跑多少步推理")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # 1. 初始化 Inferencer (继承其模型和数据加载逻辑)
    inf = Inferencer(args.config_file, output_dir=args.output_path, device=args.device)
    inf.args.load_weights = True
    inf.args.diffusion_model['model_path'] = args.checkpoint_path
    
    inf.prepare_models()
    inf.prepare_val_dataset()
    
    # 2. 注入注意力捕获器 (VizAttentionProcessor)
    store = AttentionStore()
    viz_processor = VizAttentionProcessor(store)
    
    count = 0
    for name, module in inf.diffusion_model.named_modules():
        if "action_blocks" in name and name.endswith("attn2"):
            module.processor = viz_processor
            count += 1
    print(f"✅ 已注入 {count} 个注意力可视化处理器")

    # 3. 准备 Pipeline
    pipe = CustomPipeline(
        inf.scheduler, inf.vae, inf.text_encoder, inf.tokenizer, inf.diffusion_model
    ).to(inf.device)

    # 4. 开始主循环
    dataset = inf.val_dataloader.dataset
    n_view = len(inf.args.data["train"]["valid_cam"])
    n_prev = inf.args.data['train']['n_previous']
    action_chunk = inf.args.data['train']['action_chunk']
    
    os.makedirs(args.output_path, exist_ok=True)

    # 模仿 ge_inferencer.py 的多 Episode 循环逻辑
    total_episodes = min(args.n_validation, len(dataset))
    
    for i_val in range(total_episodes):
        dataset.fix_epiidx = i_val
        dataset.fix_sidx = 0 
        dataset.fix_mem_idx = [1 for _ in range(n_prev)]
        
        episode_frames = []
        print(f"🚀 正在处理 Episode {i_val}/{total_episodes-1}...")

        for i_chunk in tqdm(range(args.n_chunk_action), desc=f"Episode {i_val}"):
            # 获取数据
            batch = next(iter(inf.val_dataloader))
            image = batch['video'][:,:,:,:n_prev] 
            prompt = batch['caption']
            
            b, c, v, t, h, w = image.shape
            image_in = rearrange(image, 'b c v t h w -> (b v) c t h w').to(inf.device, dtype=inf.weight_dtype)

            if getattr(inf.args, "add_state", False):
                history_action_state = batch["state"].to(inf.device, dtype=inf.weight_dtype)
            else:
                history_action_state = None

            store.reset()

            with torch.no_grad():
                pipe.infer(
                    image=image_in,
                    prompt=prompt[:1],
                    num_inference_steps=inf.args.num_inference_step,
                    height=h, width=w, n_view=v,
                    return_action=True,
                    return_video=False,
                    n_prev=n_prev,
                    action_chunk=action_chunk,
                    history_action_state=history_action_state,
                    action_dim=inf.args.diffusion_model["config"]["action_in_channels"],
                )

            # 注意力热力图计算
            n_layers = count
            last_step_maps = store.attention_maps[-n_layers:]
            avg_attn = torch.stack(last_step_maps).mean(dim=0)[0] 

            seq_k = avg_attn.shape[-1]
            L = seq_k // v
            gh, gw = h // 8, w // 8
            if gh * gw != L:
                ratio = np.sqrt(L / (h * w))
                gh, gw = int(h * ratio), int(w * ratio)
            if gh * gw != L:
                for i in range(int(np.sqrt(L)), 0, -1):
                    if L % i == 0:
                        gh, gw = i, L // i
                        break
            grid_size = (min(gh, gw), max(gh, gw)) if h < w else (max(gh, gw), min(gh, gw))

            vis_images = [image[0, :, v_idx, -1] for v_idx in range(v)]
            
            # 生成临时帧图片
            tmp_frame_path = os.path.join(args.output_path, f"tmp_epi{i_val}_chunk{i_chunk:03d}.png")
            visualize_attention_on_images(
                vis_images, avg_attn, tmp_frame_path, 
                action_idx=0, grid_size=grid_size, n_view=v
            )
            
            # 读取图片并转为 tensor 存储
            import cv2
            frame_bgr = cv2.imread(tmp_frame_path)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            episode_frames.append(torch.from_numpy(frame_rgb))
            
            # 删除临时文件
            os.remove(tmp_frame_path)

            # 时间推进
            dataset.fix_sidx += action_chunk
            dataset.fix_mem_idx = (np.linspace(0, dataset.fix_sidx-1, n_prev).round().astype(np.int16)).tolist()

        # 5. 使用 torchvision 保存视频 (H.264 编码，兼容性更好)
        if episode_frames:
            video_path = os.path.join(args.output_path, f"episode_{i_val}_attention.mp4")
            video_tensor = torch.stack(episode_frames) # [T, H, W, C]
            torchvision.io.write_video(video_path, video_tensor, fps=5, video_codec='libx264')
            print(f"🎬 Episode {i_val} 视频已保存至: {video_path}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
