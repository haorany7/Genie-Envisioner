import os
import sys
import argparse
import time
import threading
import sys
import select
import termios
import tty
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import torch
from yaml import load, Loader
from einops import rearrange
from xarm.wrapper import XArmAPI

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model
from utils import import_custom_class
from utils.data_utils import get_text_conditions


TARGET_H, TARGET_W = 192, 256
INITIAL_GRIPPER_POS = 300


def process_image(img):
    """Resize to training size (256x192) without crop."""
    return cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


class KeyboardListener:
    """Non-blocking keyboard listener."""
    def __init__(self):
        self.paused = False
        self.running = True
        self.old_settings = None
        self.thread = None

    def _get_key(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def _listen(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        try:
            while self.running:
                key = self._get_key()
                if key:
                    if key == ' ':
                        self.paused = not self.paused
                        status = "⏸️  已暂停" if self.paused else "▶️  继续运行"
                        print(f"\n{status} (按空格键切换)")
                    elif key == '\x03':
                        self.running = False
                        break
                time.sleep(0.01)
        finally:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def start(self):
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        print("⌨️  键盘控制已启用: 空格键暂停/继续 | Ctrl+C 退出")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)


class InferenceXArm:
    def __init__(
        self,
        config_file: str,
        model_path: str | None,
        device: str,
        exec_step: int,
        threshold: int,
        num_inference_steps: int,
    ) -> None:
        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        self.args = args

        if model_path is not None:
            self.args.diffusion_model["model_path"] = model_path

        self.device = device
        self.dtype = torch.bfloat16
        self.exec_step = exec_step
        self.threshold = threshold
        self.num_inference_steps = num_inference_steps

        self.action_dim = cd["diffusion_model"]["config"].get(
            "action_out_channels", cd["diffusion_model"]["config"]["action_in_channels"]
        )
        self.basic_action_dim = 7
        self.add_state = cd.get("add_state", False)
        self.n_prev = cd["data"]["train"]["n_previous"]
        self.chunk = cd["data"]["train"]["chunk"]
        self.action_chunk = cd["data"]["train"]["action_chunk"]

        # stats
        with open(cd["data"]["val"]["stat_file"], "r") as f:
            stats = load(f, Loader=Loader) if cd["data"]["val"]["stat_file"].endswith(".yaml") else __import__("json").load(f)
        domain = cd["data"]["train"]["domains"][0]
        action_space = cd["data"]["train"]["action_space"]
        action_stat_name = f"{domain}_{action_space}"
        state_stat_name = f"{domain}_state_{action_space}"

        self.act_min = torch.tensor(stats[action_stat_name]["q01"]).unsqueeze(0)
        self.act_max = torch.tensor(stats[action_stat_name]["q99"]).unsqueeze(0)
        self.states_min = torch.tensor(stats[state_stat_name]["q01"]).unsqueeze(0)
        self.states_max = torch.tensor(stats[state_stat_name]["q99"]).unsqueeze(0)

        self.obs = []
        self.buffer = []
        self.count = 0
        self.action_buffer = torch.zeros(self.action_chunk, self.action_dim, dtype=self.dtype)

        self.prepare_models()

    def prepare_models(self):
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class,
            textenc_class,
            self.args.pretrained_model_name_or_path
            if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path")
            else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=self.args.load_weights,
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(self.device, dtype=self.dtype).eval()
        self.text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        self.uncond_prompt_embeds = self.text_uncond["prompt_embeds"]
        self.uncond_prompt_attention_mask = self.text_uncond["prompt_attention_mask"]

        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        if getattr(self.args, "vae_path", False):
            self.vae = load_vae_models(vae_class, self.args.vae_path).to(self.device, dtype=self.dtype).eval()
        else:
            self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(
                self.device, dtype=self.dtype
            ).eval()
        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()

        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model["model_path"],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model["config"],
        ).to(self.device, dtype=self.dtype)

        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class, getattr(self.args, "diffusion_scheduler_class_path", "diffusers")
        )
        if hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()

        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )
        self.pipeline = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer, self.diffusion_model
        )

    @torch.no_grad()
    def play(self, obs, prompt, execution_step=1, state=None):
        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) / 255.0 * 2.0 - 1.0
            obs = np.transpose(obs, (0, 3, 1, 2))
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs)

        v, c, h, w = obs.shape
        if obs.max().cpu() > 1.0:
            obs = obs / 255.0 * 2.0 - 1.0
        obs = obs.to(self.device, dtype=self.dtype)

        history_action_state = state
        if self.add_state and history_action_state is not None:
            if isinstance(history_action_state, np.ndarray):
                history_action_state = torch.from_numpy(history_action_state).to(self.device, dtype=self.dtype)
            while len(history_action_state.shape) < 3:
                history_action_state = history_action_state.unsqueeze(dim=0)
            assert history_action_state.shape[-1] == self.action_dim
        else:
            history_action_state = None

        if not self.obs:
            self.obs = [obs] * self.n_prev
            self.count = self.threshold - 1
            self.buffer = [self.obs[-1]]
        else:
            if execution_step == 0:
                return self.action_buffer
            self.count += execution_step
            if self.count >= self.threshold:
                self.count = 0
                self.obs.pop(0)
                self.obs[-1] = self.buffer[0]
                self.obs.append(obs)
            else:
                self.obs[-1] = obs
            self.buffer = [self.obs[-1]]

        obs_tensor = torch.stack(self.obs, dim=1)
        obs_tensor = rearrange(obs_tensor, "v t c h w -> c v t h w")
        obs_tensor = obs_tensor.unsqueeze(0)
        obs_tensor = rearrange(obs_tensor, "b c v t h w -> (b v) c t h w")

        pred_all = self.pipeline.infer(
            image=obs_tensor,
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=self.num_inference_steps,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=1.0,
            height=h,
            width=w,
            n_view=v,
            return_action=True,
            return_video=False,
            chunk=self.chunk,
            action_chunk=self.action_chunk,
            history_action_state=history_action_state if self.add_state else None,
            noise_seed=42,
            pixel_wise_timestep=self.args.pixel_wise_timestep,
            n_chunk=1,
            n_prev=self.n_prev,
            action_dim=self.action_dim,
        )[0]

        actions_pred = pred_all["action"].detach().cpu()[0]
        actions_pred = actions_pred[:, :self.basic_action_dim]
        actions_pred = (actions_pred + 1) / 2
        actions_pred = actions_pred * (self.act_max - self.act_min + 1e-6) + self.act_min
        self.action_buffer = actions_pred.clone()
        return actions_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="wipe the plate")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--exec_step", type=int, default=30)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=20)
    parser.add_argument("--ip", type=str, default="192.168.1.209")
    parser.add_argument("--base_cam_index", type=int, default=4)
    parser.add_argument("--third_cam_index", type=int, default=10)
    parser.add_argument("--gelsight_cam_index", type=int, default=12)
    parser.add_argument("--use_third", action="store_true", help="Use third view camera")
    parser.add_argument("--use_gelsight", action="store_true", help="Use gelsight camera")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--smoothing", type=float, default=0.15)
    args = parser.parse_args()

    infer = InferenceXArm(
        config_file=args.config_file,
        model_path=args.ckpt_path,
        device=args.device,
        exec_step=args.exec_step,
        threshold=args.threshold,
        num_inference_steps=args.num_inference_steps,
    )

    arm = XArmAPI(args.ip, do_not_open=True)
    arm.connect()
    arm.clean_gripper_error()
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(4000)
    arm.set_gripper_position(INITIAL_GRIPPER_POS, wait=True)
    arm.motion_enable(True)
    arm.set_mode(1)
    arm.set_state(0)

    cap_base = cv2.VideoCapture(args.base_cam_index)
    cap_third = cv2.VideoCapture(args.third_cam_index) if args.use_third else None
    cap_gel = cv2.VideoCapture(args.gelsight_cam_index) if args.use_gelsight else None

    caps = [cap_base]
    if cap_third is not None:
        caps.append(cap_third)
    if cap_gel is not None:
        caps.append(cap_gel)

    for cap in caps:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap_base.isOpened():
        print("❌ BASE_CAM_INDEX 打开失败")
        return
    if cap_third is not None and not cap_third.isOpened():
        print("❌ THIRD_CAM_INDEX 打开失败")
        return
    if cap_gel is not None and not cap_gel.isOpened():
        print("❌ GELSIGHT_CAM_INDEX 打开失败")
        return

    kb = KeyboardListener()
    kb.start()

    code, current_qpos = arm.get_servo_angle(is_radian=True)
    last_target = np.array(current_qpos[:6])

    try:
        while kb.running:
            if kb.paused:
                time.sleep(0.1)
                continue

            for _ in range(2):
                cap_base.grab()
                if cap_third is not None:
                    cap_third.grab()
                if cap_gel is not None:
                    cap_gel.grab()
            ret_b, img_b_full = cap_base.retrieve()
            if not ret_b:
                continue
            img_t_full = None
            img_g_full = None
            if cap_third is not None:
                ret_t, img_t_full = cap_third.retrieve()
                if not ret_t:
                    continue
            if cap_gel is not None:
                ret_g, img_g_full = cap_gel.retrieve()
                if not ret_g:
                    continue

            img_b_proc = process_image(img_b_full)
            ge_b = cv2.cvtColor(img_b_proc, cv2.COLOR_BGR2RGB)
            views = [ge_b]
            debug_panes = [img_b_proc]
            if img_t_full is not None:
                img_t_proc = process_image(img_t_full)
                ge_t = cv2.cvtColor(img_t_proc, cv2.COLOR_BGR2RGB)
                views.append(ge_t)
                debug_panes.append(img_t_proc)
            if img_g_full is not None:
                img_g_proc = process_image(img_g_full)
                ge_g = cv2.cvtColor(img_g_proc, cv2.COLOR_BGR2RGB)
                views.append(ge_g)
                debug_panes.append(img_g_proc)
            stacked_obs = np.stack(views, axis=0)

            code, qpos = arm.get_servo_angle(is_radian=True)
            _, gpos = arm.get_gripper_position()
            if code != 0 or gpos is None:
                continue

            # Save model input mosaic for debugging
            # 创建三栏式可视化 (Base | Third | Gelsight/Stats)
            pane1 = img_b_proc.copy()
            pane2 = img_t_proc.copy() if img_t_full is not None else np.zeros_like(img_b_proc)
            pane3 = img_g_proc.copy() if img_g_full is not None else np.zeros_like(img_b_proc)
            
            # 在 pane3 上覆盖文字信息
            y_offset = 25
            line_height = 20
            cv2.putText(pane3, f"Prompt: {args.prompt}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += line_height
            cv2.putText(pane3, f"Qpos (deg):", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
            for j_idx, val in enumerate(qpos[:6]):
                deg = np.rad2deg(val)
                cv2.putText(pane3, f" J{j_idx+1}: {deg:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 15
            cv2.putText(pane3, f" Gpos: {gpos:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            debug_img = np.hstack([pane1, pane2, pane3])
            cv2.imwrite("eval_xarm_input.jpg", debug_img)

            current_state = np.concatenate([qpos[:6], [gpos]])
            state = torch.tensor(current_state).unsqueeze(0)
            state = (state - infer.states_min) / (infer.states_max - infer.states_min + 1e-6)
            state = state * 2 - 1
            state = torch.cat((torch.zeros_like(state), state), dim=1)

            actions = infer.play(stacked_obs, args.prompt, execution_step=args.exec_step, state=state)
            actions = actions.cpu().numpy()

            # 每一轮推理开始时，更新当前的执行起点
            for i in range(args.exec_step):
                if not kb.running or kb.paused:
                    break
                step_start_time = time.time()
                
                target_q = actions[i][:6]
                target_g = actions[i][6]

                # 仍然保留安全 Clip
                if np.any(np.abs(target_q - last_target) > 0.4):
                    target_q = np.clip(target_q, last_target - 0.1, last_target + 0.1)

                # 使用轻量平滑滤波 (α=args.smoothing)
                executed_q = (1 - args.smoothing) * last_target + args.smoothing * target_q
                arm.set_servo_angle_j(angles=executed_q, is_radian=True)

                if i % 3 == 0:
                    arm.clean_gripper_error()
                    arm.set_gripper_position(target_g, wait=False, speed=4000)

                # 精确控制频率
                elapsed = time.time() - step_start_time
                time.sleep(max(0, 1 / args.fps - elapsed))
                last_target = executed_q

    finally:
        kb.stop()
        arm.disconnect()
        cap_base.release()
        if cap_third is not None:
            cap_third.release()
        if cap_gel is not None:
            cap_gel.release()


if __name__ == "__main__":
    main()
