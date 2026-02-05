#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XArm Diffusion Policy Deployment Script (Improved)
Key improvements:
  1) Remove GPU sync in obs.max().cpu() check (performance + stability)
  2) Replace "any-joint triggers all-joint clip" with wrap-aware per-joint rate limiter
  3) Optional: update last_target from actual qpos to reduce drift
  4) Throttle debug image writing to reduce I/O jitter
  5) Normalize state in numpy float32 and let play() move it to GPU (less dtype/device churn)
"""

import os
import sys
import argparse
import time
import csv
import threading
import select
import termios
import tty

import cv2
import numpy as np
import torch
from yaml import load, Loader
from einops import rearrange
from xarm.wrapper import XArmAPI

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.model_utils import (
    load_condition_models,
    load_latent_models,
    load_vae_models,
    load_diffusion_model,
)
from utils import import_custom_class
from utils.data_utils import get_text_conditions


TARGET_H, TARGET_W = 192, 256
INITIAL_GRIPPER_POS = 300


def process_image(img):
    """Resize to training size (256x192) without crop."""
    return cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


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
                    if key == " ":
                        self.paused = not self.paused
                        status = "⏸️  已暂停" if self.paused else "▶️  继续运行"
                        print(f"\n{status} (按空格键切换)")
                    elif key == "\x03":  # Ctrl+C
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
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


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

        # stats (kept both numpy and torch forms)
        stat_path = cd["data"]["val"]["stat_file"]
        with open(stat_path, "r") as f:
            stats = (
                load(f, Loader=Loader)
                if stat_path.endswith(".yaml")
                else __import__("json").load(f)
            )

        domain = cd["data"]["train"]["domains"][0]
        action_space = cd["data"]["train"]["action_space"]
        action_stat_name = f"{domain}_{action_space}"
        state_stat_name = f"{domain}_state_{action_space}"

        act_min_np = np.array(stats[action_stat_name]["q01"], dtype=np.float32)[None, :]
        act_max_np = np.array(stats[action_stat_name]["q99"], dtype=np.float32)[None, :]
        states_min_np = np.array(stats[state_stat_name]["q01"], dtype=np.float32)[None, :]
        states_max_np = np.array(stats[state_stat_name]["q99"], dtype=np.float32)[None, :]

        self.act_min_np = act_min_np
        self.act_max_np = act_max_np
        self.states_min_np = states_min_np
        self.states_max_np = states_max_np

        # torch (move to device once)
        self.act_min = torch.tensor(act_min_np, device=self.device, dtype=self.dtype)
        self.act_max = torch.tensor(act_max_np, device=self.device, dtype=self.dtype)
        self.states_min = torch.tensor(states_min_np, device=self.device, dtype=self.dtype)
        self.states_max = torch.tensor(states_max_np, device=self.device, dtype=self.dtype)

        self.obs = []
        self.buffer = []
        self.count = 0
        self.action_buffer = torch.zeros(self.action_chunk, self.action_dim, dtype=self.dtype)

        self.prepare_models()

    def prepare_models(self):
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class,
            getattr(self.args, "tokenizer_class_path", "transformers"),
        )
        textenc_class = import_custom_class(
            self.args.textenc_class,
            getattr(self.args, "textenc_class_path", "transformers"),
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
            self.args.diffusion_model_class,
            getattr(self.args, "diffusion_model_class_path", "transformers"),
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model["model_path"],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model["config"],
        ).to(self.device, dtype=self.dtype)

        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class,
            getattr(self.args, "diffusion_scheduler_class_path", "diffusers"),
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
    def play(self, obs, prompt, execution_step=1, state=None, return_normalized=False, return_full=False):
        """
        obs: (V,H,W,3) uint8 RGB or (V,3,H,W) float in [-1,1]
        state: numpy or torch; expected already normalized to [-1,1] if add_state is enabled
        """
        # normalize obs on CPU if uint8
        if isinstance(obs, np.ndarray):
            if obs.dtype == np.uint8:
                obs = obs.astype(np.float32) / 255.0 * 2.0 - 1.0
                obs = np.transpose(obs, (0, 3, 1, 2))  # V,C,H,W
            obs = torch.from_numpy(obs)

        v, c, h, w = obs.shape
        obs = obs.to(self.device, dtype=self.dtype)  # do NOT call obs.max().cpu() (sync)

        history_action_state = state
        if self.add_state and history_action_state is not None:
            if isinstance(history_action_state, np.ndarray):
                history_action_state = torch.from_numpy(history_action_state)
            history_action_state = history_action_state.to(self.device, dtype=self.dtype)
            while len(history_action_state.shape) < 3:
                history_action_state = history_action_state.unsqueeze(dim=0)
            # keep original assertion, but now your state should match expected dim
            assert history_action_state.shape[-1] == self.action_dim, (
                f"history_action_state last dim {history_action_state.shape[-1]} != action_dim {self.action_dim}"
            )
        else:
            history_action_state = None

        # frame buffering logic
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

        obs_tensor = torch.stack(self.obs, dim=1)  # V,T,C,H,W
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

        actions_pred_full = pred_all["action"].detach()[0]  # (action_chunk, action_dim) on device
        # Use the last 7 dims as action (absolute) output.
        actions_pred = actions_pred_full[:, -self.basic_action_dim:]

        if return_normalized:
            self.action_buffer = actions_pred.clone()
            if return_full:
                return actions_pred.detach().cpu(), actions_pred_full.detach().cpu()
            return actions_pred.detach().cpu()  # normalized (for relative modes)

        # denorm to real action space (action stats)
        actions_pred = (actions_pred + 1) / 2
        actions_pred = actions_pred * (self.act_max - self.act_min + 1e-6) + self.act_min

        self.action_buffer = actions_pred.clone()
        if return_full:
            return actions_pred.detach().cpu(), actions_pred_full.detach().cpu()
        return actions_pred.detach().cpu()  # return on CPU for control thread


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
    parser.add_argument("--max_step", type=float, default=0.10, help="Max joint delta per 30Hz step (rad/step)")
    parser.add_argument("--debug_every", type=int, default=10, help="Write debug image every N cycles (0 disables)")
    parser.add_argument("--debug_action_every", type=int, default=0, help="Print action debug every N cycles (0 disables)")
    parser.add_argument("--log_dir", type=str, default="eval_logs", help="Directory for 1Hz/30Hz CSV logs")
    parser.add_argument(
        "--action_type",
        type=str,
        default="absolute",
        choices=["absolute", "relative_base", "relative_delta"],
        help=(
            "absolute: actions[i] is absolute joint target.\n"
            "relative_base: actions[i] is offset from base qpos at start of 1Hz cycle.\n"
            "relative_delta: actions[i] is delta from previous executed target (integrate)."
        ),
    )
    parser.add_argument(
        "--use_actual_last",
        action="store_true",
        help="Use actual qpos as last_target each step (reduces drift, slightly more reads).",
    )
    args = parser.parse_args()

    infer = InferenceXArm(
        config_file=args.config_file,
        model_path=args.ckpt_path,
        device=args.device,
        exec_step=args.exec_step,
        threshold=args.threshold,
        num_inference_steps=args.num_inference_steps,
    )

    # robot init
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

    # cameras
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

    # log book (CSV)
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pred_log_path = os.path.join(log_dir, f"pred_1hz_{timestamp}.csv")
    state_log_path = os.path.join(log_dir, f"state_30hz_{timestamp}.csv")
    pred_log_f = open(pred_log_path, "w", newline="")
    state_log_f = open(state_log_path, "w", newline="")
    pred_writer = csv.writer(pred_log_f)
    state_writer = csv.writer(state_log_f)
    pred_writer.writerow(
        ["time", "frame_idx", "step_idx", "action_type", "qpos0", "qpos1", "qpos2", "qpos3", "qpos4", "qpos5", "gpos",
         "pred_action0", "pred_action1", "pred_action2", "pred_action3", "pred_action4", "pred_action5", "pred_action6",
         "pred_state0", "pred_state1", "pred_state2", "pred_state3", "pred_state4", "pred_state5", "pred_state6"]
    )
    state_writer.writerow(
        ["time", "frame_idx", "step_idx", "qpos0", "qpos1", "qpos2", "qpos3", "qpos4", "qpos5", "gpos",
         "cmd0", "cmd1", "cmd2", "cmd3", "cmd4", "cmd5", "cmd_g"]
    )

    # initial last_target
    code, current_qpos = arm.get_servo_angle(is_radian=True)
    last_target = np.array(current_qpos[:6], dtype=np.float32)

    frame_idx = 0

    try:
        while kb.running:
            if kb.paused:
                time.sleep(0.1)
                continue

            # grab latest frames
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

            # preprocess images
            img_b_proc = process_image(img_b_full)
            ge_b = cv2.cvtColor(img_b_proc, cv2.COLOR_BGR2RGB)

            views = [ge_b]
            img_t_proc = None
            img_g_proc = None
            if img_t_full is not None:
                img_t_proc = process_image(img_t_full)
                ge_t = cv2.cvtColor(img_t_proc, cv2.COLOR_BGR2RGB)
                views.append(ge_t)
            if img_g_full is not None:
                img_g_proc = process_image(img_g_full)
                ge_g = cv2.cvtColor(img_g_proc, cv2.COLOR_BGR2RGB)
                views.append(ge_g)

            stacked_obs = np.stack(views, axis=0)  # V,H,W,3 uint8 RGB

            # read robot state
            code, qpos = arm.get_servo_angle(is_radian=True)
            _, gpos = arm.get_gripper_position()
            if code != 0 or gpos is None:
                continue

            # --- debug mosaic (throttled) ---
            if args.debug_every > 0 and (frame_idx % args.debug_every == 0):
                pane1 = img_b_proc.copy()
                pane2 = img_t_proc.copy() if img_t_proc is not None else np.zeros_like(img_b_proc)
                pane3 = img_g_proc.copy() if img_g_proc is not None else np.zeros_like(img_b_proc)

                y_offset = 25
                line_height = 20
                cv2.putText(
                    pane3, f"Prompt: {args.prompt}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                )
                y_offset += line_height
                cv2.putText(
                    pane3, "Qpos (deg):", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                )
                y_offset += 15
                for j_idx, val in enumerate(qpos[:6]):
                    deg = np.rad2deg(val)
                    cv2.putText(
                        pane3, f" J{j_idx+1}: {deg:.1f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
                    )
                    y_offset += 15
                cv2.putText(
                    pane3, f" Gpos: {gpos:.1f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
                )

                debug_img = np.hstack([pane1, pane2, pane3])
                cv2.imwrite("eval_xarm_input.jpg", debug_img)

            frame_idx += 1

            # --- build normalized state (numpy float32) ---
            current_state = np.concatenate([np.array(qpos[:6], dtype=np.float32), np.array([gpos], dtype=np.float32)], axis=0)[None, :]  # (1,7)
            state_norm = (current_state - infer.states_min_np) / (infer.states_max_np - infer.states_min_np + 1e-6)
            state_norm = state_norm * 2.0 - 1.0

            # NOTE: This (1,14) concatenation MUST match your training definition.
            # Keep it as in your original script; if assertion fails, the training expects a different state format.
            state_in = np.concatenate([np.zeros_like(state_norm), state_norm], axis=1).astype(np.float32)  # (1,14)

            # inference (returns CPU tensor)
            actions_norm_last, actions_norm_full = infer.play(
                stacked_obs,
                args.prompt,
                execution_step=args.exec_step,
                state=state_in,
                return_normalized=True,
                return_full=True,
            )
            actions_norm_last = actions_norm_last.numpy()
            actions_norm_full = actions_norm_full.numpy()

            pred_action_norm = actions_norm_full[:, :7]
            pred_state_norm = actions_norm_full[:, 7:]
            pred_action_abs = (pred_action_norm + 1.0) / 2.0
            pred_action_abs = pred_action_abs * (infer.act_max_np - infer.act_min_np + 1e-6) + infer.act_min_np
            pred_state_abs = (pred_state_norm + 1.0) / 2.0
            pred_state_abs = pred_state_abs * (infer.states_max_np - infer.states_min_np + 1e-6) + infer.states_min_np

            if args.action_type == "absolute":
                # Execute with the last 7 dims (treated as absolute state prediction).
                actions = pred_state_abs
            else:
                actions_norm = pred_action_norm
                # relative modes: predicted actions are in normalized space
                if args.action_type == "relative_base":
                    action_norm = actions_norm.copy()
                    # Relative for arm dims; gripper stays absolute in normalized space.
                    action_norm[:, :6] = action_norm[:, :6] + state_norm[:, :6]
                    actions = (action_norm + 1.0) / 2.0
                    actions = actions * (infer.act_max_np - infer.act_min_np + 1e-6) + infer.act_min_np
                else:  # relative_delta
                    integrated_norm = state_norm.copy()
                    actions = np.zeros_like(actions_norm, dtype=np.float32)
                    for i in range(actions_norm.shape[0]):
                        integrated_norm[:, :6] = integrated_norm[:, :6] + actions_norm[i:i+1, :6]
                        integrated_norm[:, 6:] = actions_norm[i:i+1, 6:]
                        action_norm = integrated_norm
                        action_real = (action_norm + 1.0) / 2.0
                        action_real = action_real * (infer.act_max_np - infer.act_min_np + 1e-6) + infer.act_min_np
                        actions[i] = action_real[0]

            if args.debug_action_every > 0 and (frame_idx % args.debug_action_every == 0):
                if args.action_type == "absolute":
                    print(
                        f"[DEBUG] state_norm={state_norm[0]}, "
                        f"pred_action_abs[0]={pred_action_abs[0]}, "
                        f"pred_state_abs[0]={pred_state_abs[0]}"
                    )
                else:
                    print(
                        f"[DEBUG] state_norm={state_norm[0]}, "
                        f"action_norm_pred[0]={actions_norm[0]}, "
                        f"action_abs[0]={actions[0]}, "
                        f"pred_state_abs[0]={pred_state_abs[0]}"
                    )

            # 1Hz log: current state + full predicted horizon (e.g., 54 steps)
            now_ts = time.time()
            for step_idx in range(pred_action_abs.shape[0]):
                pred_writer.writerow(
                    [
                        now_ts,
                        frame_idx,
                        step_idx,
                        args.action_type,
                        qpos[0], qpos[1], qpos[2], qpos[3], qpos[4], qpos[5],
                        gpos,
                        pred_action_abs[step_idx][0], pred_action_abs[step_idx][1], pred_action_abs[step_idx][2],
                        pred_action_abs[step_idx][3], pred_action_abs[step_idx][4], pred_action_abs[step_idx][5],
                        pred_action_abs[step_idx][6],
                        pred_state_abs[step_idx][0], pred_state_abs[step_idx][1], pred_state_abs[step_idx][2],
                        pred_state_abs[step_idx][3], pred_state_abs[step_idx][4], pred_state_abs[step_idx][5],
                        pred_state_abs[step_idx][6],
                    ]
                )
            pred_log_f.flush()

            # execute horizon

            for i in range(args.exec_step):
                if not kb.running or kb.paused:
                    break

                step_start_time = time.time()

                # optionally refresh last_target from actual robot state (reduces drift)
                if args.use_actual_last:
                    code_a, qpos_a = arm.get_servo_angle(is_radian=True)
                    if code_a == 0:
                        last_target = np.array(qpos_a[:6], dtype=np.float32)

                # decode action type (already absolute in physical space)
                target_q = actions[i][:6].astype(np.float32)

                target_g = float(actions[i][6])

                # light smoothing (EMA)
                executed_q = (1.0 - args.smoothing) * last_target + args.smoothing * target_q

                # send joint command
                arm.set_servo_angle_j(angles=executed_q.tolist(), is_radian=True)

                # gripper at ~10Hz
                if i % 3 == 0:
                    arm.clean_gripper_error()
                    arm.set_gripper_position(target_g, wait=False, speed=4000)

                # 30Hz log: current state + command
                code_s, qpos_s = arm.get_servo_angle(is_radian=True)
                _, gpos_s = arm.get_gripper_position()
                if code_s == 0 and gpos_s is not None:
                    state_writer.writerow(
                        [
                            time.time(),
                            frame_idx,
                            i,
                            qpos_s[0], qpos_s[1], qpos_s[2], qpos_s[3], qpos_s[4], qpos_s[5],
                            gpos_s,
                            executed_q[0], executed_q[1], executed_q[2], executed_q[3], executed_q[4], executed_q[5],
                            target_g,
                        ]
                    )
                    state_log_f.flush()

                # keep 30Hz
                elapsed = time.time() - step_start_time
                sleep_t = max(0.0, 1.0 / args.fps - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)

                last_target = executed_q.astype(np.float32)

    finally:
        kb.stop()
        arm.disconnect()
        pred_log_f.close()
        state_log_f.close()
        cap_base.release()
        if cap_third is not None:
            cap_third.release()
        if cap_gel is not None:
            cap_gel.release()


if __name__ == "__main__":
    main()
