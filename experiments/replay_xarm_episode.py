#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay a recorded episode on the xArm using actions stored in a LeRobot parquet file.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xarm.wrapper import XArmAPI
from yaml import load, Loader


def load_actions(parquet_path: Path) -> np.ndarray:
    df = pd.read_parquet(parquet_path)
    if "actions" not in df.columns:
        raise ValueError(f"'actions' column not found in {parquet_path}")
    actions_col = df["actions"].tolist()
    # Each row should be a list/array of shape (C,)
    actions = np.stack([np.array(a).flatten() for a in actions_col]).astype(np.float32)
    return actions


def load_action_stats(config_file: Path) -> dict:
    cd = load(open(config_file, "r"), Loader=Loader)
    stat_path = cd["data"]["val"]["stat_file"]
    with open(stat_path, "r") as f:
        stats = load(f, Loader=Loader) if stat_path.endswith(".yaml") else json.load(f)
    domain = cd["data"]["train"]["domains"][0]
    action_space = cd["data"]["train"]["action_space"]
    action_type = cd["data"]["train"]["action_type"]
    if action_type == "delta":
        action_stat_name = f"{domain}_delta_{action_space}"
    else:
        action_stat_name = f"{domain}_{action_space}"
    return {
        "action_type": action_type,
        "action_stat": stats[action_stat_name],
    }


def denorm_actions(actions: np.ndarray, action_stat: dict, gripper_index: int) -> np.ndarray:
    # Use minmax denorm for absolute actions.
    q01 = np.array(action_stat["q01"], dtype=np.float32)[None, :]
    q99 = np.array(action_stat["q99"], dtype=np.float32)[None, :]
    return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str)
    parser.add_argument("--npy_actions", type=str, help="Path to .npy actions (T,7) for replay")
    parser.add_argument("--config_file", type=str, help="Config yaml for denormalizing npy actions")
    parser.add_argument("--no_denorm", action="store_true", help="Do not denormalize npy actions")
    parser.add_argument("--action_dim", type=int, default=7, help="Number of action dims to replay")
    parser.add_argument("--gripper_index", type=int, default=6, help="Index of gripper dim to skip denorm")
    parser.add_argument("--ip", type=str, default="192.168.1.209")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=-1)
    parser.add_argument("--gripper_speed", type=int, default=4000)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.npy_actions:
        actions = np.load(args.npy_actions).astype(np.float32)
        if actions.shape[1] > args.action_dim:
            actions = actions[:, :args.action_dim]
        if args.config_file and not args.no_denorm:
            stats_info = load_action_stats(Path(args.config_file))
            if stats_info["action_type"] != "absolute":
                raise ValueError(
                    f"action_type={stats_info['action_type']} is not absolute; "
                    "denorm alone is not enough for replay."
                )
            actions = denorm_actions(actions, stats_info["action_stat"], args.gripper_index)
    else:
        if not args.parquet_path:
            raise ValueError("Either --parquet_path or --npy_actions must be provided.")
        parquet_path = Path(args.parquet_path)
        actions = load_actions(parquet_path)

    start_idx = max(args.start_idx, 0)
    end_idx = actions.shape[0] if args.num_steps < 0 else min(actions.shape[0], start_idx + args.num_steps)
    actions = actions[start_idx:end_idx]

    if actions.shape[1] < 7:
        raise ValueError(f"Expected at least 7 action dims, got {actions.shape[1]}")

    if args.dry_run:
        print(f"[DRY RUN] Loaded {len(actions)} actions from {parquet_path}")
        print(f"[DRY RUN] First action: {actions[0]}")
        return

    arm = XArmAPI(args.ip, do_not_open=True)
    arm.connect()
    arm.clean_gripper_error()
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(args.gripper_speed)
    arm.motion_enable(True)
    arm.set_mode(1)
    arm.set_state(0)

    try:
        for i, act in enumerate(actions):
            step_start = time.time()
            target_q = act[:6]
            target_g = act[6]

            arm.set_servo_angle_j(angles=target_q.tolist(), is_radian=True)
            if i % 3 == 0:
                arm.clean_gripper_error()
                arm.set_gripper_position(float(target_g), wait=False, speed=args.gripper_speed)

            elapsed = time.time() - step_start
            sleep_t = max(0.0, 1.0 / args.fps - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
