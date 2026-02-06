#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize denormalized actions from a .npy file using minmax stats.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from yaml import load, Loader


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


def denorm_actions(actions: np.ndarray, action_stat: dict) -> np.ndarray:
    q01 = np.array(action_stat["q01"], dtype=np.float32)[None, :]
    q99 = np.array(action_stat["q99"], dtype=np.float32)[None, :]
    return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_actions", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--output", type=str, default="denorm_actions.png")
    parser.add_argument("--action_dim", type=int, default=7)
    args = parser.parse_args()

    actions = np.load(args.npy_actions).astype(np.float32)
    if actions.shape[1] > args.action_dim:
        actions = actions[:, :args.action_dim]

    stats_info = load_action_stats(Path(args.config_file))
    if stats_info["action_type"] != "absolute":
        raise ValueError(
            f"action_type={stats_info['action_type']} is not absolute; denorm plot is invalid."
        )
    actions_denorm = denorm_actions(actions, stats_info["action_stat"])

    num_dims = actions_denorm.shape[1]
    rows = (num_dims + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3 * rows), sharex=True)
    axes = axes.flatten()

    x = np.arange(actions_denorm.shape[0])
    for i in range(num_dims):
        ax = axes[i]
        ax.plot(x, actions_denorm[:, i], color="tomato", linewidth=1.2)
        ax.set_title(f"Action dim {i}")
        ax.grid(True, linestyle=":", alpha=0.6)

    for j in range(num_dims, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
