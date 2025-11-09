#!/usr/bin/env python3
"""
Utility script to recompute mean/std/min/max statistics for a LeRobot-style dataset.

Example:
  python scripts/compute_dataset_statistics.py \
      --dataset-root /work/hdd/behe/AgiBotWorld-Alpha_lerobot/agibotworld \
      --action-key action \
      --state-key observation.state
"""

import argparse
import glob
import json
import math
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def pad_to_16(arr: np.ndarray) -> np.ndarray:
    """
    Actions/states can be 14-dim (joints only). We align them to 16 dims by inserting
    two gripper placeholders (zeros) mirroring the runtime dataset logic.
    """
    if arr.shape[-1] == 16:
        return arr
    if arr.shape[-1] == 14:
        left = arr[..., :7]
        right = arr[..., 7:]
        zeros = np.zeros((*arr.shape[:-1], 1), dtype=arr.dtype)
        return np.concatenate([left, zeros, right, zeros], axis=-1)
    raise ValueError(f"Unexpected feature dimension {arr.shape[-1]} (expected 14 or 16)")


def init_acc(dim: int) -> Dict[str, np.ndarray]:
    return {
        "count": 0,
        "mean": np.zeros(dim, dtype=np.float64),
        "m2": np.zeros(dim, dtype=np.float64),
        "min": np.full(dim, np.inf, dtype=np.float64),
        "max": np.full(dim, -np.inf, dtype=np.float64),
    }


def update_acc(acc: Dict[str, np.ndarray], data: np.ndarray) -> None:
    for row in data:
        acc["count"] += 1
        delta = row - acc["mean"]
        acc["mean"] += delta / acc["count"]
        delta2 = row - acc["mean"]
        acc["m2"] += delta * delta2
        acc["min"] = np.minimum(acc["min"], row)
        acc["max"] = np.maximum(acc["max"], row)


def finalize_acc(acc: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if acc["count"] < 2:
        raise ValueError("Not enough samples to compute statistics.")
    variance = acc["m2"] / (acc["count"] - 1)
    std = np.sqrt(variance)
    return acc["mean"], std, acc["min"], acc["max"]


def compute_statistics(dataset_root: str, action_key: str, state_key: str) -> Dict[str, object]:
    parquet_pattern = os.path.join(dataset_root, "data", "chunk-*", "episode_*.parquet")
    files = sorted(glob.glob(parquet_pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found with pattern: {parquet_pattern}")

    print(f"Found {len(files)} parquet files under {dataset_root}")

    action_acc = None
    delta_action_acc = None
    state_acc = None

    total_steps = 0
    for idx, pq_path in enumerate(files, start=1):
        df = pd.read_parquet(pq_path, columns=[action_key, state_key])

        actions = pad_to_16(np.stack(df[action_key].to_numpy()))
        states = pad_to_16(np.stack(df[state_key].to_numpy()))

        if action_acc is None:
            dim = actions.shape[1]
            action_acc = init_acc(dim)
            delta_action_acc = init_acc(dim)
            state_acc = init_acc(states.shape[1])

        update_acc(action_acc, actions)

        # Delta actions: last frame uses previous action for delta.
        if actions.shape[0] > 1:
            delta_actions = actions[1:] - actions[:-1]
            # keep gripper absolute values identical to dataset logic
            delta_actions[:, 6] = actions[:-1, 6]
            delta_actions[:, 13] = actions[:-1, 13]
            update_acc(delta_action_acc, delta_actions)

        update_acc(state_acc, states)
        total_steps += actions.shape[0]

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(files)} files...")

    action_mean, action_std, action_min, action_max = finalize_acc(action_acc)
    delta_mean, delta_std, delta_min, delta_max = finalize_acc(delta_action_acc)
    state_mean, state_std, state_min, state_max = finalize_acc(state_acc)

    return {
        "num_files": len(files),
        "total_steps": total_steps,
        "action": {
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
            "min": action_min.tolist(),
            "max": action_max.tolist(),
        },
        "delta_action": {
            "mean": delta_mean.tolist(),
            "std": delta_std.tolist(),
            "min": delta_min.tolist(),
            "max": delta_max.tolist(),
        },
        "state": {
            "mean": state_mean.tolist(),
            "std": state_std.tolist(),
            "min": state_min.tolist(),
            "max": state_max.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset statistics for LeRobot-style data.")
    parser.add_argument("--dataset-root", required=True, help="Root path containing data/meta/videos folders.")
    parser.add_argument("--action-key", default="action", help="Column name for actions in parquet files.")
    parser.add_argument("--state-key", default="observation.state", help="Column name for states in parquet files.")
    parser.add_argument("--output", help="Optional path to save statistics JSON.")
    args = parser.parse_args()

    stats = compute_statistics(args.dataset_root, args.action_key, args.state_key)
    text = json.dumps(stats, indent=2)
    print(text)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Saved statistics to {args.output}")


if __name__ == "__main__":
    main()

