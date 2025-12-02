#!/usr/bin/env python
"""
Compare two CALVIN tactile policy rollout runs.

Usage example:
  python scripts/compare_calvin_tactile_runs.py \\
    --run1 /work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/outputs/calvin_policy_tactile_joint_infer/20251111_164056/2025_11_11_16_48_53/Rollout \\
    --run2 /work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/outputs/calvin_policy_tactile_joint_infer/20251114_003330/2025_11_14_00_33_49/Rollout \\
    --out_dir /work/hdd/bche/haorany7/WORLD-MODEL-TOUCH/outputs/calvin_policy_tactile_joint_infer/comparison_20251114

This will generate:
  - total_loss_per_task_run1_vs_run2.png
  - per_dim_loss_across_tasks_run2.png
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def compute_losses(run1_root: Path, run2_root: Path):
    files1 = sorted(run1_root.glob("rollout_task_*.npz"))
    files2 = sorted(run2_root.glob("rollout_task_*.npz"))

    names1 = [f.name for f in files1]
    names2 = [f.name for f in files2]
    if names1 != names2:
        raise RuntimeError("File lists are not aligned between runs.")

    n_tasks = len(files1)
    print(f"Found {n_tasks} tasks.")

    total1, total2 = [], []
    per_dim1, per_dim2 = [], []

    for f1 in files1:
        f2 = run2_root / f1.name
        d1 = np.load(f1)
        d2 = np.load(f2)

        diff1 = d1["pred"] - d1["gt"]
        diff2 = d2["pred"] - d2["gt"]

        # Total MSE over time and dims
        total1.append((diff1 ** 2).mean())
        total2.append((diff2 ** 2).mean())

        # Per-dim MSE (avg over time)
        per_dim1.append((diff1 ** 2).mean(axis=0))
        per_dim2.append((diff2 ** 2).mean(axis=0))

    total1 = np.asarray(total1)
    total2 = np.asarray(total2)
    per_dim1 = np.stack(per_dim1)  # (tasks, dim)
    per_dim2 = np.stack(per_dim2)

    return names1, total1, total2, per_dim1, per_dim2


def plot_total_loss(task_names, total1, total2, out_path: Path, label1: str, label2: str):
    x = np.arange(len(task_names))
    plt.figure(figsize=(10, 4))
    plt.plot(x, total1, "-o", label=label1)
    plt.plot(x, total2, "-o", label=label2)
    plt.xlabel("task index")
    plt.ylabel("total MSE (pred vs gt)")
    plt.title("Total loss per task")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_per_dim_loss(task_names, per_dim, out_path: Path, title_prefix: str):
    x = np.arange(len(task_names))
    num_dims = per_dim.shape[1]

    plt.figure(figsize=(10, 4))
    for d in range(num_dims):
        plt.plot(x, per_dim[:, d], label=f"dim{d}")
    plt.xlabel("task index")
    plt.ylabel("per-dim MSE (avg over time)")
    plt.title(f"{title_prefix} per-dimension loss across tasks")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1", type=str, required=True, help="Rollout directory of first run (contains rollout_task_*.npz)")
    parser.add_argument("--run2", type=str, required=True, help="Rollout directory of second run")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save comparison plots")
    parser.add_argument("--label1", type=str, default="run1", help="Label for first run")
    parser.add_argument("--label2", type=str, default="run2", help="Label for second run")
    args = parser.parse_args()

    run1_root = Path(args.run1)
    run2_root = Path(args.run2)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    task_names, total1, total2, per_dim1, per_dim2 = compute_losses(run1_root, run2_root)

    plot_total_loss(
        task_names,
        total1,
        total2,
        out_dir / "total_loss_per_task_run1_vs_run2.png",
        args.label1,
        args.label2,
    )

    # Plot per-dim for the second run by default
    plot_per_dim_loss(
        task_names,
        per_dim2,
        out_dir / "per_dim_loss_across_tasks_run2.png",
        title_prefix=args.label2,
    )


if __name__ == "__main__":
    main()


