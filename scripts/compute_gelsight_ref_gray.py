#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.libero_dataset import CustomLeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Compute and cache GelSight ref_gray.")
    parser.add_argument("--data-root", required=True, help="Dataset root path.")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., task_wipe_wrist).")
    parser.add_argument("--nframes", type=int, default=5, help="Number of initial frames per episode to average.")
    parser.add_argument("--cache-path", required=True, help="Output .npy path for ref_gray.")
    args = parser.parse_args()

    dataset = CustomLeRobotDataset(
        data_roots=[args.data_root],
        domains=[args.domain],
        sample_size=(192, 256),
        sample_n_frames=1,
        preprocess="resize",
        valid_cam=["gelsight"],
        chunk=1,
        action_chunk=1,
        n_previous=1,
        previous_pick_mode="uniform",
        random_crop=False,
        ignore_seek=True,
        train_dataset=True,
        action_key="actions",
        state_key="state",
        force_field_ref_nframes=args.nframes,
        force_field_ref_cache_path=args.cache_path,
    )

    ref_gray = dataset._compute_gelsight_ref_gray()
    if ref_gray is None:
        raise RuntimeError("Failed to compute ref_gray. Check dataset path and gelsight frames.")

    os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)
    import numpy as np

    np.save(args.cache_path, ref_gray)
    print(f"[compute_gelsight_ref_gray] Saved ref_gray to {args.cache_path}")


if __name__ == "__main__":
    main()
