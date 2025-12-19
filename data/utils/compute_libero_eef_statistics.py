import os
import math
import json
from typing import List, Tuple

import pyarrow.parquet as pq


def _init_accumulators(dim: int):
    return {
        "count": [0] * dim,
        "sum": [0.0] * dim,
        "sum_sq": [0.0] * dim,
        "min": [float("inf")] * dim,
        "max": [float("-inf")] * dim,
    }


def _update_accumulators(acc, values: List[float]):
    assert len(values) == len(acc["count"])
    for i, v in enumerate(values):
        if v is None:
            continue
        v = float(v)
        acc["count"][i] += 1
        acc["sum"][i] += v
        acc["sum_sq"][i] += v * v
        if v < acc["min"][i]:
            acc["min"][i] = v
        if v > acc["max"][i]:
            acc["max"][i] = v


def _finalize_stats(acc) -> Tuple[List[float], List[float], List[float], List[float]]:
    dim = len(acc["count"])
    mean, std, vmin, vmax = [], [], [], []
    for i in range(dim):
        c = acc["count"][i]
        if c == 0:
            mean.append(0.0)
            std.append(0.0)
            vmin.append(0.0)
            vmax.append(0.0)
            continue
        m = acc["sum"][i] / c
        var = acc["sum_sq"][i] / c - m * m
        mean.append(m)
        std.append(math.sqrt(max(var, 0.0)))
        vmin.append(acc["min"][i])
        vmax.append(acc["max"][i])
    return mean, std, vmin, vmax


def compute_libero_eef_stats(
    dataset_root: str = "/work/hdd/bfxb/data/libero-hfvla",
    verbose: bool = True,
):
    """
    统一从 Libero LeRobot 数据集中重新计算：
      - action: 7D EEF 动作
      - state:  7D EEF-style 状态: [x, y, z, roll, pitch, yaw, gripper_state]
        其中 gripper_state = finger_left - finger_right

    并输出可以直接粘贴到 `statistics.py` 的 JSON 片段。
    """
    data_root = os.path.join(dataset_root, "data")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")

    parquet_files: List[str] = []
    for chunk in sorted(os.listdir(data_root)):
        chunk_dir = os.path.join(data_root, chunk)
        if not os.path.isdir(chunk_dir):
            continue
        for fn in sorted(os.listdir(chunk_dir)):
            if fn.endswith(".parquet"):
                parquet_files.append(os.path.join(chunk_dir, fn))

    if verbose:
        print(f"Found {len(parquet_files)} parquet files under {data_root}")

    # Libero: action 原生 7D, state 原生 8D
    acc_action = _init_accumulators(7)
    acc_state7 = _init_accumulators(7)

    total_rows = 0

    for i, path in enumerate(parquet_files):
        table = pq.read_table(path, columns=["action", "observation.state"])
        actions = table["action"].to_pylist()
        states = table["observation.state"].to_pylist()

        for a, s in zip(actions, states):
            # action: expect length 7
            if a is not None and len(a) == 7:
                _update_accumulators(acc_action, list(a))

            # state 8D -> 7D EEF-style: [0:6] + (6-7)
            if s is not None and len(s) == 8:
                x, y, z, roll, pitch, yaw, f_left, f_right = [float(v) for v in s]
                gripper_state = f_left - f_right
                state7 = [x, y, z, roll, pitch, yaw, gripper_state]
                _update_accumulators(acc_state7, state7)

            total_rows += 1

        if verbose and (i + 1) % 50 == 0:
            print(
                f"Processed {i+1}/{len(parquet_files)} files, "
                f"total_rows={total_rows}, "
                f"action_count_dim0={acc_action['count'][0]}, "
                f"state_count_dim0={acc_state7['count'][0]}",
                flush=True,
            )

    # Finalize
    act_mean, act_std, act_min, act_max = _finalize_stats(acc_action)
    st_mean, st_std, st_min, st_max = _finalize_stats(acc_state7)

    if verbose:
        print("\n=== Libero EEF action (7D) statistics ===")
        print("mean:", act_mean)
        print("std: ", act_std)
        print("min: ", act_min)
        print("max: ", act_max)

        print("\n=== Libero EEF-style state (7D) statistics ===")
        print("mean:", st_mean)
        print("std: ", st_std)
        print("min: ", st_min)
        print("max: ", st_max)

    # Build snippet compatible with `StatisticInfo` in statistics.py
    snippet = {
        "libero-hfvla_eef": {
            "mean": act_mean,
            "std": act_std,
            "min": act_min,
            "max": act_max,
            "normalize": "mean_std",
        },
        "libero-hfvla_state_eef": {
            "mean": st_mean,
            "std": st_std,
            "min": st_min,
            "max": st_max,
            "normalize": "mean_std",
        },
    }

    print("\n\n=== JSON snippet for `data/utils/statistics.py` ===")
    print(json.dumps(snippet, indent=2))

    return snippet


if __name__ == "__main__":
    compute_libero_eef_stats()


