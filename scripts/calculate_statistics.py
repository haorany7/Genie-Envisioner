import os
import numpy as np
import pandas as pd
import tqdm
import json
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def load_data(data_path, key="action"):
    """Load action or state data from parquet file"""
    data = pd.read_parquet(data_path)
    data = np.stack([data[key][i] for i in range(data[key].shape[0])])
    return data 

def unwrap_angles(data, angle_dims=[3, 4, 5]):
    """
    Apply unwrap to angle dimensions (Euler angles) to remove 2π discontinuities.
    
    Args:
        data: (T, C) numpy array
        angle_dims: list of dimension indices to unwrap (default: [3,4,5] for roll, pitch, yaw)
    
    Returns:
        data with unwrapped angles
    """
    T, C = data.shape
    if T > 0 and C >= 6:
        for d in angle_dims:
            if d < C:
                data[:, d] = np.unwrap(data[:, d])
    return data

def process_single_file(parquet_path, action_key, state_key, is_calvin_eef, data_type, check_jumps=False, pos_threshold=50.0, ori_threshold=1.0, joint_threshold=1.0):
    """
    Process a single parquet file and return action, delta_action, and state.
    This function is designed to be called in parallel.
    
    Args:
        check_jumps: if True, check for large jumps in action trajectory
        pos_threshold: threshold for position jump (meters, default: 0.1m = 10cm)
        ori_threshold: threshold for orientation jump (radians, default: 1.0 rad ≈ 57°)
    
    Returns:
        tuple: (action, delta_action, state, warnings) or None if error
    """
    try:
        # Load action data
        action = load_data(parquet_path, action_key)
        
        # Apply unwrap to EEF angles if needed
        if is_calvin_eef and data_type == "eef":
            # For CALVIN EEF: [x, y, z, roll, pitch, yaw, gripper]
            # Unwrap Euler angles (dims 3, 4, 5)
            action = unwrap_angles(action, angle_dims=[3, 4, 5])
        
        # Compute delta within this episode (not across episodes!)
        delta_action = None
        if len(action) > 1:
            delta_action = action[1:] - action[:-1]
        
        # Check for large jumps if requested
        warnings = []
        if check_jumps and delta_action is not None and len(delta_action) > 0:
            T, C = delta_action.shape
            if data_type == "eef":
                # Check first 6 dimensions (EEF: pos_x, pos_y, pos_z, roll, pitch, yaw)
                if C >= 6:
                    for t in range(T):
                        # Check position (x, y, z) in mm
                        pos_delta = delta_action[t, :3]
                        pos_jump = np.linalg.norm(pos_delta)
                        if pos_jump > pos_threshold:
                            warnings.append({
                                "episode": parquet_path,
                                "timestep": t + 1,  # +1 because delta is between t and t+1
                                "type": "position",
                                "jump_magnitude": float(pos_jump),
                                "threshold": pos_threshold,
                                "delta": pos_delta.tolist()
                            })
                        
                        # Check orientation (roll, pitch, yaw) in rad
                        ori_delta = delta_action[t, 3:6]
                        for i, angle_name in enumerate(['roll', 'pitch', 'yaw']):
                            angle_jump = abs(ori_delta[i])
                            if angle_jump > ori_threshold:
                                warnings.append({
                                    "episode": parquet_path,
                                    "timestep": t + 1,
                                    "type": f"orientation_{angle_name}",
                                    "jump_magnitude": float(angle_jump),
                                    "threshold": ori_threshold,
                                    "delta": float(ori_delta[i])
                                })
            else:
                # Joint space: all dims are in radians (and/or gripper), check per-dim magnitude
                for t in range(T):
                    for i in range(C):
                        angle_jump = abs(delta_action[t, i])
                        if angle_jump > joint_threshold:
                            warnings.append({
                                "episode": parquet_path,
                                "timestep": t + 1,
                                "type": f"joint_{i}",
                                "jump_magnitude": float(angle_jump),
                                "threshold": joint_threshold,
                                "delta": float(delta_action[t, i])
                            })
        
        # Load state data
        state = load_data(parquet_path, state_key)
        
        # Apply unwrap to state angles if needed
        if is_calvin_eef and data_type == "eef":
            # For CALVIN state (15D): [EE pos(3), EE ori(3), gripper_width(1), joint_pos(7), gripper_action(1)]
            # Unwrap EE orientation (dims 3, 4, 5)
            state = unwrap_angles(state, angle_dims=[3, 4, 5])
            
            # Extract 7D EEF state to match action dimensions
            # [EE pos(3), EE ori(3), gripper_action(1)]
            if state.shape[-1] == 15:
                pos_ori = state[:, :6]      # EE position + orientation
                grip_act = state[:, -1:]    # gripper action (last dim)
                state = np.concatenate([pos_ori, grip_act], axis=-1)
        
        return (action, delta_action, state, warnings)
        
    except Exception as e:
        print(f"\nError processing {parquet_path}: {e}")
        return None

def cal_statistic(data, _filter=True):
    """
    Calculate statistics.

    - mean/std are used for zscore-style normalization.
    - q01/q99 are used for min-max scaling in some dataloaders (e.g., libero_dataset.py).
    - min/max are also kept for backward compatibility / debugging.
    """
    q99 = np.percentile(data, 99, axis=0)
    q01 = np.percentile(data, 1, axis=0)

    if _filter:
        data_mask = (data >= q01) & (data <= q99)
        data_mask = data_mask.min(axis=1)
        data = data[data_mask, :]

    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    return means, stds, mins, maxs, q01, q99

def get_statistics(
    data_root, 
    data_name, 
    data_type, 
    save_path, 
    action_key="action", 
    state_key="observation.state", 
    nrnd=50000, 
    _filter=True,
    is_calvin_eef=False,
    num_workers=32,
    check_jumps=True,
    pos_threshold=50.0,
    ori_threshold=1.0,
    joint_threshold=1.0
):
    """
    Compute dataset statistics for actions, delta actions, and states.
    
    Args:
        data_root: Root directory containing parquet files
        data_name: Dataset name (e.g., "ABC_lerobot" or "D_lerobot")
        data_type: Action space type ("joint" or "eef")
        save_path: Path to save JSON statistics
        action_key: Key for action data in parquet
        state_key: Key for state data in parquet
        nrnd: Maximum number of episodes to sample
        _filter: Whether to filter outliers (1st-99th percentile)
        is_calvin_eef: If True, apply unwrap to EEF angles and handle state/delta properly
        num_workers: Number of parallel workers for processing files
        check_jumps: If True, check for large jumps in action trajectory
        pos_threshold: Position jump threshold in mm (default: 50mm)
        ori_threshold: Orientation jump threshold in radians (default: 1.0 rad ≈ 57°)
        joint_threshold: Joint jump threshold in radians (default: 1.0 rad)
    """
    
    assert data_type in ["joint", "eef"], f"data_type must be 'joint' or 'eef', got {data_type}"
    
    # Get all parquet files
    if not os.path.isdir(data_root):
        raise ValueError(f"data_root must be a directory: {data_root}")

    parquet_files = []

    # Case A: plain LeRobot layout: <data_root>/data/chunk-*/episode_*.parquet
    direct_data_dir = os.path.join(data_root, "data")
    if os.path.isdir(direct_data_dir):
        parquet_files.extend(glob.glob(os.path.join(direct_data_dir, "chunk-*", "*.parquet")))

    # Case B: domain/task subdir layout: <data_root>/<domain>/data/chunk-*/*.parquet
    if len(parquet_files) == 0:
        for task_dir in os.listdir(data_root):
            task_path = os.path.join(data_root, task_dir)
            if os.path.isdir(task_path):
                data_dir = os.path.join(task_path, "data")
                if os.path.exists(data_dir):
                    parquet_files.extend(glob.glob(os.path.join(data_dir, "chunk-*", "*.parquet")))
        
    # Case C: fallback: parquet directly under root
        if len(parquet_files) == 0:
            parquet_files = glob.glob(os.path.join(data_root, "*.parquet"))
    
    parquet_files.sort()
    print(f"Found {len(parquet_files)} parquet files")
    if len(parquet_files) == 0:
        raise RuntimeError(
            f"No parquet files found under '{data_root}'. Expected LeRobot layout like '{data_root}/data/chunk-*/episode_*.parquet'."
        )
    
    # Randomly sample if needed
    if nrnd > 0 and nrnd < len(parquet_files):
        parquet_files = list(np.random.choice(parquet_files, nrnd, replace=False))
        print(f"Randomly sampled {nrnd} files")
    
    data_list = []
    state_list = []
    delta_data_list = []
    all_warnings = []
    
    print(f"Processing episodes with {num_workers} workers...")
    
    # Create a partial function with fixed parameters
    process_func = partial(
        process_single_file,
        action_key=action_key,
        state_key=state_key,
        is_calvin_eef=is_calvin_eef,
        data_type=data_type,
        check_jumps=check_jumps,
        pos_threshold=pos_threshold,
        ori_threshold=ori_threshold,
        joint_threshold=joint_threshold
    )
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_func, path): path for path in parquet_files}
        
        # Collect results with progress bar
        for future in tqdm.tqdm(as_completed(futures), total=len(parquet_files)):
            result = future.result()
            if result is not None:
                action, delta_action, state, warnings = result
                data_list.append(action)
                if delta_action is not None:
                    delta_data_list.append(delta_action)
                state_list.append(state)
                if warnings:
                    all_warnings.extend(warnings)
    
    # Concatenate all data
    print("\nConcatenating data...")
    data_list = np.concatenate(data_list, axis=0)
    assert len(data_list.shape) == 2, f"Expected 2D array, got shape {data_list.shape}"
    print(f"Action data shape: {data_list.shape}")
    
    delta_data_list = np.concatenate(delta_data_list, axis=0) if delta_data_list else np.zeros((0, data_list.shape[1]))
    assert len(delta_data_list.shape) == 2, f"Expected 2D array, got shape {delta_data_list.shape}"
    print(f"Delta action data shape: {delta_data_list.shape}")
    
    state_list = np.concatenate(state_list, axis=0)
    assert len(state_list.shape) == 2, f"Expected 2D array, got shape {state_list.shape}"
    print(f"State data shape: {state_list.shape}")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    means, stds, mins, maxs, q01, q99 = cal_statistic(data_list, _filter=_filter)
    delta_means, delta_stds, delta_mins, delta_maxs, delta_q01, delta_q99 = cal_statistic(
        delta_data_list, _filter=_filter
    )
    state_means, state_stds, state_mins, state_maxs, state_q01, state_q99 = cal_statistic(
        state_list, _filter=_filter
    )
    
    # Build statistics dictionary
    statistics_info = {
        f"{data_name}_{data_type}": {
            "mean": means.tolist(),
            "std": stds.tolist(),
            "min": mins.tolist(),
            "max": maxs.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "normalize": "mean_std"
        },
        f"{data_name}_delta_{data_type}": {
            "mean": delta_means.tolist(),
            "std": delta_stds.tolist(),
            "min": delta_mins.tolist(),
            "max": delta_maxs.tolist(),
            "q01": delta_q01.tolist(),
            "q99": delta_q99.tolist(),
            "normalize": "mean_std"
        },
        f"{data_name}_state_{data_type}": {
            "mean": state_means.tolist(),
            "std": state_stds.tolist(),
            "min": state_mins.tolist(),
            "max": state_maxs.tolist(),
            "q01": state_q01.tolist(),
            "q99": state_q99.tolist(),
            "normalize": "mean_std"
        },
    }
    
    # Save to JSON
    print(f"\nSaving statistics to {save_path}...")
    with open(save_path, "w") as f:
        json.dump(statistics_info, f, indent=4)
    
    # Report warnings if any
    if all_warnings:
        print(f"\n{'='*60}")
        print(f"⚠️  WARNING: Found {len(all_warnings)} large action jumps!")
        print(f"{'='*60}")
        
        # Group warnings by type
        pos_warnings = [w for w in all_warnings if w['type'] == 'position']
        ori_warnings = [w for w in all_warnings if w['type'].startswith('orientation')]
        
        if pos_warnings:
            print(f"\n📍 Position jumps (> {pos_threshold}m): {len(pos_warnings)} occurrences")
            print(f"   Top 5 largest jumps:")
            sorted_pos = sorted(pos_warnings, key=lambda x: x['jump_magnitude'], reverse=True)[:5]
            for w in sorted_pos:
                print(f"   - {w['episode']}")
                print(f"     Timestep {w['timestep']}: {w['jump_magnitude']:.4f}m (delta: {w['delta']})")
        
        if ori_warnings:
            print(f"\n🔄 Orientation jumps (> {ori_threshold} rad): {len(ori_warnings)} occurrences")
            print(f"   Top 5 largest jumps:")
            sorted_ori = sorted(ori_warnings, key=lambda x: x['jump_magnitude'], reverse=True)[:5]
            for w in sorted_ori:
                print(f"   - {w['episode']}")
                print(f"     Timestep {w['timestep']}: {w['type']} = {w['jump_magnitude']:.4f} rad ({np.degrees(w['jump_magnitude']):.1f}°)")
        
        # Save warnings to a separate file
        warnings_path = save_path.replace('.json', '_warnings.json')
        with open(warnings_path, 'w') as f:
            json.dump(all_warnings, f, indent=2)
        print(f"\n💾 Full warning details saved to: {warnings_path}")
    else:
        print(f"\n✅ No large action jumps detected (pos_threshold={pos_threshold}m, ori_threshold={ori_threshold} rad)")
    
    print("Done!")
    print(f"\nStatistics keys generated:")
    for key in statistics_info.keys():
        print(f"  - {key}")
    
    return statistics_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute dataset statistics with proper angle unwrapping (parallelized)")
    parser.add_argument('--data_root', required=True, help="Root directory of the dataset")
    parser.add_argument('--data_name', required=True, help="Dataset name (e.g., ABC_lerobot, D_lerobot)")
    parser.add_argument('--data_type', required=True, choices=["joint", "eef"], help="Action space type")
    parser.add_argument('--action_key', default="action", help="Key for action data in parquet")
    parser.add_argument('--state_key', default="observation.state", help="Key for state data in parquet")
    parser.add_argument('--save_path', required=True, help="Path to save JSON statistics")
    parser.add_argument('--nrnd', type=int, default=0, help="Number of episodes to randomly sample (0 = use all)")
    parser.add_argument('--filter', action='store_true', default=True, help="Filter outliers (1-99 percentile)")
    parser.add_argument('--is_calvin_eef', action='store_true', help="Apply CALVIN EEF specific processing (unwrap angles)")
    parser.add_argument('--num_workers', type=int, default=32, help="Number of parallel workers (default: 32)")
    parser.add_argument('--check_jumps', action='store_true', default=True, help="Check for large jumps in action trajectory")
    parser.add_argument('--pos_threshold', type=float, default=50.0, help="Position jump threshold in mm (default: 50)")
    parser.add_argument('--ori_threshold', type=float, default=1.0, help="Orientation jump threshold in radians (default: 1.0)")
    parser.add_argument('--joint_threshold', type=float, default=1.0, help="Joint jump threshold in radians (default: 1.0)")
    
    args = parser.parse_args()
    
    get_statistics(
        data_root=args.data_root,
        data_name=args.data_name,
        data_type=args.data_type,
        save_path=args.save_path,
        action_key=args.action_key,
        state_key=args.state_key,
        nrnd=args.nrnd,
        _filter=args.filter,
        is_calvin_eef=args.is_calvin_eef,
        num_workers=args.num_workers,
        check_jumps=args.check_jumps,
        pos_threshold=args.pos_threshold,
        ori_threshold=args.ori_threshold,
        joint_threshold=args.joint_threshold
    )
