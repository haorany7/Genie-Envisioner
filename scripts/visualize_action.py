#!/usr/bin/env python3
"""
visualize_action.py

Plot robot joint angles, TCP position/rotation, and gripper opening
against frame number (row index) from a robot log CSV.

Usage example:
	python visualize_action.py --csv task/episode_000/robot_log.csv --out-dir visualization
	python visualize_action.py --csv task/episode_000/robot_log.csv --show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil


def load_csv(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path, comment="#", skip_blank_lines=True)
	# clean column names and drop empty columns
	df.columns = df.columns.str.strip()
	# drop columns that are all NaN (sometimes trailing commas create them)
	df = df.dropna(axis=1, how="all")
	df = df.reset_index(drop=True)
	return df


def plot_by_frame(df: pd.DataFrame, to_degrees: bool = False, save_path: Path | None = None) -> None:
	frame = np.arange(len(df))

	joints = [f"j{i}_rad" for i in range(1, 7)]
	tcp_pos = ["tcp_x_m", "tcp_y_m", "tcp_z_m"]
	tcp_rot = ["tcp_rx_rad", "tcp_ry_rad", "tcp_rz_rad"]
	gripper_col = "gripper_mm"

	# Safely select columns that exist
	joint_cols = [c for c in joints if c in df.columns]
	tcp_pos_cols = [c for c in tcp_pos if c in df.columns]
	tcp_rot_cols = [c for c in tcp_rot if c in df.columns]
	has_gripper = gripper_col in df.columns

	if not (joint_cols or tcp_pos_cols or tcp_rot_cols or has_gripper):
		raise ValueError("No recognized columns found in CSV to plot.")

	# Prepare values
	joint_vals = df[joint_cols].astype(float) if joint_cols else pd.DataFrame()
	tcp_pos_vals = df[tcp_pos_cols].astype(float) if tcp_pos_cols else pd.DataFrame()
	tcp_rot_vals = df[tcp_rot_cols].astype(float) if tcp_rot_cols else pd.DataFrame()
	gripper_vals = df[gripper_col].astype(float) if has_gripper else None

	if to_degrees:
		joint_vals = joint_vals * 180.0 / np.pi
		tcp_rot_vals = tcp_rot_vals * 180.0 / np.pi

	nrows = 4
	fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 10), sharex=True)
	if nrows == 1:
		axes = [axes]

	# Joints
	ax = axes[0]
	for col in joint_vals.columns:
		ax.plot(frame, joint_vals[col], label=col)
	ax.set_ylabel("joint (deg)" if to_degrees else "joint (rad)")
	if joint_vals.columns.any():
		ax.legend(ncol=3, fontsize="small")
	ax.grid(True)

	# TCP position
	ax = axes[1]
	for col in tcp_pos_vals.columns:
		ax.plot(frame, tcp_pos_vals[col], label=col)
	ax.set_ylabel("tcp pos (m)")
	if tcp_pos_vals.columns.any():
		ax.legend(fontsize="small")
	ax.grid(True)

	# TCP rotation
	ax = axes[2]
	for col in tcp_rot_vals.columns:
		ax.plot(frame, tcp_rot_vals[col], label=col)
	ax.set_ylabel("tcp rot (deg)" if to_degrees else "tcp rot (rad)")
	if tcp_rot_vals.columns.any():
		ax.legend(fontsize="small")
	ax.grid(True)

	# Gripper
	ax = axes[3]
	if gripper_vals is not None:
		ax.plot(frame, gripper_vals, label=gripper_col, color="tab:purple")
	ax.set_ylabel("gripper (mm)")
	ax.set_xlabel("frame")
	ax.grid(True)

	plt.tight_layout()

	if save_path:
		fig.savefig(save_path, dpi=150)
		print(f"Saved plot to {save_path}")
	else:
		plt.show()


def check_lengths(df: pd.DataFrame, episode_dir: Path) -> dict:
	"""Check that the number of action rows matches camera/gelsight images.

	Returns a dict with counts and a boolean 'all_equal'. Prints a brief
	summary and a warning when counts differ.
	"""
	counts = {}
	counts["actions"] = len(df)

	# Check gelsight and realsense folders
	cams = ["gelsight", "realsense_0", "realsense_1", "realsense_2"]
	for cam in cams:
		p = episode_dir / cam
		if p.exists() and p.is_dir():
			# count common image extensions
			img_count = sum(1 for _ in p.glob("*.png"))
			img_count += sum(1 for _ in p.glob("*.jpg"))
			img_count += sum(1 for _ in p.glob("*.jpeg"))
			counts[cam] = img_count
		else:
			counts[cam] = 0

	# Compare counts: consider equal if all non-zero counts equal actions
	nonzero_counts = [v for v in counts.values() if v > 0]
	all_equal = all(v == counts["actions"] for v in nonzero_counts)

	# Print summary
	print("Data length check:")
	print(f"  actions rows: {counts['actions']}")
	for cam in cams:
		print(f"  {cam}: {counts[cam]}")

	if not all_equal:
		print("Warning: counts differ between action rows and camera/gelsight frames.")
	else:
		print("OK: action rows match camera/gelsight frame counts.")

	counts["all_equal"] = all_equal
	return counts


def make_video_from_folder(img_dir: Path, out_path: Path, fps: int = 10) -> Path:
	"""Create a video from all images in `img_dir` (sorted lexicographically).

	Uses `imageio` with ffmpeg. Returns the written video path.
	"""
	try:
		import imageio
	except Exception as e:
		raise RuntimeError("imageio is required to create videos. Install with: pip install imageio[ffmpeg]") from e

	imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
	if not imgs:
		raise FileNotFoundError(f"No images found in {img_dir}")

	out_path.parent.mkdir(parents=True, exist_ok=True)

	# imageio expects a string path for ffmpeg writer
	with imageio.get_writer(str(out_path), fps=fps) as writer:
		for p in imgs:
			img = imageio.imread(str(p))
			writer.append_data(img)

	return out_path



def main() -> None:
	parser = argparse.ArgumentParser(description="Visualize robot log vs frame number")
	parser.add_argument("--csv", "-c", type=Path, default=Path("task/episode_000/robot_log.csv"), help="Path to robot_log.csv")
	parser.add_argument("--out-dir", "-o", type=Path, default=Path("visualization"), help="Directory to save plots (creates per-episode subfolders)")
	parser.add_argument("--show", action="store_true", help="Show interactive plot instead of saving")
	parser.add_argument("--to-degrees", action="store_true", help="Convert radians to degrees for joints and tcp rotations (disabled by default)")
	parser.add_argument("--make-video", action="store_true", help="Create video(s) from image folders")
	parser.add_argument("--camera", type=str, default="gelsight", choices=["gelsight", "realsense_0", "realsense_1", "realsense_2", "all"], help="Which camera folder to convert to video, or 'all'")
	parser.add_argument("--fps", type=int, default=30, help="Frames per second for generated video")
	args = parser.parse_args()

	if not args.csv.exists():
		print(f"CSV not found: {args.csv}", file=sys.stderr)
		sys.exit(2)

	df = load_csv(args.csv)

	# verify lengths between action rows and images
	episode_dir = args.csv.parent
	check_lengths(df, episode_dir)

	if args.show:
		plot_by_frame(df, to_degrees=args.to_degrees, save_path=None)
		return

	# Determine episode name from CSV path's parent folder (e.g. task/episode_000/robot_log.csv -> episode_000)
	episode_name = args.csv.parent.name
	save_dir = args.out_dir / episode_name
	save_dir.mkdir(parents=True, exist_ok=True)
	save_path = save_dir / "plot.png"

	plot_by_frame(df, to_degrees=args.to_degrees, save_path=save_path)

	# Optionally create videos from image folders
	if args.make_video:
		cams = [args.camera] if args.camera != "all" else ["gelsight", "realsense_0", "realsense_1", "realsense_2"]
		for cam in cams:
			cam_dir = episode_dir / cam
			if not cam_dir.exists():
				print(f"Skipping {cam}: folder not found: {cam_dir}")
				continue
			try:
				video_out = args.out_dir / episode_name / f"{cam}.mp4"
				print(f"Creating video for {cam} -> {video_out} (fps={args.fps})")
				make_video_from_folder(cam_dir, video_out, fps=args.fps)
				print(f"Saved video to {video_out}")
			except Exception as e:
				print(f"Failed to create video for {cam}: {e}")



if __name__ == "__main__":
	main()

