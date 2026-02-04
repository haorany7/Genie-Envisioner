import argparse
import time
import numpy as np
from xarm.wrapper import XArmAPI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test TCP motion with xArm servo cartesian control.")
    parser.add_argument("--ip", type=str, default="192.168.1.209", help="Robot IP, e.g. 192.168.1.209")
    parser.add_argument("--rate", type=float, default=30.0, help="Servo rate (Hz)")
    parser.add_argument("--steps", type=int, default=30, help="Interpolation steps per segment")
    parser.add_argument("--max-pos-delta", type=float, default=220.0, help="Clamp per-segment XYZ delta (mm)")
    parser.add_argument("--max-rot-delta", type=float, default=1.2, help="Clamp per-segment RPY delta (rad)")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Absolute TCP target as 'x,y,z,roll,pitch,yaw' (mm, rad)",
    )
    parser.add_argument("--return", dest="return_home", action="store_true", help="Return to start pose")
    parser.add_argument("--dry-run", action="store_true", help="Print targets without moving")
    return parser.parse_args()


def connect_arm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip, do_not_open=True)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_mode(1)  # servo mode
    arm.set_state(state=0)
    time.sleep(0.2)
    return arm


def safe_clip_delta(delta: np.ndarray, max_pos: float, max_rot: float) -> np.ndarray:
    clipped = delta.copy()
    clipped[:3] = np.clip(clipped[:3], -max_pos, max_pos)
    clipped[3:] = np.clip(clipped[3:], -max_rot, max_rot)
    return clipped


def run_to_target(
    arm: XArmAPI,
    start_tcp: np.ndarray,
    target_tcp: np.ndarray,
    steps: int,
    rate: float,
    dry_run: bool,
) -> np.ndarray:
    delta = target_tcp - start_tcp
    for i in range(1, steps + 1):
        interp = start_tcp + delta * (i / steps)
        if not dry_run:
            code = arm.set_servo_cartesian(interp.tolist(), is_radian=True)
            if code != 0:
                print(f"⚠️ set_servo_cartesian error code: {code}")
        time.sleep(1.0 / rate)
    return target_tcp


def main() -> None:
    args = parse_args()
    arm = connect_arm(args.ip)
    try:
        code, tcp = arm.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_position failed with code {code}")
        current_tcp = np.array(tcp[:6], dtype=np.float32)
        start_tcp = current_tcp.copy()
        print(f"✅ Current TCP (mm, rad): {current_tcp.tolist()}")

        if args.target is None:
            print("⚠️  No --target provided; staying at current TCP.")
            target_tcp = current_tcp.copy()
        else:
            target_vals = [float(v) for v in args.target.split(",")]
            if len(target_vals) != 6:
                raise ValueError("target must have 6 values: x,y,z,roll,pitch,yaw")
            target_tcp = np.array(target_vals, dtype=np.float32)

        raw_delta = target_tcp - current_tcp
        delta = safe_clip_delta(raw_delta, args.max_pos_delta, args.max_rot_delta)
        if not np.allclose(delta, raw_delta):
            target_tcp = current_tcp + delta
            print("⚠️  Target clamped by safety limits.")

        print(f"➡️  Move to absolute TCP (mm, rad): {target_tcp.tolist()}")
        current_tcp = run_to_target(
            arm,
            current_tcp,
            target_tcp,
            steps=args.steps,
            rate=args.rate,
            dry_run=args.dry_run,
        )
        time.sleep(0.2)

        if args.return_home:
            print("↩️  Return to start TCP")
            current_tcp = run_to_target(
                arm,
                current_tcp,
                start_tcp,
                steps=args.steps,
                rate=args.rate,
                dry_run=args.dry_run,
            )

        print("✅ TCP motion test complete.")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
