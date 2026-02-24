#!/usr/bin/env python3
"""
xArm Six-axis Force Torque Sensor Test Script

Modes:
  1. monitor  — Real-time force/torque reading (terminal only)
  2. live     — Real-time force/torque with live matplotlib plots
  3. teach    — Admittance control: hand-guide the robot with force feedback
  4. zero     — Set current state as sensor zero point

Usage:
  python scripts/test_ft_sensor.py --ip 192.168.1.209 --mode monitor --duration 30
  python scripts/test_ft_sensor.py --ip 192.168.1.209 --mode live    --duration 60
  python scripts/test_ft_sensor.py --ip 192.168.1.209 --mode teach   --duration 30
  python scripts/test_ft_sensor.py --ip 192.168.1.209 --mode zero
"""

import argparse
import csv
import os
import signal
import sys
import time
import threading
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from xarm.wrapper import XArmAPI


# ── Helpers ──────────────────────────────────────────────────────────────────

LABELS_FORCE = ["Fx (N)", "Fy (N)", "Fz (N)"]
LABELS_TORQUE = ["Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]
LABELS_ALL = ["Fx(N)", "Fy(N)", "Fz(N)", "Tx(Nm)", "Ty(Nm)", "Tz(Nm)"]


def _clear_line():
    sys.stdout.write("\033[2K\r")
    sys.stdout.flush()


def _fmt_vec(v, width=9, prec=3):
    return "  ".join(f"{x:>{width}.{prec}f}" for x in v)


# ── Monitor Mode ─────────────────────────────────────────────────────────────

def run_monitor(arm: XArmAPI, duration: float, rate_hz: float, log_file: str | None):
    """Read and display force/torque data in real-time."""
    print("\n" + "=" * 78)
    print("  Force/Torque Monitor  (Ctrl+C to stop)")
    print("=" * 78)

    arm.set_ft_sensor_enable(1)
    time.sleep(0.3)

    header = f"{'t(s)':>7}  " + "  ".join(f"{l:>9}" for l in LABELS_ALL) + f"  {'|F|(N)':>9}"
    print(header)
    print("-" * len(header))

    dt = 1.0 / max(1e-6, rate_hz)
    rows = []
    t0 = time.time()
    stop = False

    def _sigint(sig, frame):
        nonlocal stop
        stop = True

    prev_handler = signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop:
            elapsed = time.time() - t0
            if elapsed > duration:
                break

            code, ft_data = arm.get_ft_sensor_data()
            if code != 0 or ft_data is None:
                ft_data = arm.ft_ext_force
                if ft_data is None:
                    time.sleep(dt)
                    continue

            ft = np.array(ft_data[:6], dtype=float)
            mag_f = np.linalg.norm(ft[:3])

            line = f"{elapsed:7.2f}  {_fmt_vec(ft)}  {mag_f:9.3f}"
            _clear_line()
            sys.stdout.write(line)
            sys.stdout.flush()

            rows.append({"t": round(elapsed, 4), **{l: round(v, 5) for l, v in zip(LABELS_ALL, ft)}, "|F|": round(mag_f, 5)})
            time.sleep(dt)
    finally:
        signal.signal(signal.SIGINT, prev_handler)
        print()

    arm.set_ft_sensor_enable(0)
    _save_and_summary(rows, log_file)


# ── Live Visualization Mode ─────────────────────────────────────────────────

def run_live(arm: XArmAPI, duration: float, rate_hz: float, log_file: str | None,
             window_s: float = 10.0):
    """Real-time force/torque visualization with matplotlib."""
    import matplotlib
    matplotlib.use("TkAgg")  # interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    print("\n" + "=" * 78)
    print("  Live Force/Torque Visualization  (close window or Ctrl+C to stop)")
    print(f"  Window: {window_s}s  |  Rate: {rate_hz} Hz  |  Duration: {duration}s")
    print("=" * 78)

    arm.set_ft_sensor_enable(1)
    time.sleep(0.3)

    max_pts = int(window_s * rate_hz)
    t_buf = deque(maxlen=max_pts)
    f_bufs = [deque(maxlen=max_pts) for _ in range(6)]
    mag_buf = deque(maxlen=max_pts)
    rows = []
    t0 = time.time()
    lock = threading.Lock()
    stop_event = threading.Event()

    # ── Data collection thread ──
    def _collect():
        dt = 1.0 / max(1e-6, rate_hz)
        while not stop_event.is_set():
            elapsed = time.time() - t0
            if elapsed > duration:
                stop_event.set()
                break
            code, ft_data = arm.get_ft_sensor_data()
            if code != 0 or ft_data is None:
                ft_data = arm.ft_ext_force
            if ft_data is not None:
                ft = np.array(ft_data[:6], dtype=float)
                mag = float(np.linalg.norm(ft[:3]))
                with lock:
                    t_buf.append(elapsed)
                    for i in range(6):
                        f_bufs[i].append(ft[i])
                    mag_buf.append(mag)
                    rows.append({"t": round(elapsed, 4),
                                 **{l: round(v, 5) for l, v in zip(LABELS_ALL, ft)},
                                 "|F|": round(mag, 5)})
            time.sleep(dt)

    collector = threading.Thread(target=_collect, daemon=True)
    collector.start()

    # ── Matplotlib setup ──
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("xArm Force/Torque Sensor — Live", fontsize=14, fontweight="bold")

    # Force subplot
    ax_f = axes[0]
    ax_f.set_ylabel("Force (N)")
    ax_f.set_title("Forces")
    ax_f.grid(True, linestyle=":", alpha=0.5)
    colors_f = ["#e41a1c", "#377eb8", "#4daf4a"]
    lines_f = [ax_f.plot([], [], color=c, linewidth=1.5, label=l)[0]
               for c, l in zip(colors_f, LABELS_FORCE)]
    ax_f.legend(loc="upper left", fontsize=9)

    # Torque subplot
    ax_t = axes[1]
    ax_t.set_ylabel("Torque (Nm)")
    ax_t.set_title("Torques")
    ax_t.grid(True, linestyle=":", alpha=0.5)
    colors_t = ["#ff7f00", "#984ea3", "#a65628"]
    lines_t = [ax_t.plot([], [], color=c, linewidth=1.5, label=l)[0]
               for c, l in zip(colors_t, LABELS_TORQUE)]
    ax_t.legend(loc="upper left", fontsize=9)

    # Magnitude subplot
    ax_m = axes[2]
    ax_m.set_ylabel("|F| (N)")
    ax_m.set_xlabel("Time (s)")
    ax_m.set_title("Force Magnitude")
    ax_m.grid(True, linestyle=":", alpha=0.5)
    line_mag, = ax_m.plot([], [], color="#333333", linewidth=2, label="|F|")
    ax_m.legend(loc="upper left", fontsize=9)

    # Current value text
    txt_f = ax_f.text(0.98, 0.95, "", transform=ax_f.transAxes, ha="right", va="top",
                      fontsize=10, fontfamily="monospace",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    txt_t = ax_t.text(0.98, 0.95, "", transform=ax_t.transAxes, ha="right", va="top",
                      fontsize=10, fontfamily="monospace",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    txt_m = ax_m.text(0.98, 0.95, "", transform=ax_m.transAxes, ha="right", va="top",
                      fontsize=10, fontfamily="monospace",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def _update(frame_idx):
        with lock:
            if len(t_buf) < 2:
                return lines_f + lines_t + [line_mag, txt_f, txt_t, txt_m]
            ts = np.array(t_buf)
            fs = [np.array(f_bufs[i]) for i in range(6)]
            mg = np.array(mag_buf)

        t_min, t_max = ts[-1] - window_s, ts[-1]

        # Update force lines
        for i, ln in enumerate(lines_f):
            ln.set_data(ts, fs[i])
        ax_f.set_xlim(t_min, t_max)
        f_all = np.concatenate(fs[:3])
        if len(f_all) > 0:
            margin = max(1.0, (f_all.max() - f_all.min()) * 0.1)
            ax_f.set_ylim(f_all.min() - margin, f_all.max() + margin)
        txt_f.set_text(f"Fx={fs[0][-1]:+.1f}  Fy={fs[1][-1]:+.1f}  Fz={fs[2][-1]:+.1f}")

        # Update torque lines
        for i, ln in enumerate(lines_t):
            ln.set_data(ts, fs[i + 3])
        ax_t.set_xlim(t_min, t_max)
        t_all = np.concatenate(fs[3:])
        if len(t_all) > 0:
            margin = max(0.1, (t_all.max() - t_all.min()) * 0.1)
            ax_t.set_ylim(t_all.min() - margin, t_all.max() + margin)
        txt_t.set_text(f"Tx={fs[3][-1]:+.2f}  Ty={fs[4][-1]:+.2f}  Tz={fs[5][-1]:+.2f}")

        # Update magnitude line
        line_mag.set_data(ts, mg)
        ax_m.set_xlim(t_min, t_max)
        if len(mg) > 0:
            margin = max(1.0, (mg.max() - mg.min()) * 0.1)
            ax_m.set_ylim(mg.min() - margin, mg.max() + margin)
        txt_m.set_text(f"|F|={mg[-1]:.1f} N")

        return lines_f + lines_t + [line_mag, txt_f, txt_t, txt_m]

    ani = FuncAnimation(fig, _update, interval=50, blit=False, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        collector.join(timeout=2.0)
        arm.set_ft_sensor_enable(0)

    _save_and_summary(rows, log_file)


# ── Teach Mode ───────────────────────────────────────────────────────────────

def run_teach(arm: XArmAPI, duration: float, rate_hz: float, log_file: str | None,
              K_pos: float = 0, K_ori: float = 0,
              M: float = 0.05, live: bool = False):
    """
    Admittance control: set zero stiffness so the robot can be hand-guided.
    Force data is logged during teaching.
    """
    print("\n" + "=" * 78)
    print("  Teach Mode (Admittance)  —  hand-guide the robot")
    print(f"  K_pos={K_pos}  K_ori={K_ori}  M={M}  duration={duration}s")
    print("  Press Ctrl+C to stop early")
    print("=" * 78)

    J = M * 0.01

    arm.set_ft_sensor_admittance_parameters(
        [M, M, M, J, J, J],
        [K_pos, K_pos, K_pos, K_ori, K_ori, K_ori],
        [0] * 6,
    )

    c_axis = [1, 1, 1, 1, 1, 1]
    ref_frame = 0

    arm.set_ft_sensor_enable(1)
    time.sleep(0.3)

    arm.set_ft_sensor_mode(1)
    arm.set_state(0)

    print("\n🤖 Robot is now compliant — you can drag it.\n")

    if live:
        # Use live visualization while teaching
        _teach_live(arm, duration, rate_hz, log_file)
    else:
        _teach_terminal(arm, duration, rate_hz, log_file)

    # Reset
    print("\n🔄 Resetting sensor mode...")
    arm.set_ft_sensor_mode(0)
    arm.set_ft_sensor_enable(0)
    arm.set_state(0)


def _teach_terminal(arm, duration, rate_hz, log_file):
    """Teach with terminal output only."""
    header = f"{'t(s)':>7}  " + "  ".join(f"{l:>9}" for l in LABELS_ALL) + f"  {'|F|(N)':>9}"
    print(header)
    print("-" * len(header))

    dt = 1.0 / max(1e-6, rate_hz)
    rows = []
    t0 = time.time()
    stop = False

    def _sigint(sig, frame):
        nonlocal stop
        stop = True

    prev_handler = signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop:
            elapsed = time.time() - t0
            if elapsed > duration:
                break
            ft_data = arm.ft_ext_force
            if ft_data is not None:
                ft = np.array(ft_data[:6], dtype=float)
                mag_f = np.linalg.norm(ft[:3])
                line = f"{elapsed:7.2f}  {_fmt_vec(ft)}  {mag_f:9.3f}"
                _clear_line()
                sys.stdout.write(line)
                sys.stdout.flush()
                rows.append({"t": round(elapsed, 4), **{l: round(v, 5) for l, v in zip(LABELS_ALL, ft)}, "|F|": round(mag_f, 5)})
            time.sleep(dt)
    finally:
        signal.signal(signal.SIGINT, prev_handler)
        print()

    _save_and_summary(rows, log_file)


def _teach_live(arm, duration, rate_hz, log_file):
    """Teach with live matplotlib visualization."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    max_pts = int(10.0 * rate_hz)
    t_buf = deque(maxlen=max_pts)
    f_bufs = [deque(maxlen=max_pts) for _ in range(6)]
    mag_buf = deque(maxlen=max_pts)
    rows = []
    t0 = time.time()
    lock = threading.Lock()
    stop_event = threading.Event()

    def _collect():
        dt = 1.0 / max(1e-6, rate_hz)
        while not stop_event.is_set():
            elapsed = time.time() - t0
            if elapsed > duration:
                stop_event.set()
                break
            ft_data = arm.ft_ext_force
            if ft_data is not None:
                ft = np.array(ft_data[:6], dtype=float)
                mag = float(np.linalg.norm(ft[:3]))
                with lock:
                    t_buf.append(elapsed)
                    for i in range(6):
                        f_bufs[i].append(ft[i])
                    mag_buf.append(mag)
                    rows.append({"t": round(elapsed, 4),
                                 **{l: round(v, 5) for l, v in zip(LABELS_ALL, ft)},
                                 "|F|": round(mag, 5)})
            time.sleep(dt)

    collector = threading.Thread(target=_collect, daemon=True)
    collector.start()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("xArm Teach Mode — Live Force/Torque", fontsize=14, fontweight="bold")

    colors_f = ["#e41a1c", "#377eb8", "#4daf4a"]
    colors_t = ["#ff7f00", "#984ea3", "#a65628"]

    ax_f, ax_t, ax_m = axes
    ax_f.set_ylabel("Force (N)"); ax_f.set_title("Forces"); ax_f.grid(True, linestyle=":", alpha=0.5)
    ax_t.set_ylabel("Torque (Nm)"); ax_t.set_title("Torques"); ax_t.grid(True, linestyle=":", alpha=0.5)
    ax_m.set_ylabel("|F| (N)"); ax_m.set_xlabel("Time (s)"); ax_m.set_title("Force Magnitude"); ax_m.grid(True, linestyle=":", alpha=0.5)

    lines_f = [ax_f.plot([], [], color=c, linewidth=1.5, label=l)[0] for c, l in zip(colors_f, LABELS_FORCE)]
    lines_t = [ax_t.plot([], [], color=c, linewidth=1.5, label=l)[0] for c, l in zip(colors_t, LABELS_TORQUE)]
    line_mag, = ax_m.plot([], [], color="#333333", linewidth=2, label="|F|")
    ax_f.legend(loc="upper left", fontsize=9)
    ax_t.legend(loc="upper left", fontsize=9)
    ax_m.legend(loc="upper left", fontsize=9)

    txt_f = ax_f.text(0.98, 0.95, "", transform=ax_f.transAxes, ha="right", va="top",
                      fontsize=10, fontfamily="monospace",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    txt_t = ax_t.text(0.98, 0.95, "", transform=ax_t.transAxes, ha="right", va="top",
                      fontsize=10, fontfamily="monospace",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    txt_m = ax_m.text(0.98, 0.95, "", transform=ax_m.transAxes, ha="right", va="top",
                      fontsize=10, fontfamily="monospace",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    window_s = 10.0

    def _update(_):
        with lock:
            if len(t_buf) < 2:
                return lines_f + lines_t + [line_mag, txt_f, txt_t, txt_m]
            ts = np.array(t_buf)
            fs = [np.array(f_bufs[i]) for i in range(6)]
            mg = np.array(mag_buf)

        t_min, t_max = ts[-1] - window_s, ts[-1]
        for i, ln in enumerate(lines_f):
            ln.set_data(ts, fs[i])
        ax_f.set_xlim(t_min, t_max)
        f_all = np.concatenate(fs[:3])
        if len(f_all) > 0:
            margin = max(1.0, (f_all.max() - f_all.min()) * 0.1)
            ax_f.set_ylim(f_all.min() - margin, f_all.max() + margin)
        txt_f.set_text(f"Fx={fs[0][-1]:+.1f}  Fy={fs[1][-1]:+.1f}  Fz={fs[2][-1]:+.1f}")

        for i, ln in enumerate(lines_t):
            ln.set_data(ts, fs[i + 3])
        ax_t.set_xlim(t_min, t_max)
        t_all = np.concatenate(fs[3:])
        if len(t_all) > 0:
            margin = max(0.1, (t_all.max() - t_all.min()) * 0.1)
            ax_t.set_ylim(t_all.min() - margin, t_all.max() + margin)
        txt_t.set_text(f"Tx={fs[3][-1]:+.2f}  Ty={fs[4][-1]:+.2f}  Tz={fs[5][-1]:+.2f}")

        line_mag.set_data(ts, mg)
        ax_m.set_xlim(t_min, t_max)
        if len(mg) > 0:
            margin = max(1.0, (mg.max() - mg.min()) * 0.1)
            ax_m.set_ylim(mg.min() - margin, mg.max() + margin)
        txt_m.set_text(f"|F|={mg[-1]:.1f} N")

        return lines_f + lines_t + [line_mag, txt_f, txt_t, txt_m]

    ani = FuncAnimation(fig, _update, interval=50, blit=False, cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        collector.join(timeout=2.0)

    _save_and_summary(rows, log_file)


# ── Zero Mode ────────────────────────────────────────────────────────────────

def run_zero(arm: XArmAPI):
    """Set current state as sensor zero point."""
    print("\n⚠️  Setting current state as force sensor zero point...")
    arm.set_ft_sensor_enable(1)
    time.sleep(0.3)

    code, ft_before = arm.get_ft_sensor_data()
    if code == 0 and ft_before is not None:
        print(f"  Before zero: {_fmt_vec(ft_before[:6])}")

    arm.set_ft_sensor_zero()
    time.sleep(0.5)

    code, ft_after = arm.get_ft_sensor_data()
    if code == 0 and ft_after is not None:
        print(f"  After zero:  {_fmt_vec(ft_after[:6])}")

    arm.set_ft_sensor_enable(0)
    print("✅ Zero point set.")


# ── Shared Utils ─────────────────────────────────────────────────────────────

def _save_and_summary(rows, log_file):
    """Save CSV and print summary."""
    if log_file and rows:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        with open(log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n📁 Log saved: {log_file}  ({len(rows)} samples)")

    if rows:
        arr = np.array([[r[l] for l in LABELS_ALL] for r in rows])
        print("\n── Summary ──")
        print(f"  Duration : {rows[-1]['t']:.1f} s  |  Samples : {len(rows)}")
        print(f"  Fx range : [{arr[:,0].min():.3f}, {arr[:,0].max():.3f}] N")
        print(f"  Fy range : [{arr[:,1].min():.3f}, {arr[:,1].max():.3f}] N")
        print(f"  Fz range : [{arr[:,2].min():.3f}, {arr[:,2].max():.3f}] N")
        print(f"  |F| max  : {np.linalg.norm(arr[:,:3], axis=1).max():.3f} N")
        print(f"  Tx range : [{arr[:,3].min():.3f}, {arr[:,3].max():.3f}] Nm")
        print(f"  Ty range : [{arr[:,4].min():.3f}, {arr[:,4].max():.3f}] Nm")
        print(f"  Tz range : [{arr[:,5].min():.3f}, {arr[:,5].max():.3f}] Nm")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="xArm Force/Torque Sensor Test")
    parser.add_argument("--ip", type=str, default="192.168.1.209", help="xArm IP address")
    parser.add_argument("--mode", type=str, default="monitor",
                        choices=["monitor", "live", "teach", "zero"],
                        help="Test mode: monitor, live (with plot), teach (admittance), zero")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    parser.add_argument("--rate", type=float, default=50.0, help="Sampling rate in Hz")
    parser.add_argument("--log-file", type=str, default=None, help="Save force data to CSV")
    parser.add_argument("--window", type=float, default=10.0, help="Live plot window in seconds")
    # Teach-mode params
    parser.add_argument("--K-pos", type=float, default=0, help="Linear stiffness (0~2000 N/m)")
    parser.add_argument("--K-ori", type=float, default=0, help="Rotational stiffness (0~20 Nm/rad)")
    parser.add_argument("--mass", type=float, default=0.05, help="Equivalent mass (0.02~1 kg)")
    parser.add_argument("--live", action="store_true", help="Enable live plot in teach mode too")
    args = parser.parse_args()

    # Connect
    print(f"🔌 Connecting to xArm at {args.ip} ...")
    arm = XArmAPI(args.ip)
    arm.motion_enable(enable=True)
    arm.clean_error()
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)
    print(f"✅ Connected. Firmware: {arm.version}")

    try:
        if args.mode == "monitor":
            run_monitor(arm, args.duration, args.rate, args.log_file)
        elif args.mode == "live":
            run_live(arm, args.duration, args.rate, args.log_file, window_s=args.window)
        elif args.mode == "teach":
            run_teach(arm, args.duration, args.rate, args.log_file,
                      K_pos=args.K_pos, K_ori=args.K_ori, M=args.mass,
                      live=args.live)
        elif args.mode == "zero":
            run_zero(arm)
    finally:
        arm.disconnect()
        print("\n🔌 Disconnected.")


if __name__ == "__main__":
    main()
