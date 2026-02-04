import os
import torch
import numpy as np
import time
import cv2
import csv
import threading
import sys
import select
import termios
import tty
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from xarm.wrapper import XArmAPI

# 导入 GE 核心推理类
from web_infer_utils.MVActor import MVActor

# --- GE 模型配置 ---
GE_CONFIG = "configs/ltx_model/combined_usb_wipe_tcp/action_model_combined_usb_wipe_tcp_deployment.yaml"
GE_WEIGHTS = "checkpoints/task_combined_usb_wipe_tcp_action/step_30000/diffusion_pytorch_model.safetensors"
DOMAIN_NAME = "combined_usb_wipe_tcp" # 对应 stats.json 中的前缀
NUM_INFERENCE_STEPS = 10  # 采样步数，越大越准但越慢
PROMPT = "wipe the plate"
THRESHOLD = 20  # 参考 LIBERO 脚本

# --- 机器人与环境配置 ---
ROBOT_IP = "192.168.1.209"
BASE_CAM_INDEX = 4 
THIRD_CAM_INDEX = 10
FPS = 30
EXECUTE_STEPS = 30 # 一次推理执行多少步动作
SMOOTHING = 0.15
LOG_DIR = "ge_inference_logs"

# --- 图像尺寸 (需与 GE 训练配置 [192, 256] 对齐) ---
TARGET_H, TARGET_W = 192, 256

def process_image(img):
    """直接缩放到 GE 训练尺寸 (256x192)，不进行裁剪"""
    return cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


class KeyboardListener:
    """非阻塞键盘输入监听器"""
    def __init__(self):
        self.paused = False
        self.visualize_requested = False
        self.running = True
        self.old_settings = None
        self.thread = None
        
    def _get_key(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None
    
    def _listen(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        try:
            while self.running:
                key = self._get_key()
                if key:
                    if key == ' ':
                        self.paused = not self.paused
                        status = "⏸️  已暂停" if self.paused else "▶️  继续运行"
                        print(f"\n{status} (按空格键切换)")
                    elif key.lower() == 'v':
                        self.visualize_requested = True
                        print(f"\n📊 可视化请求已记录...")
                    elif key == '\x03':
                        self.running = False
                        break
                time.sleep(0.01)
        finally:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def start(self):
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        print("⌨️  键盘控制已启用:")
        print("   空格键: 暂停/继续 | V键: 生成可视化 | Ctrl+C: 退出")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)

class TrackingLogger:
    """记录模型预测和实际执行的跟踪数据 (TCP)"""
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"ge_tracking_{self.timestamp}.csv")
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "step", "p_x", "p_y", "p_z", "p_rx", "p_ry", "p_rz", "p_g", "a_x", "a_y", "a_z", "a_rx", "a_ry", "a_rz", "a_g"])
        self.gripper_history = deque(maxlen=100)
        self.full_gripper_data = []
        self.full_tcp_data = []
        self.full_timestamps = []
        
    def log(self, timestamp: float, step: int, pred_tcp: np.ndarray, pred_gripper: float, actual_tcp: np.ndarray, actual_gripper: float):
        self.csv_writer.writerow([f"{timestamp:.6f}", step, *pred_tcp.tolist(), pred_gripper, *actual_tcp.tolist(), actual_gripper])
        self.csv_file.flush()
        error = pred_gripper - actual_gripper
        self.gripper_history.append({'pred': pred_gripper, 'actual': actual_gripper, 'error': error, 'error_abs': abs(error)})
        self.full_gripper_data.append({'pred': pred_gripper, 'actual': actual_gripper, 'error': error})
        self.full_tcp_data.append({'pred': pred_tcp.copy(), 'actual': actual_tcp.copy()})
        self.full_timestamps.append(timestamp)
    
    def get_stats(self):
        if len(self.gripper_history) == 0: return None
        gripper_errors = [h['error_abs'] for h in self.gripper_history]
        return {'gripper_mean_error': np.mean(gripper_errors), 'gripper_max_error': np.max(gripper_errors), 'total_samples': len(self.gripper_history)}

    def generate_visualization(self):
        if len(self.full_gripper_data) == 0: return None
        timestamps = np.array(self.full_timestamps)
        relative_time = timestamps - timestamps[0]
        gripper_pred = np.array([d['pred'] for d in self.full_gripper_data])
        gripper_actual = np.array([d['actual'] for d in self.full_gripper_data])
        
        fig = plt.figure(figsize=(12, 6))
        plt.plot(relative_time, gripper_pred, label='Predicted', color='blue', linestyle='--')
        plt.plot(relative_time, gripper_actual, label='Actual', color='green')
        plt.title('GE Action Tracking: Gripper Position')
        plt.xlabel('Time (s)')
        plt.ylabel('Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(self.log_dir, f"ge_viz_{self.timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        return plot_path

def run():
    print("🚀 正在启动 GE 推理脚本...")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # 1. 初始化 GE Actor
    try:
        actor = MVActor(
            config_file=GE_CONFIG,
            transformer_file=GE_WEIGHTS,
            domain_name=DOMAIN_NAME,
            num_inference_steps=NUM_INFERENCE_STEPS,
            threshold=THRESHOLD,
            action_dim=14, # 对应 config 中的 action_in_channels: 14
            norm_type="minmax"
        )
        print("✅ GE 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 初始化机器人
    try:
        arm = XArmAPI(ROBOT_IP)
        arm.connect()
        arm.clean_gripper_error()
        arm.set_gripper_enable(True)
        arm.set_gripper_mode(0)
        arm.set_gripper_speed(4000)
        arm.motion_enable(True)
        arm.set_mode(1)
        arm.set_state(0)
        print("✅ 机器人连接成功")
    except Exception as e:
        print(f"❌ 机器人连接失败: {e}")
        return

    # 3. 初始化摄像头
    cap_base = cv2.VideoCapture(BASE_CAM_INDEX)
    cap_third = cv2.VideoCapture(THIRD_CAM_INDEX)
    for cap in [cap_base, cap_third]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    logger = TrackingLogger(LOG_DIR)
    kb = KeyboardListener()
    kb.start()

    code, current_tcp = arm.get_position(is_radian=True)
    current_tcp = np.array(current_tcp[:6])
    # 维护角度历史，用 np.unwrap 对齐训练数据处理逻辑
    angle_history = deque(maxlen=2)
    angle_history.append(current_tcp[3:].copy())

    print("✅ 准备就绪！请观察 ge_debug.jpg")

    try:
        while kb.running:
            if kb.paused:
                time.sleep(0.1)
                continue
            
            # A. 图像采集
            for _ in range(2): cap_base.grab(); cap_third.grab()
            ret1, img_b_full = cap_base.retrieve()
            ret2, img_t_full = cap_third.retrieve()
            if not ret1 or not ret2: continue

            # 缩放到 GE 训练尺寸 (256x192) 并转为 RGB
            img_b_proc = process_image(img_b_full)
            img_t_proc = process_image(img_t_full)
            ge_b = cv2.cvtColor(img_b_proc, cv2.COLOR_BGR2RGB)
            ge_t = cv2.cvtColor(img_t_proc, cv2.COLOR_BGR2RGB)
            stacked_obs = np.stack([ge_b, ge_t], axis=0)

            # B. 状态获取
            code, tcp_pos = arm.get_position(is_radian=True)
            _, gpos = arm.get_gripper_position()
            if code != 0: continue
            raw_tcp = np.array(tcp_pos[:6])
            angle_history.append(raw_tcp[3:].copy())
            unwrapped_angles = np.unwrap(np.array(angle_history), axis=0)[-1]
            raw_tcp[3:] = unwrapped_angles
            current_state = np.concatenate([raw_tcp, [gpos]])
            # 格式对齐: [action(7), state(7)]
            # 推理输入时，前7维(action)暂且补零，后7维(state)填入当前状态
            padded_state = np.concatenate([np.zeros(7), current_state])

            # C. 推理
            # GE Actor.play 会自动处理归一化和反归一化
            start_infer = time.time()
            actions = actor.play(
                obs=stacked_obs,
                prompt=PROMPT,
                state=padded_state,
                state_zeropadding=[7, 0],
                ndim_action=14,
                execution_step=EXECUTE_STEPS
            )
            infer_duration = time.time() - start_infer

            # D. 执行动作
            for i in range(EXECUTE_STEPS):
                if not kb.running or kb.paused: break
                
                # 根据 [action, state] 格式，动作在【前 7 维】
                target_tcp = actions[i][:6]
                target_g = actions[i][6]

                # 绝对 TCP 执行：用当前读回位姿作为平滑与安全限制的参考
                code_now, tcp_now = arm.get_position(is_radian=True)
                if code_now == 0:
                    raw_now = np.array(tcp_now[:6])
                    angle_history.append(raw_now[3:].copy())
                    unwrapped_angles = np.unwrap(np.array(angle_history), axis=0)[-1]
                    raw_now[3:] = unwrapped_angles
                    current_tcp = raw_now
                
                # 对角度 (roll, pitch, yaw) 进行 unwrap 处理，确保平滑插值走最短路径
                angle_diff = target_tcp[3:] - current_tcp[3:]
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
                target_tcp[3:] = current_tcp[3:] + angle_diff

                if i == 0:
                    delta_tcp = target_tcp - current_tcp
                    print(f"🧭 TCP abs target: {target_tcp.tolist()}")
                    print(f"🧭 TCP current:   {current_tcp.tolist()}")
                    print(f"🧭 TCP delta:     {delta_tcp.tolist()}")
                    print(f"🧭 State head(7): {padded_state[:7].tolist()}")
                    print(f"🧭 State tail(7): {padded_state[7:].tolist()}")
                    # 验证反归一化是否由于维数对齐而正常
                    print(f"🧭 Pred Raw Action[:7]: {actions[i][:7].tolist()}")

                if np.any(np.abs(target_tcp[:3] - current_tcp[:3]) > 50.0):
                    target_tcp[:3] = np.clip(target_tcp[:3], current_tcp[:3] - 10.0, current_tcp[:3] + 10.0)
                if np.any(np.abs(target_tcp[3:] - current_tcp[3:]) > 0.4):
                    target_tcp[3:] = np.clip(target_tcp[3:], current_tcp[3:] - 0.1, current_tcp[3:] + 0.1)

                executed_tcp = (1 - SMOOTHING) * current_tcp + SMOOTHING * target_tcp
                arm.set_servo_cartesian(executed_tcp, is_radian=True)
                
                if i % 3 == 0:
                    arm.clean_gripper_error()
                    arm.set_gripper_position(target_g, wait=False, speed=4000)
                
                time.sleep(1/FPS)
                
                # 记录
                actual_code, actual_tcp = arm.get_position(is_radian=True)
                _, actual_gpos = arm.get_gripper_position()
                if actual_code == 0:
                    logger.log(time.time(), i, target_tcp, target_g, np.array(actual_tcp[:6]), actual_gpos)
                
                current_tcp = executed_tcp

            # 生成 Debug 图
            pane1 = cv2.resize(img_b_full, (320, 240))
            pane2 = cv2.resize(img_t_full, (320, 240))
            pane3 = np.zeros((240, 320, 3), dtype=np.uint8)
            stats = logger.get_stats()
            if stats:
                cv2.putText(pane3, f"GE Infer: {infer_duration:.2f}s", (10, 30), 1, 1, (0, 255, 255), 1)
                cv2.putText(pane3, f"Err Mean: {stats['gripper_mean_error']:.2f}", (10, 60), 1, 1, (0, 255, 0), 1)
            debug_img = np.hstack([pane1, pane2, pane3])
            cv2.imwrite("ge_debug.jpg", debug_img)

            if kb.visualize_requested:
                kb.visualize_requested = False
                p = logger.generate_visualization()
                print(f"📈 可视化已生成: {p}")

    except KeyboardInterrupt:
        print("\n🛑 用户停止")
    finally:
        kb.stop()
        logger.csv_file.close()
        arm.disconnect()
        cap_base.release()
        cap_third.release()
        print("✅ 程序已退出")

if __name__ == "__main__":
    run()
