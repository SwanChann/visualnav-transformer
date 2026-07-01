#!/usr/bin/env python3
"""Lite3 real-robot bridge for NoMaD navigation host.

该模块实现 ExternalBridgePlatform 所需的桥接接口，
内部使用 lite3_host_control 的 Lite3Controller 控制机器人运动，
使用 OpenCV 读取 Orin 上的 USB/CSI 相机。

典型用法：
  python scripts/deployment/nomad_navigation_host.py \
    --backend real --interactive \
    --bridge-module deployment.lite3_real_bridge \
    --bridge-class Lite3RealBridge \
    --bridge-config scripts/configs/navigation_host/lite3_real_bridge_config.json
"""

from __future__ import annotations

# Allow direct execution from any scripts/<category>/ path.
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve()
while SCRIPTS_ROOT.name != "scripts" and SCRIPTS_ROOT.parent != SCRIPTS_ROOT:
    SCRIPTS_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

# 确保真机控制与状态机相关模块在 sys.path 中
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SIM_ROOT = SCRIPTS_ROOT / "simulation"
for candidate in (SCRIPTS_ROOT, SIM_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from tooling.project_paths import LITE3_HOST_CONTROL_ROOT

LITE3_CTRL_DIR = LITE3_HOST_CONTROL_ROOT
ctrl_dir_str = str(LITE3_CTRL_DIR)
if ctrl_dir_str not in sys.path:
    sys.path.insert(0, ctrl_dir_str)

from lite3_controller import Lite3Controller, Twist


def _load_camera_calibration(calibration_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one camera calibration JSON file."""
    path = Path(calibration_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    camera_matrix = np.asarray(payload["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.asarray(payload["dist_coeffs"], dtype=np.float32).reshape(-1)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3, got {camera_matrix.shape}")
    if dist_coeffs.size < 4:
        raise ValueError("dist_coeffs must contain at least 4 coefficients")
    return camera_matrix, dist_coeffs


class OrinCamera:
    """Orin 上的相机封装，支持 USB 相机和 CSI 相机。"""

    def __init__(
        self,
        device: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: str = "v4l2",
        fourcc: str | None = "MJPG",
        use_csi: bool = False,
        csi_sensor_id: int = 0,
        csi_flip: int = 0,
        calibration_path: str | None = None,
        undistort_alpha: float = 0.0,
    ) -> None:
        """
        Args:
            device:        USB 相机设备号或 GStreamer pipeline 字符串
            width:         图像宽度
            height:        图像高度
            fps:           帧率
            use_csi:       是否使用 CSI 相机（通过 GStreamer nvarguscamerasrc）
            csi_sensor_id: CSI 传感器编号
            csi_flip:      CSI 图像翻转方式 (0=不翻转, 2=180°)
        """
        self.width = width
        self.height = height
        self.backend = str(backend or "default").lower()
        self.fourcc = None if fourcc in {None, ""} else str(fourcc).upper()
        self.calibration_path = calibration_path
        self.undistort_alpha = float(undistort_alpha)
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self._new_camera_matrix: np.ndarray | None = None
        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None
        self._rectify_ready = False
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_ts = 0.0
        self._last_read_error = ""
        self._frames_captured = 0
        self._capture_fps = 0.0
        self._fps_window_start = time.time()
        self._fps_window_count = 0
        self._closed = False
        self._capture_thread: threading.Thread | None = None

        def open_capture():
            if use_csi:
                # Jetson CSI 相机通过 GStreamer pipeline
                pipeline = (
                    f"nvarguscamerasrc sensor-id={csi_sensor_id} ! "
                    f"video/x-raw(memory:NVMM), width={width}, height={height}, "
                    f"format=NV12, framerate={fps}/1 ! "
                    f"nvvidconv flip-method={csi_flip} ! "
                    f"video/x-raw, width={width}, height={height}, format=BGRx ! "
                    f"videoconvert ! video/x-raw, format=BGR ! "
                    f"appsink drop=true max-buffers=1 sync=false"
                )
                return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            # USB 相机偶尔会在刚释放或刚插入时打开失败，重试可避免误判。
            device_value = int(device) if isinstance(device, (int, str)) and str(device).isdigit() else device
            if self.backend == "v4l2" and hasattr(cv2, "CAP_V4L2"):
                cap = cv2.VideoCapture(device_value, cv2.CAP_V4L2)
            else:
                cap = cv2.VideoCapture(device_value)
            if self.fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc[:4]))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        self.cap = None
        for attempt in range(1, 4):
            self.cap = open_capture()
            if self.cap.isOpened():
                break
            self.cap.release()
            self.cap = None
            if attempt < 3:
                print(f"[OrinCamera] 相机打开失败，重试 {attempt}/3: device={device}, CSI={use_csi}")
                time.sleep(0.5)

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"无法打开相机: device={device}, use_csi={use_csi}")

        if calibration_path:
            self.camera_matrix, self.dist_coeffs = _load_camera_calibration(calibration_path)
            self._prepare_rectify_maps()
            print(f"[OrinCamera] 已加载相机标定文件: {Path(calibration_path).resolve()}")

        # 预热：丢弃前几帧
        for _ in range(5):
            self.cap.read()

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        print(
            f"[OrinCamera] 相机已打开: requested={width}x{height}@{fps}fps, "
            f"actual={actual_w}x{actual_h}@{actual_fps:.1f}fps, CSI={use_csi}, "
            f"backend={self.backend}, fourcc={self.fourcc or 'default'}"
        )

    def _capture_loop(self) -> None:
        """Continuously drain the camera and keep only the newest frame."""
        while not self._closed:
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_frame_ts = time.time()
                    self._last_read_error = ""
                    self._frames_captured += 1
                    self._fps_window_count += 1
                    elapsed = self._latest_frame_ts - self._fps_window_start
                    if elapsed >= 1.0:
                        self._capture_fps = self._fps_window_count / max(elapsed, 1e-6)
                        self._fps_window_start = self._latest_frame_ts
                        self._fps_window_count = 0
            else:
                self._last_read_error = "camera returned no frame"
                time.sleep(0.01)

    def _prepare_rectify_maps(self) -> None:
        if self.camera_matrix is None or self.dist_coeffs is None:
            return
        image_size = (int(self.width), int(self.height))
        self._new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            image_size,
            self.undistort_alpha,
            image_size,
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self._new_camera_matrix,
            image_size,
            cv2.CV_16SC2,
        )
        self._rectify_ready = True

    def read_bgr(self, timeout: float = 2.0, max_age: float = 0.5) -> np.ndarray:
        """返回后台采集线程保存的最新 BGR 帧。"""
        deadline = time.time() + max(float(timeout), 0.0)
        while True:
            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
                frame_ts = self._latest_frame_ts
            if frame is not None:
                frame_age = time.time() - frame_ts
                if frame_age <= max(float(max_age), 0.0):
                    if self._rectify_ready and self._map1 is not None and self._map2 is not None:
                        frame = cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)
                    return frame
            if self._closed:
                raise RuntimeError("[OrinCamera] 相机已经关闭")
            if time.time() >= deadline:
                detail = f": {self._last_read_error}" if self._last_read_error else ""
                raise RuntimeError(f"[OrinCamera] 等待最新相机帧超时或帧已陈旧{detail}")
            time.sleep(0.005)

    def read(self) -> Image.Image:
        """读取最新帧并返回 PIL Image (RGB)。"""
        frame = self.read_bgr()
        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def status(self) -> str:
        with self._frame_lock:
            frame_age = time.time() - self._latest_frame_ts if self._latest_frame is not None else None
            fps = self._capture_fps
            total = self._frames_captured
        if frame_age is None:
            return "camera=no-frame"
        return f"camera_fps={fps:.1f}, frame_age={frame_age:.3f}s, frames={total}"

    def close(self) -> None:
        self._closed = True
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class Lite3RealBridge:
    """
    Lite3 真机桥接类，连接导航主机和绝影 Lite3。

    职责：
      1. 管理 Orin 上相机的采集 (render_camera)
      2. 通过 Lite3Controller 发送 Twist 速度指令 (apply_command)
      3. 提供起立/急停等基础控制
      4. 从运动主机状态上报中获取机器人姿态 (get_pose)

    注意：
      - get_pose 返回的位置是基于里程计的估计值，精度有限
      - 高度和速度来自运动主机上报的状态数据
      - 协议要求速度指令频率 ≥ 20Hz，心跳 ≥ 2Hz
    """

    def __init__(
        self,
        robot_ip: str = "192.168.2.1",
        robot_port: int = 43893,
        local_port: int = 43897,
        enable_camera: bool = True,
        camera_device: int | str = 0,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        camera_backend: str = "v4l2",
        camera_fourcc: str | None = "MJPG",
        use_csi: bool = False,
        csi_sensor_id: int = 0,
        csi_flip: int = 0,
        camera_calibration_path: str | None = None,
        undistort_alpha: float = 0.0,
        twist_hz: float = 25.0,
        max_linear_x: float = 0.2,
        max_yaw_rate: float = 0.6,
        gait: str = "low",
        standup_wait: float = 3.0,
    ) -> None:
        """
        Args:
            robot_ip:       运动主机 IP
            robot_port:     运动主机 UDP 端口
            local_port:     本地监听端口
            camera_device:  相机设备号或 pipeline
            camera_width:   图像宽度
            camera_height:  图像高度
            camera_fps:     帧率
            use_csi:        是否使用 CSI 相机
            csi_sensor_id:  CSI 传感器编号
            csi_flip:       CSI 翻转方式
            twist_hz:       速度指令下发频率
            max_linear_x:   最大前后线速度 (安全限幅)
            max_yaw_rate:   最大偏航角速度 (安全限幅)
            gait:           初始步态 ("low"/"mid"/"high")
            standup_wait:   起立等待时间 (秒)
        """
        # 安全参数
        self.max_linear_x = max_linear_x
        self.max_yaw_rate = max_yaw_rate
        self.twist_hz = twist_hz
        self.gait = gait
        self.standup_wait = standup_wait

        self.camera = None
        if enable_camera:
            self.camera = OrinCamera(
                device=camera_device,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
                backend=camera_backend,
                fourcc=camera_fourcc,
                use_csi=use_csi,
                csi_sensor_id=csi_sensor_id,
                csi_flip=csi_flip,
                calibration_path=camera_calibration_path,
                undistort_alpha=undistort_alpha,
            )
        else:
            print("[Lite3RealBridge] Camera disabled for this host run.")

        # 初始化 Lite3 控制器
        self.ctrl = Lite3Controller(
            robot_ip=robot_ip,
            robot_port=robot_port,
            local_port=local_port,
            auto_heartbeat=True,
            heartbeat_hz=5.0,
        )
        self.ctrl.start()
        print(f"[Lite3RealBridge] 控制器已连接: {robot_ip}:{robot_port}")

        # 里程计累积 (命令自积分，精度有限)
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0
        self._last_odom_time = time.time()
        self._last_vx = 0.0
        self._last_wz = 0.0
        self._is_standing = False
        self._twist_active = False

    def render_camera(self) -> Image.Image:
        """获取当前相机图像。"""
        if self.camera is None:
            raise RuntimeError("Lite3RealBridge camera is disabled for this host run.")
        return self.camera.read()

    def camera_status(self) -> str:
        if self.camera is None:
            return "camera=disabled"
        return self.camera.status()

    def standup(self, duration: float | None = None) -> None:
        """控制机器人起立并准备接收速度指令。

        幂等：若机器人已处于站立态（通过运动主机上报的 basic_state 判定），
        只补齐自主/移动/步态切换，不再下发 stand_up_or_down 切换指令，
        避免对已经站着的机器人意外发出"趴下/起立切换"。

        Args:
            duration: 覆盖默认的等待时间（秒）。实际等待时间的下限为
                      底层协议固定的 3 秒（prepare_for_twist_control 内部 sleep），
                      额外时长取 max(0, duration - 3)。None 时使用 self.standup_wait。
        """
        total_wait = float(duration) if duration is not None else float(self.standup_wait)

        # 允许运动主机上报状态稍晚到达
        wait_start = time.time()
        already_up = bool(self._is_standing)
        if already_up:
            print("[Lite3RealBridge] 已经处于站立状态，补齐自主/移动模式切换")
        else:
            while self.ctrl.robot_state is None and time.time() - wait_start < 2.0:
                time.sleep(0.05)

            state = self.ctrl.robot_state
            if state is not None and hasattr(state, "robot_basic_state"):
                # 6 = 力控(站立)，5 = 正在起立
                if state.robot_basic_state in (5, 6):
                    already_up = True
                    print(
                        f"[Lite3RealBridge] 机器人运动主机上报 basic_state={state.robot_basic_state}，"
                        "已处于站立/起立态，跳过 stand_up_or_down 切换指令"
                    )

        if not already_up:
            print("[Lite3RealBridge] 机器人处于非站立态，执行起立序列...")
            # 复用底层协议流程：心跳 -> 起立 -> 固定 3 秒等待
            self.ctrl.prepare_for_twist_control()
            extra_wait = max(0.0, total_wait - 3.0)
            if extra_wait > 0:
                time.sleep(extra_wait)
        else:
            # 机器人已站立，只做模式切换（这些指令是幂等的）
            self.ctrl.set_auto_mode()
            time.sleep(0.3)
            self.ctrl.set_walk_mode()
            time.sleep(0.3)

        # 设置步态（幂等）
        self.ctrl.set_gait(self.gait)
        time.sleep(0.5)

        self._is_standing = True
        self._last_odom_time = time.time()
        print("[Lite3RealBridge] 起立/模式切换完成，已进入自主+移动模式")

    def apply_command(self, command) -> None:
        """
        接收 MotionCommand 并发送到 Lite3。

        command 包含 linear_x, linear_y, yaw_rate 三个属性。

        安全护栏：
          1. NaN/Inf 检测：任一字段非有限时强制改写为零速度，避免 NaN 经过
             np.clip 再打包成 UDP 帧下发。
          2. 幅值 clip 到 bridge 的安全范围。
          3. 不在此处 sleep；控制循环节拍由上层决定。
        """
        raw = (float(command.linear_x), float(command.linear_y), float(command.yaw_rate))
        if not all(np.isfinite(v) for v in raw):
            print(
                f"[Lite3RealBridge] ⚠️ 检测到非有限命令 (vx={raw[0]}, vy={raw[1]}, wz={raw[2]})，"
                "改为下发零速度"
            )
            raw = (0.0, 0.0, 0.0)

        vx = float(np.clip(raw[0], -self.max_linear_x, self.max_linear_x))
        vy = float(np.clip(raw[1], -0.3, 0.3))
        wz = float(np.clip(raw[2], -self.max_yaw_rate, self.max_yaw_rate))

        twist = Twist(linear_x=vx, linear_y=vy, angular_z=wz)

        if not self._twist_active:
            self.ctrl.start_twist_control(twist, hz=self.twist_hz)
            self._twist_active = True
        else:
            self.ctrl.update_twist(twist)

        # 更新里程计（命令自积分，精度有限，仅供导航粗略参考）
        now = time.time()
        dt = now - self._last_odom_time
        self._odom_x += self._last_vx * np.cos(self._odom_yaw) * dt
        self._odom_y += self._last_vx * np.sin(self._odom_yaw) * dt
        self._odom_yaw += self._last_wz * dt
        self._last_odom_time = now
        self._last_vx = vx
        self._last_wz = wz

    def send_command(self, linear_x: float, linear_y: float, yaw_rate: float) -> None:
        """兼容接口：ExternalBridgePlatform 可能调用 send_command。"""
        # 独立真机部署脚本不强依赖仿真侧 lite3_system 包，避免 Orin 上路径缺失。
        self.apply_command(SimpleNamespace(linear_x=linear_x, linear_y=linear_y, yaw_rate=yaw_rate))

    def prepare_for_twist_control(self) -> None:
        """进入键盘/速度控制前的准备流程。"""
        self.standup(self.standup_wait)

    def set_gait(self, gait: str) -> None:
        """设置 Lite3 步态档位。"""
        if gait not in {"low", "mid", "high"}:
            raise ValueError(f"Unsupported gait: {gait}")
        self.gait = gait
        self.ctrl.set_gait(gait)

    def stop_motion(self) -> None:
        """温和停止当前 Twist 控制。"""
        self.ctrl.stop_twist_control()
        self._twist_active = False
        self._last_vx = 0.0
        self._last_wz = 0.0

    def release_manual_control(self) -> None:
        """释放自主速度控制，让 Lite3 自带手柄恢复运动控制权。"""
        self.stop_motion()
        self.ctrl.set_manual_mode()
        print("[Lite3RealBridge] 已切回手动模式，自带手柄可接管运动控制")

    def get_pose(self):
        """
        返回 (position, yaw)。

        position: np.ndarray shape (2,)，基于命令自积分的 (x, y) 粗估
        yaw: float，偏航角 (rad)。若运动主机上报了 rpy，优先使用上报值；
             否则退回命令自积分的 yaw。

        注意：position 目前始终使用命令自积分，机器人实际受外力或控制未跟上时
              会产生显著漂移，仅可用于 waypoint 局部导航的粗粒度参考，不可
              作为全局定位。
        """
        state = self.ctrl.robot_state
        if state is not None and hasattr(state, "rpy"):
            yaw_rad = float(np.deg2rad(state.rpy[2]))
            return np.array([self._odom_x, self._odom_y]), yaw_rad

        return np.array([self._odom_x, self._odom_y]), float(self._odom_yaw)

    def get_height(self) -> float:
        """返回机器人身体高度估计。"""
        # Lite3 站立高度约 0.35m
        return 0.35

    def get_forward_speed(self) -> float:
        """
        返回当前前向速度（m/s）。

        优先使用运动主机上报的 vel_body[0]；若状态尚未上报或状态陈旧，
        返回 0.0 并打印一次 warning，让上层的 stuck detector 自然判定
        "未在前进"——而不是用命令速度冒充实测速度导致检测失效。
        """
        state = self.ctrl.robot_state
        if state is not None and hasattr(state, "vel_body"):
            return float(state.vel_body[0])
        # 不用 self._last_vx 冒充——会骗过 stuck detector
        if not getattr(self, "_warned_no_vel", False):
            print("[Lite3RealBridge] ⚠️ 运动主机状态未上报 vel_body，get_forward_speed 返回 0")
            self._warned_no_vel = True
        return 0.0

    def is_fallen(self) -> bool:
        """检查机器人是否摔倒。"""
        state = self.ctrl.robot_state
        if state is not None:
            # 基本状态 8 = 失控保护
            if state.robot_basic_state == 8:
                return True
            # 检查横滚角是否异常 (> 45°)
            if hasattr(state, 'rpy') and abs(state.rpy[0]) > 45:
                return True
        return False

    def prepare_task(self, mode: str) -> None:
        """任务切换前的前置清理（由 Lite3System 在构造时调用）。

        场景：上一个任务以 estop/failed 结束，机器人可能处于趴下或失控保护
        状态。如果不重置站立标志，下一轮 StandState 会跳过起立，直接进入
        navigate/explore 对一个趴着的机器人发 Twist，导致"以为在动实际没动"。

        行为：
          - 读取 robot_state.robot_basic_state，若为趴下(1)/正在趴下(7)/失控保护(8)，
            清除 bridge 的 _is_standing 以及 Lite3System 读取的 _nomad_is_standing 标志，
            让后续 StandState 重新真正起立。
          - 若状态未知，保守地不改动标志，交给 StandState 自行处理。
          - 若 twist 后台线程仍在运行而机器人已进入非站立态，顺便停掉 twist，
            避免 close 前持续发送速度指令。
        """
        state = self.ctrl.robot_state
        if state is None or not hasattr(state, "robot_basic_state"):
            return

        basic = int(state.robot_basic_state)
        # 1 = 趴下, 7 = 正在趴下, 8 = 失控保护
        if basic in (1, 7, 8):
            if self._is_standing:
                print(
                    f"[Lite3RealBridge] prepare_task(mode={mode}): "
                    f"机器人实际状态 basic={basic}，清除站立标志以便后续重新起立"
                )
            self._is_standing = False
            if hasattr(self, "_nomad_is_standing"):
                self._nomad_is_standing = False
            if self._twist_active:
                self.ctrl.stop_twist_control()
                self._twist_active = False

    def viewer_alive(self) -> bool:
        """真机模式下始终返回 True（没有 MuJoCo viewer）。"""
        return True

    def emergency_stop(self) -> None:
        """紧急停止：立即停止所有运动。"""
        print("[Lite3RealBridge] ⚠️ 执行急停!")
        if self._twist_active:
            self.ctrl.stop_twist_control()
            self._twist_active = False
        self.ctrl.soft_estop()

    def stop(self) -> None:
        """停止运动（温和版本）。"""
        self.stop_motion()

    def close(self) -> None:
        """释放所有资源。"""
        print("[Lite3RealBridge] 正在释放资源...")
        self.release_manual_control()
        self.ctrl.stop()
        if self.camera is not None:
            self.camera.close()
        print("[Lite3RealBridge] 资源已释放")


# ============================================================================
# 独立测试入口
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lite3 Real Bridge 独立测试")
    parser.add_argument("--robot-ip", default="192.168.2.1")
    parser.add_argument("--camera-device", default="0")
    parser.add_argument("--use-csi", action="store_true")
    parser.add_argument("--test-camera-only", action="store_true", help="仅测试相机，不连接机器人")
    parser.add_argument("--test-standup", action="store_true", help="测试起立流程")
    parser.add_argument("--test-twist", action="store_true", help="测试低速前进")
    args = parser.parse_args()

    if args.test_camera_only:
        print("=== 相机独立测试 ===")
        cam = OrinCamera(device=args.camera_device, use_csi=args.use_csi)
        for i in range(30):
            img = cam.read()
            print(f"  帧 {i}: {img.size}")
        cam.close()
        print("相机测试完成")
        sys.exit(0)

    bridge = Lite3RealBridge(
        robot_ip=args.robot_ip,
        camera_device=args.camera_device,
        use_csi=args.use_csi,
    )

    try:
        # 测试相机
        print("=== 测试相机 ===")
        img = bridge.render_camera()
        print(f"  图像尺寸: {img.size}")

        if args.test_standup:
            print("\n=== 测试起立 ===")
            bridge.standup(3.0)
            time.sleep(2.0)

            if args.test_twist:
                print("\n=== 测试低速前进 (2秒) ===")
                bridge.apply_command(SimpleNamespace(linear_x=0.15, linear_y=0.0, yaw_rate=0.0))
                time.sleep(2.0)
                bridge.apply_command(SimpleNamespace(linear_x=0.0, linear_y=0.0, yaw_rate=0.0))
                print("  停止")

        pos, yaw = bridge.get_pose()
        print(f"\n  位置: ({pos[0]:.3f}, {pos[1]:.3f}), yaw={yaw:.3f}")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        bridge.close()
