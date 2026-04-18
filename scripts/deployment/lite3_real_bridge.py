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

import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# 确保 lite3_host_control 在 sys.path 中
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LITE3_CTRL_DIR = REPO_ROOT / "lite3_host_control"
if str(LITE3_CTRL_DIR) not in sys.path:
    sys.path.insert(0, str(LITE3_CTRL_DIR))

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
        self.calibration_path = calibration_path
        self.undistort_alpha = float(undistort_alpha)
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self._new_camera_matrix: np.ndarray | None = None
        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None
        self._rectify_ready = False

        if use_csi:
            # Jetson CSI 相机通过 GStreamer pipeline
            pipeline = (
                f"nvarguscamerasrc sensor-id={csi_sensor_id} ! "
                f"video/x-raw(memory:NVMM), width={width}, height={height}, "
                f"format=NV12, framerate={fps}/1 ! "
                f"nvvidconv flip-method={csi_flip} ! "
                f"video/x-raw, width={width}, height={height}, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink"
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            # USB 相机
            self.cap = cv2.VideoCapture(int(device) if isinstance(device, (int, str)) and str(device).isdigit() else device)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开相机: device={device}, use_csi={use_csi}")

        if calibration_path:
            self.camera_matrix, self.dist_coeffs = _load_camera_calibration(calibration_path)
            self._prepare_rectify_maps()
            print(f"[OrinCamera] 已加载相机标定文件: {Path(calibration_path).resolve()}")

        # 预热：丢弃前几帧
        for _ in range(5):
            self.cap.read()

        print(f"[OrinCamera] 相机已打开: {width}x{height}@{fps}fps, CSI={use_csi}")

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

    def read(self) -> Image.Image:
        """读取一帧并返回 PIL Image (RGB)。"""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("[OrinCamera] 相机读取失败")
        if self._rectify_ready and self._map1 is not None and self._map2 is not None:
            frame = cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)
        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def close(self) -> None:
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
        robot_ip: str = "192.168.1.120",
        robot_port: int = 43893,
        local_port: int = 43897,
        camera_device: int | str = 0,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        use_csi: bool = False,
        csi_sensor_id: int = 0,
        csi_flip: int = 0,
        camera_calibration_path: str | None = None,
        undistort_alpha: float = 0.0,
        twist_hz: float = 25.0,
        max_linear_x: float = 0.4,
        max_yaw_rate: float = 0.8,
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

        # 初始化相机
        self.camera = OrinCamera(
            device=camera_device,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            use_csi=use_csi,
            csi_sensor_id=csi_sensor_id,
            csi_flip=csi_flip,
            calibration_path=camera_calibration_path,
            undistort_alpha=undistort_alpha,
        )

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

        # 里程计累积 (简单积分，精度有限)
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
        return self.camera.read()

    def standup(self, duration: float = 3.0) -> None:
        """控制机器人起立并准备接收速度指令。"""
        if self._is_standing:
            print("[Lite3RealBridge] 已经处于站立状态")
            return

        print("[Lite3RealBridge] 执行起立序列...")
        self.ctrl.prepare_for_twist_control()
        time.sleep(max(0, self.standup_wait - 3.0))  # prepare_for_twist_control 内已等待3秒

        # 设置步态
        self.ctrl.set_gait(self.gait)
        time.sleep(0.5)

        self._is_standing = True
        self._last_odom_time = time.time()
        print("[Lite3RealBridge] 起立完成，已进入自主+移动模式")

    def apply_command(self, command) -> None:
        """
        接收 MotionCommand 并发送到 Lite3。

        command 包含 linear_x, linear_y, yaw_rate 三个属性。
        """
        vx = float(np.clip(command.linear_x, -self.max_linear_x, self.max_linear_x))
        vy = float(np.clip(command.linear_y, -0.3, 0.3))
        wz = float(np.clip(command.yaw_rate, -self.max_yaw_rate, self.max_yaw_rate))

        twist = Twist(linear_x=vx, linear_y=vy, angular_z=wz)

        if not self._twist_active:
            self.ctrl.start_twist_control(twist, hz=self.twist_hz)
            self._twist_active = True
        else:
            self.ctrl.update_twist(twist)

        # 更新里程计
        now = time.time()
        dt = now - self._last_odom_time
        self._odom_x += self._last_vx * np.cos(self._odom_yaw) * dt
        self._odom_y += self._last_vx * np.sin(self._odom_yaw) * dt
        self._odom_yaw += self._last_wz * dt
        self._last_odom_time = now
        self._last_vx = vx
        self._last_wz = wz

        # NoMaD 导航周期约 67ms，这里 sleep 模拟一个控制周期
        time.sleep(0.067)

    def send_command(self, linear_x: float, linear_y: float, yaw_rate: float) -> None:
        """兼容接口：ExternalBridgePlatform 可能调用 send_command。"""
        from lite3_system.interfaces import MotionCommand
        self.apply_command(MotionCommand(linear_x, linear_y, yaw_rate))

    def get_pose(self):
        """
        返回 (position, yaw)。

        position: np.ndarray shape (2,)，基于里程计的 (x, y) 估计
        yaw: float，偏航角 (rad)

        注意：真机里程计精度有限，仅供导航参考。
        如果运动主机有状态上报，优先使用上报数据。
        """
        state = self.ctrl.robot_state
        if state is not None:
            # 使用运动主机上报的体速度更新里程计
            # state.vel_body[0] = 前后速度, state.rpy[2] = yaw (度)
            yaw_rad = np.deg2rad(state.rpy[2]) if hasattr(state, 'rpy') else self._odom_yaw
            return np.array([self._odom_x, self._odom_y]), float(yaw_rad)

        return np.array([self._odom_x, self._odom_y]), float(self._odom_yaw)

    def get_height(self) -> float:
        """返回机器人身体高度估计。"""
        # Lite3 站立高度约 0.35m
        return 0.35

    def get_forward_speed(self) -> float:
        """返回当前前向速度。"""
        state = self.ctrl.robot_state
        if state is not None and hasattr(state, 'vel_body'):
            return float(state.vel_body[0])
        return float(self._last_vx)

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
        if self._twist_active:
            self.ctrl.stop_twist_control()
            self._twist_active = False

    def close(self) -> None:
        """释放所有资源。"""
        print("[Lite3RealBridge] 正在释放资源...")
        if self._twist_active:
            self.ctrl.stop_twist_control()
            self._twist_active = False
        self.ctrl.stop()
        self.camera.close()
        print("[Lite3RealBridge] 资源已释放")


# ============================================================================
# 独立测试入口
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lite3 Real Bridge 独立测试")
    parser.add_argument("--robot-ip", default="192.168.1.120")
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
                from lite3_system.interfaces import MotionCommand
                bridge.apply_command(MotionCommand(0.15, 0.0, 0.0))
                time.sleep(2.0)
                bridge.apply_command(MotionCommand(0.0, 0.0, 0.0))
                print("  停止")

        pos, yaw = bridge.get_pose()
        print(f"\n  位置: ({pos[0]:.3f}, {pos[1]:.3f}), yaw={yaw:.3f}")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        bridge.close()
