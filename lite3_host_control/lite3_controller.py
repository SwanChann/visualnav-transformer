"""
绝影Lite3 上位机通讯控制器
基于 UDP 协议与运动主机通讯, 实现 Twist(线速度+角速度) 瞬时控制、启动、急停等功能。
"""

import socket
import struct
import time
import threading
import math
from dataclasses import dataclass
from typing import Optional, Callable

from lite3_command import (
    ControlCmd, RecvCmd, BasicState, GaitState,
    RobotStateUpload, RobotJointAngle, RobotJointVel,
    pack_simple_cmd, pack_complex_cmd, pack_velocity_cmd,
    unpack_command_head, COMMAND_HEAD_SIZE, CMD_TYPE_COMPLEX,
    DEFAULT_MOTION_HOST_IP, DEFAULT_MOTION_HOST_PORT, DEFAULT_DATA_REPORT_PORT,
)


# ============================================================================
# Twist 数据结构 (仿 ROS geometry_msgs/Twist)
# ============================================================================
@dataclass
class Twist:
    """
    线速度 + 角速度

    linear_x:  前后线速度 (m/s), 正值向前, 范围 [-1.0, 1.0]
    linear_y:  左右线速度 (m/s), 正值向左, 范围 [-0.5, 0.5]  (注: 协议中正值向右)
    angular_z: 偏航角速度 (rad/s), 正值左转, 范围 [-1.5, 1.5] (注: 协议中正值向右)
    """
    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0


# ============================================================================
# 上位机控制器
# ============================================================================
class Lite3Controller:
    """
    绝影Lite3 上位机 UDP 控制器

    功能:
        - 心跳维持
        - Twist 瞬时速度控制 (线速度+角速度)
        - 起立 / 趴下 / 软急停
        - 运动模式切换 (原地/移动)
        - 控制模式切换 (自主/手动)
        - 步态切换
        - 状态接收与回调
    """

    def __init__(
        self,
        robot_ip: str = DEFAULT_MOTION_HOST_IP,
        robot_port: int = DEFAULT_MOTION_HOST_PORT,
        local_port: int = DEFAULT_DATA_REPORT_PORT,
        auto_heartbeat: bool = True,
        heartbeat_hz: float = 5.0,
    ):
        """
        Args:
            robot_ip:       运动主机 IP
            robot_port:     运动主机 UDP 端口 (默认 43893)
            local_port:     本地监听端口, 接收运动主机上报数据 (默认 43897)
            auto_heartbeat: 是否自动发送心跳
            heartbeat_hz:   心跳频率 (Hz), 文档要求≥2Hz
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.local_port = local_port

        # --- 发送 socket ---
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # --- 接收 socket ---
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._recv_sock.bind(('0.0.0.0', self.local_port))
        self._recv_sock.settimeout(1.0)

        # --- 状态数据 ---
        self.robot_state: Optional[RobotStateUpload] = None
        self.joint_angle: Optional[RobotJointAngle] = None
        self.joint_vel: Optional[RobotJointVel] = None
        # 连接健康：记录最近一次成功收到 ROBOT_STATE 的墙钟时间；
        # 若长期未更新意味着与运动主机链路异常（断网 / 运动主机崩溃）。
        self._last_state_received_ts: Optional[float] = None
        # 发送侧错误计数与最近一次错误 log 时间（限频，避免刷屏）
        self._send_error_count = 0
        self._last_send_error_log_ts: float = 0.0

        # --- 回调 ---
        self._state_callback: Optional[Callable] = None
        self._joint_callback: Optional[Callable] = None

        # --- 线程控制 ---
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._twist_thread: Optional[threading.Thread] = None

        self._heartbeat_interval = 1.0 / heartbeat_hz
        self._auto_heartbeat = auto_heartbeat

        # --- Twist 控制状态 ---
        self._twist_active = False
        self._current_twist = Twist()
        self._twist_hz = 25.0  # 速度指令下发频率 (文档要求轴指令≥20Hz)
        self._twist_lock = threading.Lock()

    # ========================================================================
    # 连接管理
    # ========================================================================
    def start(self):
        """启动控制器 (心跳 + 数据接收)"""
        if self._running:
            return
        self._running = True

        # 启动接收线程
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        # 启动心跳
        if self._auto_heartbeat:
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

        print(f"[Lite3Controller] 已启动, 目标={self.robot_ip}:{self.robot_port}")

    def stop(self):
        """停止控制器"""
        self._running = False
        self._twist_active = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)
        if self._recv_thread:
            self._recv_thread.join(timeout=2)
        if self._twist_thread:
            self._twist_thread.join(timeout=2)
        self._send_sock.close()
        self._recv_sock.close()
        print("[Lite3Controller] 已停止")

    # ========================================================================
    # 低层发送
    # ========================================================================
    def _send_raw(self, data: bytes):
        """原始发送 UDP 数据包。

        网络异常静默会掩盖 Lite3 断联，这里改为：仍不抛出（否则心跳/twist
        线程直接崩），但累加错误计数并按节流频率打印 warning，让上层可以
        通过 get_send_error_count() 观察链路状态。
        """
        try:
            self._send_sock.sendto(data, (self.robot_ip, self.robot_port))
        except OSError as exc:
            if not self._running:
                return
            self._send_error_count += 1
            now = time.time()
            # 至多每 3 秒打印一次，避免刷屏
            if now - self._last_send_error_log_ts > 3.0:
                print(f"[Lite3Controller] ⚠️ UDP 发送失败 (累计 {self._send_error_count}): {exc}")
                self._last_send_error_log_ts = now

    def send_simple_cmd(self, code: int, value: int = 0):
        """发送简单指令"""
        self._send_raw(pack_simple_cmd(code, value))

    def send_velocity_cmd(self, code: int, velocity: float):
        """发送速度指令 (复杂指令)"""
        self._send_raw(pack_velocity_cmd(code, velocity))

    # ========================================================================
    # 心跳
    # ========================================================================
    def send_heartbeat(self):
        """发送一次心跳"""
        self.send_simple_cmd(ControlCmd.HEARTBEAT)

    def _heartbeat_loop(self):
        """心跳后台线程"""
        while self._running:
            self.send_heartbeat()
            time.sleep(self._heartbeat_interval)

    # ========================================================================
    # 数据接收
    # ========================================================================
    def register_state_callback(self, cb: Callable):
        """注册机器人状态更新回调 cb(state: RobotStateUpload)"""
        self._state_callback = cb

    def register_joint_callback(self, cb: Callable):
        """注册关节数据更新回调 cb(angle: RobotJointAngle, vel: RobotJointVel)"""
        self._joint_callback = cb

    def _recv_loop(self):
        """数据接收后台线程"""
        while self._running:
            try:
                raw, addr = self._recv_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[Lite3Controller] 接收异常: {e}")
                continue

            if len(raw) < COMMAND_HEAD_SIZE:
                continue

            code, param_size, cmd_type = unpack_command_head(raw)

            if cmd_type == 1 and len(raw) >= COMMAND_HEAD_SIZE + param_size:
                data = raw[COMMAND_HEAD_SIZE: COMMAND_HEAD_SIZE + param_size]
                self._dispatch_recv(code, data)

    def _dispatch_recv(self, code: int, data: bytes):
        """分发接收到的数据"""
        if code == RecvCmd.ROBOT_STATE:
            try:
                self.robot_state = RobotStateUpload.from_bytes(data)
                self._last_state_received_ts = time.time()
                if self._state_callback:
                    self._state_callback(self.robot_state)
            except Exception as e:
                print(f"[Lite3Controller] 解析状态数据异常: {e}")

        elif code == RecvCmd.JOINT_ANGLE:
            try:
                self.joint_angle = RobotJointAngle.from_bytes(data)
                if self._joint_callback:
                    self._joint_callback(self.joint_angle, self.joint_vel)
            except Exception as e:
                print(f"[Lite3Controller] 解析关节角度异常: {e}")

        elif code == RecvCmd.JOINT_VEL:
            try:
                self.joint_vel = RobotJointVel.from_bytes(data)
            except Exception as e:
                print(f"[Lite3Controller] 解析关节速度异常: {e}")

    # ========================================================================
    #  核心控制: 起立 / 趴下 / 急停
    # ========================================================================
    def stand_up_or_down(self):
        """起立/趴下 (轮流切换)"""
        self.send_simple_cmd(ControlCmd.STAND_UP_DOWN)
        print("[Lite3Controller] 发送 起立/趴下 指令")

    def soft_estop(self):
        """软急停: 立即停止运动"""
        if self._twist_active:
            self._twist_active = False  # 同时停止 Twist 控制
            if self._twist_thread and self._twist_thread.is_alive():
                self._twist_thread.join(timeout=0.1)
                
        self.send_simple_cmd(ControlCmd.SOFT_ESTOP)
        print("[Lite3Controller] ⚠️ 软急停!")

    def joint_back_zero(self):
        """关节回零"""
        self.send_simple_cmd(ControlCmd.JOINT_BACK_ZERO)
        print("[Lite3Controller] 发送 关节回零 指令")

    # ========================================================================
    #  运动模式切换
    # ========================================================================
    def set_stand_mode(self):
        """切换到原地模式"""
        self.send_simple_cmd(ControlCmd.MODE_STAND)
        print("[Lite3Controller] 切换到 原地模式")

    def set_walk_mode(self):
        """切换到移动模式"""
        self.send_simple_cmd(ControlCmd.MODE_WALK)
        print("[Lite3Controller] 切换到 移动模式")

    # ========================================================================
    #  控制模式切换
    # ========================================================================
    def set_auto_mode(self):
        """切换到自主模式 (响应速度指令)"""
        self.send_simple_cmd(ControlCmd.CTRL_AUTO)
        print("[Lite3Controller] 切换到 自主模式")

    def set_manual_mode(self):
        """切换到手动模式 (响应手柄)"""
        self.send_simple_cmd(ControlCmd.CTRL_MANUAL)
        print("[Lite3Controller] 切换到 手动模式")

    # ========================================================================
    #  步态切换
    # ========================================================================
    def set_gait(self, gait: str):
        """
        切换步态

        Args:
            gait: "low" / "mid" / "high" / "crawl" / "grip" / "general" / "high_step"
        """
        gait_map = {
            "low":       ControlCmd.GAIT_LOW_SPEED,
            "mid":       ControlCmd.GAIT_MID_SPEED,
            "high":      ControlCmd.GAIT_HIGH_SPEED,
            "crawl":     ControlCmd.GAIT_CRAWL_TOGGLE,
            "grip":      ControlCmd.GAIT_GRIP_OBSTACLE,
            "general":   ControlCmd.GAIT_GENERAL_OBSTACLE,
            "high_step": ControlCmd.GAIT_HIGH_STEP,
        }
        code = gait_map.get(gait)
        if code is None:
            print(f"[Lite3Controller] 未知步态: {gait}, 可选: {list(gait_map.keys())}")
            return
        self.send_simple_cmd(code)
        print(f"[Lite3Controller] 切换步态: {gait}")

    # ========================================================================
    #  Twist 瞬时速度控制 (核心功能)
    # ========================================================================
    def send_twist(self, twist: Twist):
        """
        发送一次 Twist 速度指令 (需处于自主模式 + 移动模式)

        内部将 Twist 拆分为 3 条独立的复杂指令:
          - VEL_FORWARD (0x0140): 前后线速度
          - VEL_LATERAL (0x0145): 左右线速度
          - VEL_YAW     (0x0141): 偏航角速度

        注意: 协议中正值向右/向前, 这里统一为
              linear_x > 0 = 向前
              linear_y > 0 = 向左 (协议取反)
              angular_z > 0 = 左转 (协议取反)
        """
        # 限幅
        vx = max(-1.0, min(1.0, twist.linear_x))
        vy = max(-0.5, min(0.5, -twist.linear_y))    # 取反: 协议正值向右
        wz = max(-1.5, min(1.5, -twist.angular_z))   # 取反: 协议正值向右转

        self.send_velocity_cmd(ControlCmd.VEL_FORWARD, vx)
        self.send_velocity_cmd(ControlCmd.VEL_LATERAL, vy)
        self.send_velocity_cmd(ControlCmd.VEL_YAW, wz)

    def start_twist_control(self, twist: Twist, hz: float = 25.0):
        """
        开始持续 Twist 控制 (后台线程以 hz 频率持续下发)

        Args:
            twist: 目标速度
            hz:    下发频率 (≥20Hz, 文档要求轴指令≥20Hz, 超时250ms自动停止)
                   上限 clamp 到 100Hz，避免误传大值导致线程忙等。
        """
        with self._twist_lock:
            self._current_twist = twist
            self._twist_hz = min(100.0, max(20.0, hz))

        if not self._twist_active:
            self._twist_active = True
            self._twist_thread = threading.Thread(target=self._twist_loop, daemon=True)
            self._twist_thread.start()
            print(f"[Lite3Controller] 开始 Twist 控制: vx={twist.linear_x}, vy={twist.linear_y}, wz={twist.angular_z}")

    def update_twist(self, twist: Twist):
        """更新当前 Twist 目标 (控制已启动时)"""
        with self._twist_lock:
            self._current_twist = twist

    def stop_twist_control(self):
        """停止 Twist 控制 (安全结束后台线程后发送零速)"""
        if self._twist_active:
            self._twist_active = False
            if self._twist_thread and self._twist_thread.is_alive():
                # 等待线程安全退窗，避免其在最后一次循环中用旧速度覆盖了下方的"零速度"指令
                self._twist_thread.join(timeout=0.1)
                
        # 确保后台发包完全停止后，发送最终的零速指令让机器人停下
        self.send_twist(Twist(0.0, 0.0, 0.0))
        print("[Lite3Controller] Twist 控制已停止")

    def _twist_loop(self):
        """Twist 持续下发后台线程"""
        interval = 1.0 / self._twist_hz
        while self._twist_active and self._running:
            with self._twist_lock:
                twist = Twist(
                    self._current_twist.linear_x,
                    self._current_twist.linear_y,
                    self._current_twist.angular_z,
                )
            self.send_twist(twist)
            time.sleep(interval)

    # ========================================================================
    #  轴指令 (原地模式下的姿态控制)
    # ========================================================================
    def set_body_height(self, value: int):
        """调整身体高度, 范围[-32767, 32767], 死区[-20000, 20000]"""
        value = max(-32767, min(32767, value))
        self.send_simple_cmd(ControlCmd.AXIS_HEIGHT, value & 0xFFFFFFFF)

    def set_body_pitch(self, value: int):
        """调整俯仰角, 范围[-32767, 32767], 死区[-6553, 6553]"""
        value = max(-32767, min(32767, value))
        self.send_simple_cmd(ControlCmd.AXIS_PITCH, value & 0xFFFFFFFF)

    def set_body_roll(self, value: int):
        """调整横滚角, 范围[-32767, 32767], 死区[-12553, 12553]"""
        value = max(-32767, min(32767, value))
        self.send_simple_cmd(ControlCmd.AXIS_ROLL, value & 0xFFFFFFFF)

    def set_body_yaw(self, value: int):
        """调整偏航角, 范围[-32767, 32767], 死区[-9553, 9553]"""
        value = max(-32767, min(32767, value))
        self.send_simple_cmd(ControlCmd.AXIS_YAW, value & 0xFFFFFFFF)

    # ========================================================================
    #  持续运动
    # ========================================================================
    def enable_keep_moving(self):
        """开启持续运动 (原地踏步)"""
        # 文档: value = -1 (0xFFFFFFFF unsigned) 开启
        self.send_simple_cmd(ControlCmd.KEEP_MOVING, 0xFFFFFFFF)
        print("[Lite3Controller] 开启持续运动")

    def disable_keep_moving(self):
        """关闭持续运动"""
        self.send_simple_cmd(ControlCmd.KEEP_MOVING, 2)
        print("[Lite3Controller] 关闭持续运动")

    # ========================================================================
    #  高级: 完整运动控制流程
    # ========================================================================
    def prepare_for_twist_control(self):
        """
        一键准备 Twist 速度控制 (常用流程):
        1. 发送心跳确认连接
        2. 起立
        3. 等待稳定
        4. 切换到自主模式
        5. 切换到移动模式
        """
        print("[Lite3Controller] === 准备 Twist 速度控制 ===")

        # 1. 心跳
        for _ in range(5):
            self.send_heartbeat()
            time.sleep(0.2)

        # 2. 起立
        self.stand_up_or_down()
        print("[Lite3Controller] 等待起立 (3s)...")
        time.sleep(3.0)

        # 3. 切换到自主模式
        self.set_auto_mode()
        time.sleep(0.5)

        # 4. 切换到移动模式
        self.set_walk_mode()
        time.sleep(0.5)

        print("[Lite3Controller] === 准备完成, 可以发送 Twist 指令 ===")

    # ========================================================================
    #  连接健康
    # ========================================================================
    def is_connection_stale(self, timeout: float = 2.0) -> bool:
        """检查与运动主机的链路是否陈旧。

        True 表示自上次收到 ROBOT_STATE 以来已经超过 timeout 秒（或还从未
        收到过状态）。上层（桥接、导航主机）可以据此触发 emergency_stop 或
        自动退出。
        """
        if self._last_state_received_ts is None:
            return True
        return (time.time() - self._last_state_received_ts) > float(timeout)

    def get_send_error_count(self) -> int:
        """返回累计 UDP 发送错误次数（用于上层监控）。"""
        return int(self._send_error_count)

    # ========================================================================
    #  状态查询
    # ========================================================================
    def get_state_str(self) -> str:
        """获取可读的机器人状态字符串"""
        if self.robot_state is None:
            return "未收到状态数据"

        s = self.robot_state
        state_names = {
            1: "趴下", 4: "准备起立", 5: "正在起立", 6: "力控(站立)",
            7: "正在趴下", 8: "失控保护", 9: "姿态调整",
            11: "翻身中", 17: "回零", 18: "后空翻中", 20: "打招呼中",
        }
        gait_names = {
            0: "低速", 2: "通用越障", 4: "中速", 5: "高速",
            6: "抓地越障", 12: "太空步", 13: "高踏步越障",
        }
        return (
            f"基本状态: {state_names.get(s.robot_basic_state, s.robot_basic_state)} | "
            f"步态: {gait_names.get(s.robot_gait_state, s.robot_gait_state)} | "
            f"电量: {s.battery_level*100:.1f}% | "
            f"RPY: [{s.rpy[0]:.1f}°, {s.rpy[1]:.1f}°, {s.rpy[2]:.1f}°] | "
            f"体速度: [{s.vel_body[0]:.2f}, {s.vel_body[1]:.2f}, {s.vel_body[2]:.2f}] m/s"
        )


# ============================================================================
# 便捷使用示例
# ============================================================================
if __name__ == "__main__":

    print("=" * 60)
    print("绝影Lite3 上位机 Twist 控制示例")
    print("=" * 60)

    # ---- 初始化 ----
    ctrl = Lite3Controller(
        robot_ip="192.168.2.1",
        robot_port=43893,
        local_port=43897,
    )

    # 注册状态回调 (可选)
    def on_state(state: RobotStateUpload):
        pass  # 可以在这里打印或记录状态

    ctrl.register_state_callback(on_state)

    # ---- 启动 ----
    ctrl.start()

    try:
        # 一键准备: 起立 → 自主模式 → 移动模式
        ctrl.prepare_for_twist_control()

        # ==== Twist 控制示例 ====
        # 向前走 0.3 m/s, 持续 3 秒
        print("\n[Demo] 向前行走 0.3m/s, 3秒...")
        ctrl.start_twist_control(Twist(linear_x=0.3))
        time.sleep(3.0)

        # 原地左转 0.5 rad/s, 持续 2 秒
        print("[Demo] 原地左转 0.5rad/s, 2秒...")
        ctrl.update_twist(Twist(angular_z=0.5))
        time.sleep(2.0)

        # 斜向前+右转
        print("[Demo] 斜向移动, 2秒...")
        ctrl.update_twist(Twist(linear_x=0.2, linear_y=-0.1, angular_z=-0.3))
        time.sleep(2.0)

        # 停止运动
        ctrl.stop_twist_control()
        print("[Demo] 运动停止")

        # 打印当前状态
        print(f"\n[状态] {ctrl.get_state_str()}")

        # 趴下
        time.sleep(1.0)
        ctrl.stand_up_or_down()
        print("[Demo] 趴下")

    except KeyboardInterrupt:
        print("\n[Demo] 用户中断, 执行急停...")
        ctrl.soft_estop()

    finally:
        ctrl.stop()
        print("[Demo] 程序结束")
