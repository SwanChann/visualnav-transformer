"""
绝影Lite3 上位机通讯指令定义模块
基于《绝影Lite3运动主机通讯接口(beta) V1.0.7》
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ============================================================================
# 协议常量
# ============================================================================
DEFAULT_MOTION_HOST_IP = "192.168.2.1"
DEFAULT_MOTION_HOST_PORT = 43893
DEFAULT_DATA_REPORT_PORT = 43897

# CommandHead.type 值
CMD_TYPE_SIMPLE = 0   # 简单指令
CMD_TYPE_COMPLEX = 1  # 复杂指令


# ============================================================================
# 指令码定义 - 控制指令集 (下发到运动主机)
# ============================================================================
class ControlCmd:
    """控制指令码 (Section 1.2)"""

    # 1.2.1 心跳 (≥2Hz)
    HEARTBEAT           = 0x21040001

    # 1.2.2 基本状态转换
    STAND_UP_DOWN       = 0x21010202  # 起立/趴下轮流切换
    SOFT_ESTOP          = 0x21020C0E  # 软急停
    JOINT_BACK_ZERO     = 0x21010C05  # 回零

    # 1.2.3 轴指令 (原地模式 & 移动模式)
    AXIS_PITCH          = 0x21010130  # 俯仰角 / 前后平移
    AXIS_ROLL           = 0x21010131  # 横滚角 / 左右平移
    AXIS_HEIGHT         = 0x21010102  # 身体高度 (仅原地模式)
    AXIS_YAW            = 0x21010135  # 偏航角 / 左右转弯

    # 1.2.4 运动模式切换
    MODE_STAND          = 0x21010D05  # 原地模式
    MODE_WALK           = 0x21010D06  # 移动模式

    # 1.2.5 步态切换
    GAIT_LOW_SPEED      = 0x21010300  # 平地低速步态
    GAIT_MID_SPEED      = 0x21010307  # 平地中速步态
    GAIT_HIGH_SPEED     = 0x21010303  # 平地高速步态
    GAIT_CRAWL_TOGGLE   = 0x21010406  # 正常/匍匐切换
    GAIT_GRIP_OBSTACLE  = 0x21010402  # 抓地越障
    GAIT_GENERAL_OBSTACLE = 0x21010401  # 通用越障
    GAIT_HIGH_STEP      = 0x21010407  # 高踏步越障

    # 1.2.6 动作指令
    ACTION_TWIST_BODY   = 0x21010204  # 扭身体 (力控状态)
    ACTION_ROLL_OVER    = 0x21010205  # 翻身 (趴下状态)
    ACTION_MOONWALK     = 0x2101030C  # 太空步 (力控状态)
    ACTION_BACKFLIP     = 0x21010502  # 后空翻 (趴下状态)
    ACTION_WAVE         = 0x21010507  # 打招呼 (趴下状态)
    ACTION_JUMP_FWD     = 0x2101050B  # 向前跳 (趴下状态)
    ACTION_TWIST_JUMP   = 0x2101020D  # 扭身跳 (力控状态)

    # 1.2.7 控制模式切换
    CTRL_AUTO           = 0x21010C03  # 自主模式 (响应感知主机速度)
    CTRL_MANUAL         = 0x21010C02  # 手动模式 (响应手柄速度)

    # 1.2.8 保存数据
    SAVE_DATA           = 0x21010C01

    # 1.2.9 持续运动
    KEEP_MOVING         = 0x21010C06  # value: -1=开启, 2=关闭

    # 1.2.10 语音指令
    VOICE_CMD           = 0x21010C0A

    # 1.2.10 扬声器指令
    SPEAKER_CMD         = 0x2101030D  # value: 0=关, 1=开, 2=查询

    # 1.2.11 感知设置
    AI_OPTION           = 0x21012109  # value: 0x00=关全部, 0x20=停障, 0xC0=跟随

    # 1.2.12 速度指令 (复杂指令, 自主模式)
    VEL_FORWARD         = 0x0140      # 前后速度 m/s, 范围 [-1.0, 1.0]
    VEL_LATERAL         = 0x0145      # 左右速度 m/s, 范围 [-0.5, 0.5]
    VEL_YAW             = 0x0141      # 旋转角速度 rad/s, 范围 [-1.5, 1.5]


# ============================================================================
# 语音指令值
# ============================================================================
class VoiceValue(IntEnum):
    STAND_UP    = 1
    SIT_DOWN    = 2
    FORWARD     = 3
    BACKWARD    = 4
    LEFT        = 5
    RIGHT       = 6
    STOP        = 7
    HEAD_DOWN   = 8
    HEAD_UP     = 9
    LOOK_LEFT   = 11
    LOOK_RIGHT  = 12
    TURN_LEFT_90  = 13
    TURN_RIGHT_90 = 14
    TURN_BACK_180 = 15
    WAVE        = 22


# ============================================================================
# 接收指令码定义 (运动主机上报)
# ============================================================================
class RecvCmd:
    """接收指令码 (Section 1.3)"""
    ROBOT_STATE         = 0x0901  # 机器人状态, 50Hz
    JOINT_ANGLE         = 0x0902  # 关节角度, 100Hz
    JOINT_VEL           = 0x0903  # 关节角速度, 100Hz


# ============================================================================
# 机器人状态枚举
# ============================================================================
class BasicState(IntEnum):
    """robot_basic_state 枚举"""
    LYING_DOWN      = 1
    PREPARING       = 4
    STANDING_UP     = 5
    FORCE_CONTROL   = 6   # 力控(静止站立)
    LYING_DOWN_ING  = 7
    OUT_OF_CONTROL  = 8
    POSTURE_ADJUST  = 9
    ROLLING_OVER    = 11
    BACK_TO_ZERO    = 17
    BACKFLIP        = 18
    WAVING          = 20


class GaitState(IntEnum):
    """robot_gait_state 枚举"""
    LOW_SPEED       = 0
    GENERAL_OBSTACLE = 2
    MID_SPEED       = 4
    HIGH_SPEED      = 5
    GRIP_OBSTACLE   = 6
    MOONWALK        = 12
    HIGH_STEP       = 13


# ============================================================================
# 数据结构
# ============================================================================

# CommandHead: 12 字节 (3 x uint32_t, 小端)
COMMAND_HEAD_FMT = '<III'
COMMAND_HEAD_SIZE = struct.calcsize(COMMAND_HEAD_FMT)  # 12

# 复杂指令: CommandHead + data
MAX_DATA_SIZE = 256 * 4  # uint32_t data[256] = 1024 bytes


@dataclass
class RobotStateUpload:
    """机器人状态信息 (0x0901, 50Hz)"""
    robot_basic_state: int = 0
    robot_gait_state: int = 0
    rpy: list = field(default_factory=lambda: [0.0, 0.0, 0.0])          # roll, pitch, yaw (°)
    rpy_vel: list = field(default_factory=lambda: [0.0, 0.0, 0.0])      # rad/s
    xyz_acc: list = field(default_factory=lambda: [0.0, 0.0, 0.0])      # m/s²
    pos_world: list = field(default_factory=lambda: [0.0, 0.0, 0.0])    # x(m), y(m), yaw(rad)
    vel_world: list = field(default_factory=lambda: [0.0, 0.0, 0.0])    # m/s, m/s, rad/s
    vel_body: list = field(default_factory=lambda: [0.0, 0.0, 0.0])     # m/s, m/s, rad/s
    touch_down_and_stair_trot: int = 0  # 占位
    is_charging: bool = False            # 占位
    error_state: int = 0                 # 占位
    robot_motion_state: int = 0
    battery_level: float = 0.0           # 0~1
    task_state: int = 0                  # 占位
    is_robot_need_move: bool = False
    zero_position_flag: bool = False
    ultrasound: list = field(default_factory=lambda: [0.0, 0.0])  # forward, backward (m)

    # RobotStateUpload 结构体格式 (小端):
    # int, int, 3d, 3d, 3d, 3d, 3d, 3d, I, ?, I, i, d, i, ?, ?, 2d
    STRUCT_FMT = '<ii3d3d3d3d3d3dI?Iidi??2d'

    @classmethod
    def from_bytes(cls, data: bytes):
        """从二进制数据解析"""
        obj = cls()
        values = struct.unpack(cls.STRUCT_FMT, data[:struct.calcsize(cls.STRUCT_FMT)])
        idx = 0
        obj.robot_basic_state = values[idx]; idx += 1
        obj.robot_gait_state = values[idx]; idx += 1
        obj.rpy = list(values[idx:idx+3]); idx += 3
        obj.rpy_vel = list(values[idx:idx+3]); idx += 3
        obj.xyz_acc = list(values[idx:idx+3]); idx += 3
        obj.pos_world = list(values[idx:idx+3]); idx += 3
        obj.vel_world = list(values[idx:idx+3]); idx += 3
        obj.vel_body = list(values[idx:idx+3]); idx += 3
        obj.touch_down_and_stair_trot = values[idx]; idx += 1
        obj.is_charging = values[idx]; idx += 1
        obj.error_state = values[idx]; idx += 1
        obj.robot_motion_state = values[idx]; idx += 1
        obj.battery_level = values[idx]; idx += 1
        obj.task_state = values[idx]; idx += 1
        obj.is_robot_need_move = values[idx]; idx += 1
        obj.zero_position_flag = values[idx]; idx += 1
        obj.ultrasound = list(values[idx:idx+2]); idx += 2
        return obj


@dataclass
class RobotJointAngle:
    """关节角度 (0x0902, 100Hz), 12个关节, 单位rad"""
    # 顺序: FL侧摆/髋/膝, FR侧摆/髋/膝, HL侧摆/髋/膝, HR侧摆/髋/膝
    joint_angle: list = field(default_factory=lambda: [0.0]*12)

    @classmethod
    def from_bytes(cls, data: bytes):
        obj = cls()
        obj.joint_angle = list(struct.unpack('<12d', data[:96]))
        return obj


@dataclass
class RobotJointVel:
    """关节角速度 (0x0903, 100Hz), 12个关节, 单位rad/s"""
    joint_vel: list = field(default_factory=lambda: [0.0]*12)

    @classmethod
    def from_bytes(cls, data: bytes):
        obj = cls()
        obj.joint_vel = list(struct.unpack('<12d', data[:96]))
        return obj


# ============================================================================
# 协议打包/解包工具函数
# ============================================================================

def pack_simple_cmd(code: int, value: int = 0) -> bytes:
    """
    打包简单指令: [code(4B) | value(4B) | type=0(4B)]
    """
    return struct.pack(COMMAND_HEAD_FMT, code, value, CMD_TYPE_SIMPLE)


def pack_complex_cmd(code: int, data: bytes) -> bytes:
    """
    打包复杂指令: [code(4B) | data_len(4B) | type=1(4B) | data...]
    """
    head = struct.pack(COMMAND_HEAD_FMT, code, len(data), CMD_TYPE_COMPLEX)
    return head + data


def pack_velocity_cmd(code: int, velocity: float) -> bytes:
    """
    打包速度指令 (复杂指令, double类型数据)
    """
    data = struct.pack('<d', velocity)
    return pack_complex_cmd(code, data)


def unpack_command_head(raw: bytes):
    """
    解包 CommandHead, 返回 (code, param_size, type)
    """
    return struct.unpack(COMMAND_HEAD_FMT, raw[:COMMAND_HEAD_SIZE])
