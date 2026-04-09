#!/usr/bin/env python3
"""
NoMaD + Lite3 MuJoCo 集成仿真导航（纯 Python，无 ROS 依赖）

将 NoMaD 视觉导航策略与 Lite3 四足机器人 RL 运动策略在 MuJoCo 物理引擎中集成，
实现真正的关节级仿真导航。等价于 sdk_deploy 的架构，但不需要 ROS2。

层级架构:
  MuJoCo 相机渲染 (RGB)
     ↓
  NoMaD 视觉编码 + DDPM/DDIM 扩散推理 (4Hz)  ← nomad.pth
     ↓
  PD 速度控制器 → (v_cmd, ω_cmd)
     ↓
  RL 运动策略 (50Hz, ONNX)                    ← policy.onnx
     ↓
  关节 PD 力矩控制 (1000Hz)
     ↓
  MuJoCo 物理仿真

════════════════════════════════════════════════════════════
  运行方式
════════════════════════════════════════════════════════════

  ▸ Step 1: 生成仿真 Topomap（固定地图导航前必须先生成对应地图的 topomap）
    python scripts/nomad_mujoco_lite3_nav.py --mode generate-topomap --map easy
    python scripts/nomad_mujoco_lite3_nav.py --mode generate-topomap --map medium
    python scripts/nomad_mujoco_lite3_nav.py --mode generate-topomap --map hard
    python scripts/nomad_mujoco_lite3_nav.py --mode generate-topomap --map hard --topomap-nodes 30

  ▸ Step 2a: 固定地图导航（NoMaD + RL 联合仿真，到达目标返回 exit 0）
    python scripts/nomad_mujoco_lite3_nav.py --mode navigate --map easy
    python scripts/nomad_mujoco_lite3_nav.py --mode navigate --map medium --no-gui
    python scripts/nomad_mujoco_lite3_nav.py --mode navigate --map hard --max-steps 200

  ▸ Step 2b: DDIM 加速推理（替代 DDPM，步数可选 10/5/3/2/1）
    python scripts/nomad_mujoco_lite3_nav.py --mode navigate --map easy --scheduler ddim
    python scripts/nomad_mujoco_lite3_nav.py --mode navigate --map easy --scheduler ddim --ddim-steps 5
    python scripts/nomad_mujoco_lite3_nav.py --mode navigate --map hard --scheduler ddim --ddim-steps 3

  ▸ Step 2c: 随机出生/目标点模式（自动生成在线 topomap，无需预生成）
    python scripts/nomad_mujoco_lite3_nav.py --mode navigate --map easy --random
    python scripts/nomad_mujoco_lite3_nav.py --mode navigate --map medium --random --scheduler ddim
    python scripts/nomad_mujoco_lite3_nav.py --mode navigate --map hard --random --no-gui

  ▸ 探索模式（NoMaD 无目标随机探索 + RL）
    python scripts/nomad_mujoco_lite3_nav.py --mode explore --map medium
    python scripts/nomad_mujoco_lite3_nav.py --mode explore --no-gui --max-steps 100

  ▸ 纯运动测试（不加载 NoMaD，验证 RL 策略控制行走）
    python scripts/nomad_mujoco_lite3_nav.py --mode walk-test
    python scripts/nomad_mujoco_lite3_nav.py --mode walk-test --no-gui

════════════════════════════════════════════════════════════
  参数说明
════════════════════════════════════════════════════════════

  --mode              运行模式: walk-test | explore | navigate | generate-topomap
  --map               地图选择: easy | medium | hard (默认 easy)
                        easy   = 装饰走廊 5m, 无障碍, 墙壁图案
                        medium = 装饰走廊 7m, 单障碍, 墙壁图案
                        hard   = 装饰走廊 8m, 双障碍, 墙壁图案
  --no-gui            无头模式（不显示 MuJoCo 3D 窗口）
  --max-steps         最大 NoMaD 推理步数 (默认 200 ≈ 50s 仿真时间)
  --waypoint          选择扩散轨迹中第几个路径点 (默认 2, 范围 0~7)
  --standup-time      站立预热时间 (默认 3.0s)
  --save-fpv          保存头部 FPV 相机图像到 results 目录
  --topomap-dir       自定义 topomap 目录 (覆盖 --map 默认路径)
  --topomap-traj      GoStanford 轨迹名 (未指定 --topomap-dir 时使用)
  --topomap-step      数据集 topomap 采样步长 (默认 5)
  --topomap-nodes     generate-topomap / --random 生成节点数 (默认 20)
  --radius            topomap 定位搜索半径 (默认 4)
  --close-threshold   时间距离阈值, <阈值则推进子目标 (默认 3.0)
  --scheduler         扩散推理调度器: ddpm (默认 10步) / ddim (可加速)
  --ddim-steps        DDIM 去噪步数 (默认 10, 可选 5/3/2/1 加速推理)
  --random            随机生成出生点和目标点, 自动在线生成 topomap

════════════════════════════════════════════════════════════
  可视化 (导航模式三画面)
════════════════════════════════════════════════════════════

  1. MuJoCo 3D 窗口 — 机器人仿真画面（--no-gui 关闭）
  2. OpenCV FPV 窗口 — 左半: 头部摄像头实时画面 | 右半: 目标点摄像头画面
  3. 终端日志 — 步数、位置、距离、速度、恢复事件等

════════════════════════════════════════════════════════════
  地图配置 (--map)
════════════════════════════════════════════════════════════

  easy:   走廊 [-2, 7]×[-1.2, 1.2],  目标 (5, 0),  无障碍, 墙壁装饰
  medium: 走廊 [-2, 10]×[-1.5, 1.5], 目标 (7, 0),  障碍@(4,0.6), 墙壁装饰
  hard:   走廊 [-2, 12]×[-1.5, 1.5], 目标 (8, 0),  障碍@(4,0.6) @(6,-0.6), 墙壁装饰

════════════════════════════════════════════════════════════
  关键常量
════════════════════════════════════════════════════════════

  GOAL_REACH_DIST      = 0.5 m      物理距离阈值 (满足 < 0.75m 约束)
  MAX_V / MAX_W        = 0.4 / 0.8  NoMaD PD 最大线速度/角速度
  NOMAD_HZ             = 4 Hz       NoMaD 推理频率
  NUM_DIFFUSION_ITERS  = 10         DDPM/DDIM 训练总步数
  STUCK_VEL_THRESHOLD  = 0.05 m/s   卡住检测本体速度阈值
  STUCK_CHECK_WINDOW   = 3          连续低速判定窗口
  RECOVERY_BACK/TURN   = 8/6 步     碰撞恢复后退/转向步数
"""

import os
import sys
import time
import math
import argparse
from datetime import datetime
from collections import deque

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms
import cv2
import mujoco
import mujoco.viewer
import onnxruntime as ort

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "train"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "deployment/src"))

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# ── 路径常量 ──────────────────────────────────────────────
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_WEIGHTS = os.path.join(PROJECT_ROOT, "deployment/model_weights/nomad.pth")
DATASET_ROOT = os.path.join(PROJECT_ROOT, "nomad_dataset/go_stanford")
LITE3_XML = os.path.join(
    PROJECT_ROOT,
    "sdk_deploy/src/Lite3_sdk_deploy/Lite3_description/lite3_mjcf/mjcf/Lite3.xml",
)
ONNX_POLICY = os.path.join(
    PROJECT_ROOT, "sdk_deploy/src/Lite3_sdk_deploy/policy/policy.onnx"
)

# ── NoMaD 参数 ────────────────────────────────────────────
IMAGE_SIZE = (96, 96)
CONTEXT_SIZE = 3
LEN_TRAJ_PRED = 8
NUM_DIFFUSION_ITERS = 10
NUM_SAMPLES = 8
ENCODING_SIZE = 256
MAX_V = 0.4  # m/s  NoMaD PD controller clamp
MAX_W = 0.8  # rad/s
NOMAD_HZ = 4  # NoMaD inference rate

# ── 碰撞恢复（闭环速度控制）参数 ──────────────────────────
STUCK_VEL_THRESHOLD = 0.05   # 本体速度低于此值视为卡住 (m/s)
STUCK_CHECK_WINDOW = 3       # 连续 N 个 NoMaD 步都低速才判定卡住
RECOVERY_BACK_STEPS = 8      # 后退步数（NoMaD 周期）
RECOVERY_TURN_STEPS = 6      # 转向步数
RECOVERY_BACK_VEL = -0.3     # 后退速度 (m/s)
RECOVERY_TURN_VEL = 0.6      # 转向角速度 (rad/s)
GOAL_REACH_DIST = 0.5        # 物理距离到达判定 (m), 满足 < 0.75m 约束

ACTION_STATS = {
    "min": np.array([-2.5, -4.0]),
    "max": np.array([5.0, 4.0]),
}

_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ── RL 运动策略参数（来自 lite3_policy_runner.hpp）────────
SIM_DT = 0.001  # MuJoCo physics timestep (1000 Hz)
POLICY_DT = 0.02  # RL policy timestep (50 Hz)
POLICY_DECIMATION = int(POLICY_DT / SIM_DT)  # 20 sim steps per policy step
NOMAD_PERIOD_STEPS = int(1.0 / NOMAD_HZ / SIM_DT)  # 250 sim steps per NoMaD cycle
RL_STEPS_PER_NOMAD = int(NOMAD_PERIOD_STEPS / POLICY_DECIMATION)  # ~12

# 关节默认角度（RL 策略训练使用的站立姿态）
DOF_DEFAULT = np.array(
    [0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6],
    dtype=np.float32,
)

# MuJoCo 仿真初始关节角度（来自 mujoco_simulation_ros2.py）
JOINT_INIT = np.array(
    [
        0, -1.35453, 2.54948,
        0, -1.35453, 2.54948,
        0, -1.35453, 2.54948,
        0, -1.35453, 2.54948,
    ],
    dtype=np.float32,
)

# 动作缩放（来自 lite3_policy_runner.hpp）
ACTION_SCALE = np.array(
    [0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
     0.125, 0.25, 0.25, 0.125, 0.25, 0.25],
    dtype=np.float32,
)

OMEGA_SCALE = 0.25  # 角速度缩放
DOF_VEL_SCALE = 0.05  # 关节速度缩放
KP = 30.0  # PD 位置增益
KD = 1.0  # PD 速度增益

# 相机参数
CAM_WIDTH = 320
CAM_HEIGHT_PX = 240

# ── 场景地图配置（3 级难度） ──────────────────────────────
SCENE_MAPS = {
    "easy": {
        "name": "Easy - 装饰走廊 5m 无障碍",
        "corridor_half_width": 1.2,
        "corridor_x_min": -2.0,
        "corridor_x_max": 7.0,
        "wall_height": 0.5,
        "obstacles": [],
        "goal_pos": [5.0, 0.0],
        "robot_start": [0.0, 0.0],
        "topomap_waypoints": None,
        "decorations": [
            # 左墙面板 (深棕木色)
            {"pos": [2.0, 1.13, 0.3], "size": [0.4, 0.03, 0.18], "rgba": [0.45, 0.28, 0.15, 1]},
            # 右墙面板 (深蓝标识)
            {"pos": [3.5, -1.13, 0.3], "size": [0.35, 0.03, 0.15], "rgba": [0.15, 0.22, 0.5, 1]},
            # 靠墙小物体 (暗绿盒子)
            {"pos": [3.0, 1.0, 0.12], "size": [0.08, 0.08, 0.12], "rgba": [0.25, 0.42, 0.2, 1]},
            # 终点墙彩条 (暗红色宽面板)
            {"pos": [6.92, 0.0, 0.3], "size": [0.03, 0.5, 0.22], "rgba": [0.6, 0.18, 0.1, 1]},
        ],
    },
    "medium": {
        "name": "Medium - 装饰走廊 7m 单障碍",
        "corridor_half_width": 1.5,
        "corridor_x_min": -2.0,
        "corridor_x_max": 10.0,
        "wall_height": 0.5,
        "obstacles": [
            {"pos": [4.0, 0.6, 0.25], "size": [0.25, 0.25, 0.25]},
        ],
        "goal_pos": [7.0, 0.0],
        "robot_start": [0.0, 0.0],
        "topomap_waypoints": None,
        "decorations": [
            # 左墙 x=2 (木色)
            {"pos": [2.0, 1.43, 0.3], "size": [0.4, 0.03, 0.18], "rgba": [0.45, 0.28, 0.15, 1]},
            # 右墙 x=3.5 (灰色)
            {"pos": [3.5, -1.43, 0.3], "size": [0.35, 0.03, 0.15], "rgba": [0.5, 0.5, 0.48, 1]},
            # 左墙 x=6 (深蓝)
            {"pos": [6.0, 1.43, 0.28], "size": [0.3, 0.03, 0.15], "rgba": [0.15, 0.22, 0.5, 1]},
            # 靠墙物体 (暗绿盒子)
            {"pos": [4.5, -1.2, 0.1], "size": [0.1, 0.1, 0.1], "rgba": [0.25, 0.42, 0.2, 1]},
            # 靠墙物体 (棕色柱)
            {"pos": [5.5, 1.2, 0.15], "size": [0.07, 0.07, 0.15], "rgba": [0.45, 0.28, 0.15, 1]},
            # 终点墙彩条
            {"pos": [9.92, 0.0, 0.3], "size": [0.03, 0.6, 0.22], "rgba": [0.5, 0.12, 0.1, 1]},
        ],
    },
    "hard": {
        "name": "Hard - 装饰走廊 8m 双障碍",
        "corridor_half_width": 1.5,
        "corridor_x_min": -2.0,
        "corridor_x_max": 12.0,
        "wall_height": 0.5,
        "obstacles": [
            {"pos": [4.0, 0.6, 0.25], "size": [0.2, 0.2, 0.25]},
            {"pos": [6.0, -0.6, 0.25], "size": [0.2, 0.2, 0.25]},
        ],
        "goal_pos": [8.0, 0.0],
        "robot_start": [0.0, 0.0],
        "topomap_waypoints": None,
        "decorations": [
            # 左墙 x=2 (木色)
            {"pos": [2.0, 1.43, 0.3], "size": [0.4, 0.03, 0.18], "rgba": [0.45, 0.28, 0.15, 1]},
            # 右墙 x=3 (灰色)
            {"pos": [3.0, -1.43, 0.3], "size": [0.35, 0.03, 0.15], "rgba": [0.5, 0.5, 0.48, 1]},
            # 左墙 x=5 (深蓝)
            {"pos": [5.0, 1.43, 0.28], "size": [0.3, 0.03, 0.15], "rgba": [0.15, 0.22, 0.5, 1]},
            # 右墙 x=7 (赤褐)
            {"pos": [7.0, -1.43, 0.3], "size": [0.3, 0.03, 0.15], "rgba": [0.6, 0.3, 0.15, 1]},
            # 靠墙物体
            {"pos": [3.5, 1.2, 0.12], "size": [0.08, 0.08, 0.12], "rgba": [0.25, 0.42, 0.2, 1]},
            {"pos": [7.5, -1.2, 0.1], "size": [0.1, 0.1, 0.1], "rgba": [0.45, 0.28, 0.15, 1]},
            # 终点墙彩条 (双色)
            {"pos": [11.92, 0.3, 0.3], "size": [0.03, 0.25, 0.22], "rgba": [0.5, 0.12, 0.1, 1]},
            {"pos": [11.92, -0.3, 0.3], "size": [0.03, 0.25, 0.22], "rgba": [0.15, 0.22, 0.5, 1]},
        ],
    },
}

# 活动场景配置（由 --map 参数在 main() 中设置，默认 easy）
SCENE_CONFIG = SCENE_MAPS["easy"]


def get_default_topomap_dir(map_name):
    """返回指定地图的默认 topomap 存储目录。"""
    return os.path.join(PROJECT_ROOT, "topomaps", map_name)


# ══════════════════════════════════════════════════════════
#  MuJoCo 模型加载（加载地图场景 + 头部 FPV 相机）
# ══════════════════════════════════════════════════════════
def load_lite3_model_with_scene(xml_path, scene_config=None):
    """加载 Lite3.xml 并根据场景配置动态注入地图（墙壁、障碍物）和头部 FPV 相机。

    Args:
        xml_path: Lite3.xml 路径
        scene_config: 场景配置字典（默认使用全局 SCENE_CONFIG）
    """
    if scene_config is None:
        scene_config = SCENE_CONFIG
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    mesh_dir = os.path.abspath(os.path.join(xml_dir, "..", "meshes"))

    with open(xml_path) as f:
        xml_str = f.read()

    # 1. 将 mesh 相对路径替换为绝对路径（from_xml_string 需要）
    xml_str = xml_str.replace('file="../meshes/', f'file="{mesh_dir}/')

    # 2. 在 TORSO body 前端（"头部"）添加第一人称相机（FPV）
    #    Lite3 无独立头部 body，TORSO 前端即为头部位置
    #    pos: x=0.25（超过前腿 hip 0.1745）, z=0.08（高于体中心模拟头部视角）
    #    camera 朝向: body +X (前方), xyaxes 定义 cam X=-Y(右), cam Y=+Z(上)
    fpv_camera = (
        '      <camera name="fpv_camera" pos="0.25 0 0.08" '
        'xyaxes="0 -1 0 0 0 1" fovy="80"/>\n'
    )
    xml_str = xml_str.replace(
        '<site name="TORSO_site"',
        fpv_camera + '      <site name="TORSO_site"',
    )

    # 3. 根据场景配置在 </worldbody> 前注入地图元素
    sc = scene_config
    hw = sc["corridor_half_width"]
    wh = sc["wall_height"]
    x_min = sc["corridor_x_min"]
    x_max = sc["corridor_x_max"]
    cx = (x_min + x_max) / 2
    half_len = (x_max - x_min) / 2
    gx, gy = sc["goal_pos"]

    scene_xml = f"""
    <!-- 走廊墙壁（地面已在 Lite3.xml 中定义） -->
    <body name="wall_left" pos="{cx} {hw} {wh}">
      <geom type="box" size="{half_len} 0.05 {wh}" rgba="0.85 0.82 0.78 1"
            contype="1" conaffinity="1" condim="3"/>
    </body>
    <body name="wall_right" pos="{cx} {-hw} {wh}">
      <geom type="box" size="{half_len} 0.05 {wh}" rgba="0.85 0.82 0.78 1"
            contype="1" conaffinity="1" condim="3"/>
    </body>
    <body name="wall_end" pos="{x_max} 0 {wh}">
      <geom type="box" size="0.05 {hw} {wh}" rgba="0.85 0.82 0.78 1"
            contype="1" conaffinity="1" condim="3"/>
    </body>
    <body name="wall_back" pos="{x_min} 0 {wh}">
      <geom type="box" size="0.05 {hw} {wh}" rgba="0.85 0.82 0.78 1"
            contype="1" conaffinity="1" condim="3"/>
    </body>
"""
    for i, obs in enumerate(sc.get("obstacles", [])):
        ox, oy, oz = obs["pos"]
        sx, sy, sz = obs["size"]
        scene_xml += f"""
    <body name="obstacle{i+1}" pos="{ox} {oy} {oz}">
      <geom type="box" size="{sx} {sy} {sz}" rgba="0.6 0.3 0.1 1"
            contype="1" conaffinity="1" condim="3"/>
    </body>
"""
    # 装饰物 (纯视觉, 不参与碰撞)
    for i, dec in enumerate(sc.get("decorations", [])):
        dx, dy, dz = dec["pos"]
        dsx, dsy, dsz = dec["size"]
        r, g, b, a = dec["rgba"]
        scene_xml += f"""
    <body name="decor_{i}" pos="{dx} {dy} {dz}">
      <geom type="box" size="{dsx} {dsy} {dsz}" rgba="{r} {g} {b} {a}"
            contype="0" conaffinity="0"/>
    </body>
"""

    scene_xml += f"""
    <!-- 目标标记（绿色） -->
    <body name="goal_marker" pos="{gx} {gy} 0.02">
      <geom type="cylinder" size="0.15 0.05" rgba="0 0.9 0 0.8"
            contype="0" conaffinity="0"/>
    </body>
"""
    xml_str = xml_str.replace("</worldbody>", scene_xml + "  </worldbody>")

    # 4. 从修改后的字符串加载模型
    model = mujoco.MjModel.from_xml_string(xml_str)
    print(f"[MuJoCo] Scene loaded: corridor [{x_min}, {x_max}] x [-{hw}, {hw}], "
          f"goal=({gx}, {gy}), {len(sc.get('obstacles', []))} obstacles")
    return model


# ══════════════════════════════════════════════════════════
#  MuJoCo Lite3 仿真环境 + RL 运动策略
# ══════════════════════════════════════════════════════════
class MuJoCoLite3Env:
    """MuJoCo 仿真环境：Lite3 四足机器人 + ONNX RL 运动策略。

    纯 Python 实现，等价于 sdk_deploy 中:
      - mujoco_simulation_ros2.py  (物理仿真 + PD 力矩控制)
      - lite3_policy_runner.hpp    (RL 策略推理 + 观测构建)
    """

    def __init__(self, xml_path=LITE3_XML, onnx_path=ONNX_POLICY, gui=True):
        # ── 加载 MuJoCo 模型（含场景） ──
        print(f"[MuJoCo] Loading model with scene from: {xml_path}")
        self.model = load_lite3_model_with_scene(xml_path)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)

        assert self.model.nu == 12, f"Expected 12 actuators, got {self.model.nu}"

        # ── 初始化姿态 ──
        self._set_initial_pose()

        # ── 加载 ONNX 运动策略 ──
        print(f"[ONNX] Loading locomotion policy: {onnx_path}")
        self.ort_session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

        # ── RL 策略状态 ──
        self.last_action = np.zeros(12, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)  # [vx, vy, wz]

        # ── 渲染器（离屏，用于 NoMaD 视觉输入） ──
        self.renderer = mujoco.Renderer(
            self.model, height=CAM_HEIGHT_PX, width=CAM_WIDTH
        )
        # 找到 fpv_camera 的 ID
        self.fpv_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv_camera"
        )
        assert self.fpv_cam_id >= 0, "fpv_camera not found in model"
        print(f"[MuJoCo] FPV camera id: {self.fpv_cam_id}")

        # ── GUI viewer（可选） ──
        self.gui = gui
        self.viewer = None
        if gui:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                print("[MuJoCo] GUI viewer launched")
            except Exception as e:
                print(f"[MuJoCo] GUI not available: {e}")
                self.gui = False

        # ── 计数器与实时节拍 ──
        self.sim_step = 0
        self.policy_step = 0
        self._wall_t0 = time.time()  # 用于 GUI 实时同步

        mujoco.mj_forward(self.model, self.data)
        print(
            f"[MuJoCo] Ready: SIM_DT={SIM_DT}, POLICY_DT={POLICY_DT}, "
            f"Decimation={POLICY_DECIMATION}, RL/NoMaD={RL_STEPS_PER_NOMAD}"
        )

    def _set_initial_pose(self):
        """设置初始姿态并让机器人自由落地建立接地。"""
        qpos0 = self.data.qpos.copy()
        qpos0[:3] = [0.0, 0.0, 0.35]  # 略高
        qpos0[3:7] = [1.0, 0.0, 0.0, 0.0]  # 朝向 +X
        qpos0[7:19] = JOINT_INIT
        self.data.qpos[:] = qpos0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # 自由落体让四足接地（无控制输入）
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)
        print(
            f"[MuJoCo] After settling: z={self.data.qpos[2]:.3f}, "
            f"contacts={self.data.ncon}"
        )

    # ── 机器人状态获取 ──────────────────────────────────

    def get_robot_state(self):
        """获取机器人状态，用于构建 RL 观测（对标 C++ 的 RobotBasicState）。"""
        # 旋转矩阵（body → world）
        base_quat = self.data.qpos[3:7].copy()  # MuJoCo [w, x, y, z]
        base_rot = np.zeros(9)
        mujoco.mju_quat2Mat(base_rot, base_quat)
        base_rot = base_rot.reshape(3, 3)

        # 体坐标系角速度
        base_omega = self.data.qvel[3:6].copy().astype(np.float32)

        # 重力投影到体坐标系: R^T @ [0, 0, -1]
        projected_gravity = (base_rot.T @ np.array([0.0, 0.0, -1.0])).astype(
            np.float32
        )

        # 关节状态
        joint_pos = self.data.qpos[7:19].copy().astype(np.float32)
        joint_vel = self.data.qvel[6:18].copy().astype(np.float32)

        return {
            "base_omega": base_omega,
            "projected_gravity": projected_gravity,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "base_pos": self.data.qpos[:3].copy(),
            "base_quat": base_quat,
            "base_rot": base_rot,
        }

    # ── RL 策略推理（对标 lite3_policy_runner.hpp）──────

    def build_observation(self, state):
        """构建 45-dim RL 策略观测向量。

        布局（与 lite3_policy_runner.hpp 一致）:
          [0:3]   base_omega * omega_scale
          [3:6]   projected_gravity
          [6:9]   command [vx, vy, wz]
          [9:21]  joint_pos - dof_default
          [21:33] joint_vel * dof_vel_scale
          [33:45] last_action (raw, unscaled)
        """
        obs = np.zeros(45, dtype=np.float32)
        obs[0:3] = state["base_omega"] * OMEGA_SCALE
        obs[3:6] = state["projected_gravity"]
        obs[6:9] = self.command
        obs[9:21] = state["joint_pos"] - DOF_DEFAULT
        obs[21:33] = state["joint_vel"] * DOF_VEL_SCALE
        obs[33:45] = self.last_action
        return obs

    def run_onnx_policy(self, obs):
        """ONNX 推理: obs[1,45] → action[12]。"""
        result = self.ort_session.run(
            None, {"obs": obs.reshape(1, 45).astype(np.float32)}
        )
        return result[0].flatten()

    # ── PD 力矩控制（对标 mujoco_simulation_ros2.py）────

    def apply_pd_torque(self, target_pos):
        """施加 PD 关节力矩控制: τ = Kp*(target - q) + Kd*(0 - dq)。"""
        q = self.data.qpos[7:19]
        dq = self.data.qvel[6:18]
        torque = KP * (target_pos - q) + KD * (0.0 - dq)
        torque = np.clip(torque, -30.0, 30.0)
        self.data.ctrl[:] = torque

    # ── 仿真步进 ──────────────────────────────────────────

    def step_policy(self):
        """执行一步 RL 策略（50Hz）+ POLICY_DECIMATION 步物理仿真。

        流程:
          1. 读取状态 → 构建 45-dim 观测
          2. ONNX 推理 → 12-dim raw action
          3. action * scale + default → 关节目标位置
          4. 20 步 PD 力矩控制 (1000Hz)
        """
        state = self.get_robot_state()
        obs = self.build_observation(state)

        raw_action = self.run_onnx_policy(obs)
        self.last_action = raw_action.copy()

        # 动作后处理: action * scale + default → 关节目标
        target_pos = raw_action * ACTION_SCALE + DOF_DEFAULT

        for _ in range(POLICY_DECIMATION):
            self.apply_pd_torque(target_pos)
            mujoco.mj_step(self.model, self.data)
            self.sim_step += 1

        self.policy_step += 1

        # ── GUI 实时可视化同步 ──
        if self.viewer:
            self.viewer.sync()
            # 实时节拍：等待到 wall-clock 追上仿真时间
            sim_time = self.sim_step * SIM_DT
            wall_elapsed = time.time() - self._wall_t0
            sleep_time = sim_time - wall_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        return state

    def step_nomad_period(self):
        """执行一个 NoMaD 周期（~0.25s = RL_STEPS_PER_NOMAD 个策略步）。"""
        state = None
        for _ in range(RL_STEPS_PER_NOMAD):
            state = self.step_policy()
        return state

    def set_command(self, v_forward, v_side, w_yaw):
        """设置运动指令 [vx, vy, ωz]（单位: m/s, rad/s）。"""
        self.command = np.array(
            [v_forward, v_side, w_yaw], dtype=np.float32
        )

    def reset_wall_clock(self):
        """重置实时时钟基准（用于 GUI 实时同步）。"""
        self._wall_t0 = time.time()
        self.sim_step = 0

    def standup(self, duration=3.0):
        """站立稳定：策略从地面开始同时站立并适应指令。"""
        print(f"[MuJoCo] Warming up policy for {duration:.1f}s...")
        self.reset_wall_clock()
        n_steps = int(duration / POLICY_DT)
        for i in range(n_steps):
            self.step_policy()
            if i % 50 == 0:
                state = self.get_robot_state()
                z = state["base_pos"][2]
                pg = state["projected_gravity"]
                print(
                    f"  warmup step {i:4d}/{n_steps} | z={z:.3f} | "
                    f"gravity_z={pg[2]:.3f}"
                )
        state = self.get_robot_state()
        print(
            f"[MuJoCo] Warmup complete. z={state['base_pos'][2]:.3f}, "
            f"gravity_z={state['projected_gravity'][2]:.3f}"
        )

    # ── 相机渲染 ──────────────────────────────────────────

    def render_camera(self):
        """渲染 FPV 相机图像，返回 PIL Image (RGB)。"""
        self.renderer.update_scene(self.data, camera=self.fpv_cam_id)
        pixels = self.renderer.render()
        return PILImage.fromarray(pixels)

    # ── 位姿获取 ──────────────────────────────────────────

    def get_pose(self):
        """返回 (xy, yaw)。"""
        pos = self.data.qpos[:3].copy()
        w, x, y, z = self.data.qpos[3:7]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.array([pos[0], pos[1]]), yaw

    def get_height(self):
        return float(self.data.qpos[2])

    def is_fallen(self, grace_steps=150):
        """摔倒检测（策略需要 ~3s 从地面站起，grace period 内不检测）。"""
        if self.policy_step < grace_steps:
            return False
        state = self.get_robot_state()
        return state["projected_gravity"][2] > -0.5 or state["base_pos"][2] < 0.1

    # ── 本体速度反馈（闭环控制核心）──────────────────────

    def get_body_velocity(self):
        """获取机器人本体在世界坐标系下的线速度 [vx, vy, vz] (m/s)。

        直接从 MuJoCo qvel[:3] 读取，这是物理引擎计算的真实本体速度，
        包含了碰撞、摩擦等物理效应的影响。用于闭环速度控制。
        """
        return self.data.qvel[:3].copy().astype(np.float32)

    def get_body_speed_forward(self):
        """获取机器人沿自身前进方向的实际速度标量 (m/s)。

        将世界系线速度投影到机器人朝向（body +X），
        正值=前进，负值=后退。撞墙时此值趋近于 0。
        """
        state = self.get_robot_state()
        world_vel = self.data.qvel[:3].copy()  # [vx, vy, vz] world
        # body +X 方向在世界坐标系中的表示 = R @ [1,0,0]
        forward_world = state["base_rot"] @ np.array([1.0, 0.0, 0.0])
        return float(np.dot(world_vel[:2], forward_world[:2]))

    def set_robot_position(self, x, y, yaw=0.0, settle_steps=500):
        """将机器人传送到指定位置并重新落地稳定。"""
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        self.data.qpos[:3] = [x, y, 0.35]
        self.data.qpos[3:7] = [cy, 0.0, 0.0, sy]
        self.data.qpos[7:19] = JOINT_INIT
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

    def move_goal_marker(self, x, y):
        """移动目标点标记到新位置。"""
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker"
        )
        if body_id >= 0:
            self.model.body_pos[body_id] = [x, y, 0.02]
            mujoco.mj_forward(self.model, self.data)

    def close(self):
        if self.viewer:
            try:
                self.viewer.close()
            except Exception:
                pass
        self.renderer.close()
        cv2.destroyAllWindows()

    def viewer_alive(self):
        """检查 GUI 窗口是否仍然打开。"""
        if not self.viewer:
            return True  # 无头模式始终视为存活
        try:
            return self.viewer.is_running()
        except Exception:
            return False


# ══════════════════════════════════════════════════════════
#  碰撞检测 & 闭环速度反馈（卡住检测器）
# ══════════════════════════════════════════════════════════
class StuckDetector:
    """检测机器人是否卡住（碰撞障碍物导致本体速度远低于指令速度）。

    原理: 记录最近 N 个 NoMaD 周期的本体前进速度，
    如果连续低于阈值且指令速度>0，则判定卡住，需要进入恢复状态。
    """

    def __init__(self, window=STUCK_CHECK_WINDOW, vel_threshold=STUCK_VEL_THRESHOLD):
        self.window = window
        self.vel_threshold = vel_threshold
        self.vel_history = deque(maxlen=window)
        self.cmd_history = deque(maxlen=window)

    def update(self, actual_forward_vel, cmd_forward_vel):
        """每个 NoMaD 周期调用一次，记录实际与指令速度。"""
        self.vel_history.append(abs(actual_forward_vel))
        self.cmd_history.append(abs(cmd_forward_vel))

    def is_stuck(self):
        """判断是否卡住: 连续 N 步指令>0 但本体速度接近 0。"""
        if len(self.vel_history) < self.window:
            return False
        # 指令速度均 > 某个最小值（说明在主动前进）
        cmds_active = all(c > 0.05 for c in self.cmd_history)
        # 本体速度均低于阈值（说明没在走）
        vels_low = all(v < self.vel_threshold for v in self.vel_history)
        return cmds_active and vels_low

    def reset(self):
        self.vel_history.clear()
        self.cmd_history.clear()


def show_fpv_realtime(cam_img, step_i, extra_text="", goal_view=None):
    """在 OpenCV 窗口中实时显示头部 FPV 相机画面（可选：含目标点视角）。

    Args:
        cam_img: PIL Image (RGB) - 当前 FPV 画面
        step_i: 当前步数
        extra_text: 叠加显示的状态文字
        goal_view: PIL Image (RGB) - 目标点摄像头画面（可选）
    """
    frame = np.array(cam_img)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # 放大到可视尺寸
    display = cv2.resize(frame_bgr, (640, 480), interpolation=cv2.INTER_NEAREST)
    # 叠加步数和状态信息
    cv2.putText(display, f"Step {step_i} | FPV Camera", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if extra_text:
        cv2.putText(display, extra_text, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if goal_view is not None:
        gframe = np.array(goal_view)
        gframe_bgr = cv2.cvtColor(gframe, cv2.COLOR_RGB2BGR)
        gdisp = cv2.resize(gframe_bgr, (640, 480), interpolation=cv2.INTER_NEAREST)
        cv2.putText(gdisp, "GOAL VIEW (target)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        combined = np.hstack([display, gdisp])
        cv2.imshow("Lite3 Navigation: FPV | Goal", combined)
    else:
        cv2.imshow("Lite3 Head FPV Camera", display)
    cv2.waitKey(1)


# ══════════════════════════════════════════════════════════
#  NoMaD 模型加载与推理（复用 nomad_pybullet_nav.py 逻辑）
# ══════════════════════════════════════════════════════════
def build_nomad_model(device):
    """构建并加载 NoMaD 模型。"""
    print("[NoMaD] Building model...")
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=ENCODING_SIZE,
        context_size=CONTEXT_SIZE,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=ENCODING_SIZE,
        down_dims=[64, 128, 256],
        cond_predict_scale=False,
    )
    dist_pred_net = DenseNetwork(embedding_dim=ENCODING_SIZE)
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_net,
    )
    print(f"[NoMaD] Loading weights: {MODEL_WEIGHTS}")
    state_dict = torch.load(MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print("[NoMaD] Model loaded successfully!")
    return model


def pil_to_tensor(pil_img):
    """PIL Image → [3, 96, 96] 归一化张量。"""
    pil_img = pil_img.convert("RGB").resize(IMAGE_SIZE)
    return _TRANSFORM(pil_img)


def build_obs_tensor(frame_buffer):
    """从帧缓冲构建 [1, (C+1)*3, 96, 96] 观测张量。"""
    frames = list(frame_buffer)
    while len(frames) < CONTEXT_SIZE + 1:
        frames.insert(0, frames[0].clone())
    frames = frames[-(CONTEXT_SIZE + 1) :]
    return torch.cat(frames, dim=0).unsqueeze(0)


def unnormalize_action(ndeltas):
    """扩散输出 → 实际位移。"""
    ndeltas = (ndeltas + 1) / 2.0
    return ndeltas * (ACTION_STATS["max"] - ACTION_STATS["min"]) + ACTION_STATS["min"]


def make_scheduler(kind="ddpm", num_train_timesteps=NUM_DIFFUSION_ITERS):
    """创建扩散调度器 (DDPM 或 DDIM)。"""
    common = dict(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    if kind == "ddpm":
        return DDPMScheduler(**common)
    return DDIMScheduler(**common)


def diffusion_inference(
    model, obs_cond, device,
    scheduler_kind="ddpm",
    num_steps=NUM_DIFFUSION_ITERS,
    num_samples=NUM_SAMPLES,
    guidance_scale=None,
    uncond=None,
):
    """DDPM/DDIM 扩散推理，返回反归一化的路径点 [N, 8, 2]。

    Args:
        scheduler_kind: 'ddpm' 或 'ddim'
        num_steps: 去噪步数（DDIM 可用更少步数加速）
        guidance_scale: CFG 引导强度。None 或 0.0 表示不使用 CFG。
        uncond: 无条件编码（goal masked），用于 CFG。
    """
    scheduler = make_scheduler(scheduler_kind, num_train_timesteps=NUM_DIFFUSION_ITERS)
    cond = obs_cond.repeat(num_samples, 1)
    uncond_rep = None if uncond is None else uncond.repeat(num_samples, 1)
    naction = torch.randn((num_samples, LEN_TRAJ_PRED, 2), device=device)
    scheduler.set_timesteps(num_steps)
    use_cfg = guidance_scale is not None and guidance_scale != 0.0 and uncond_rep is not None
    with torch.no_grad():
        for k in scheduler.timesteps:
            noise_pred = model(
                "noise_pred_net", sample=naction, timestep=k, global_cond=cond
            )
            if use_cfg:
                noise_uncond = model(
                    "noise_pred_net", sample=naction, timestep=k, global_cond=uncond_rep
                )
                noise_pred = (1.0 + guidance_scale) * noise_pred - guidance_scale * noise_uncond
            naction = scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample
    ndeltas = naction.cpu().numpy()
    ndeltas = unnormalize_action(ndeltas)
    return np.cumsum(ndeltas, axis=1)


def pd_controller(waypoint, dt=1.0 / NOMAD_HZ, max_v=MAX_V, max_w=MAX_W):
    """路径点 (dx, dy) → 速度指令 (v, ω)。"""
    dx, dy = float(waypoint[0]), float(waypoint[1])
    EPS = 1e-8
    if abs(dx) < EPS and abs(dy) < EPS:
        return 0.0, 0.0
    elif abs(dx) < EPS:
        v = 0.0
        w = np.sign(dy) * np.pi / (2 * dt)
    else:
        v = dx / dt
        w = np.arctan2(dy, dx) / dt
    v = float(np.clip(v, 0, max_v))
    w = float(np.clip(w, -max_w, max_w))
    return v, w


# ══════════════════════════════════════════════════════════
#  Topomap 加载
# ══════════════════════════════════════════════════════════
def load_topomap_from_dataset(traj_name, step=5):
    """从 GoStanford 数据集加载 topomap 节点。"""
    traj_path = os.path.join(DATASET_ROOT, traj_name)
    if not os.path.isdir(traj_path):
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    images = sorted(
        [f for f in os.listdir(traj_path) if f.endswith((".jpg", ".png"))],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    topomap = []
    for i in range(0, len(images), step):
        img_path = os.path.join(traj_path, images[i])
        topomap.append(PILImage.open(img_path).convert("RGB"))
    print(f"[Topomap] Loaded {len(topomap)} nodes from '{traj_name}' (step={step})")
    return topomap


def load_topomap_from_dir(topo_dir):
    """从本地目录加载 topomap 节点图像（按文件名数字排序）。

    用于加载 generate-topomap 模式生成的仿真场景 topomap，
    确保导航目标图像与仿真地图完全一致。
    """
    if not os.path.isdir(topo_dir):
        raise FileNotFoundError(f"Topomap directory not found: {topo_dir}")
    images = sorted(
        [f for f in os.listdir(topo_dir) if f.endswith((".jpg", ".png"))],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    topomap = []
    for img_name in images:
        img_path = os.path.join(topo_dir, img_name)
        topomap.append(PILImage.open(img_path).convert("RGB"))
    print(f"[Topomap] Loaded {len(topomap)} nodes from '{topo_dir}'")
    return topomap


# ══════════════════════════════════════════════════════════
#  Topomap 生成（从仿真场景采样头部 FPV 图像）
# ══════════════════════════════════════════════════════════
#  随机出生/目标点 & 在线 Topomap 生成
# ══════════════════════════════════════════════════════════
def generate_random_spawn_goal(scene_config, min_dist=3.0, margin=1.0):
    """在走廊范围内随机生成出生点和目标点，确保不与障碍物重叠。

    Args:
        scene_config: 场景配置字典
        min_dist: 起点和终点之间的最小距离 (m)
        margin: 距墙壁/边界的安全边距 (m)
    Returns:
        (spawn_pos, goal_pos): 各为 [x, y] 列表
    """
    x_min = scene_config["corridor_x_min"] + margin
    x_max = scene_config["corridor_x_max"] - margin
    hw = scene_config["corridor_half_width"] - margin * 0.5
    obstacles = scene_config.get("obstacles", [])

    for _ in range(200):
        sx = np.random.uniform(x_min, x_max - min_dist)
        sy = np.random.uniform(-hw, hw)
        gx = np.random.uniform(sx + min_dist, x_max)
        gy = np.random.uniform(-hw, hw)

        valid = True
        for obs in obstacles:
            ox, oy = obs["pos"][0], obs["pos"][1]
            osx, osy = obs["size"][0], obs["size"][1]
            safe = 0.4
            for px, py in [(sx, sy), (gx, gy)]:
                if abs(px - ox) < osx + safe and abs(py - oy) < osy + safe:
                    valid = False
                    break
            if not valid:
                break
        if valid and np.hypot(gx - sx, gy - sy) >= min_dist:
            return [sx, sy], [gx, gy]

    # Fallback 使用默认配置
    return list(scene_config["robot_start"]), list(scene_config["goal_pos"])


def generate_topomap_online(env, spawn_pos, goal_pos, num_nodes=20):
    """在线生成从出生点到目标点的 topomap（内存中，不保存文件）。

    通过在起点到终点之间等间距采样机器人位姿，渲染 FPV 图像，
    构建与实际场景完全一致的 topomap 序列。

    Returns:
        topomap: PIL Image 列表
    """
    stand_z = 0.28
    positions = np.column_stack([
        np.linspace(spawn_pos[0], goal_pos[0], num_nodes),
        np.linspace(spawn_pos[1], goal_pos[1], num_nodes),
    ])
    # 计算朝向目标的 yaw
    dx = goal_pos[0] - spawn_pos[0]
    dy = goal_pos[1] - spawn_pos[1]
    yaw = math.atan2(dy, dx)
    cy, sy_q = math.cos(yaw / 2), math.sin(yaw / 2)

    topomap = []
    for i in range(num_nodes):
        px, py = positions[i]
        env.data.qpos[:3] = [px, py, stand_z]
        env.data.qpos[3:7] = [cy, 0.0, 0.0, sy_q]
        env.data.qpos[7:19] = JOINT_INIT
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        img = env.render_camera()
        topomap.append(img)

    print(f"[OnlineTopomap] Generated {num_nodes} nodes from "
          f"({spawn_pos[0]:.2f},{spawn_pos[1]:.2f}) to ({goal_pos[0]:.2f},{goal_pos[1]:.2f})")
    return topomap


def capture_goal_view(env, goal_pos, yaw=None):
    """在目标点渲染 FPV 画面，用于可视化和最终 topomap 节点对比。

    Args:
        env: MuJoCoLite3Env 实例
        goal_pos: [x, y]
        yaw: 朝向角（None 则朝 +X）
    Returns:
        PIL Image (RGB)
    """
    stand_z = 0.28
    if yaw is None:
        yaw = 0.0
    cy, sy_q = math.cos(yaw / 2), math.sin(yaw / 2)
    env.data.qpos[:3] = [goal_pos[0], goal_pos[1], stand_z]
    env.data.qpos[3:7] = [cy, 0.0, 0.0, sy_q]
    env.data.qpos[7:19] = JOINT_INIT
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    return env.render_camera()


# ══════════════════════════════════════════════════════════
#  Topomap 生成（从仿真场景采样头部 FPV 图像）
# ══════════════════════════════════════════════════════════
def run_generate_topomap(args):
    """在仿真场景中采样头部 FPV 图像，生成与仿真地图一致的 topomap。

    支持两种路径模式:
      - 直线模式: 无障碍地图，沿 x 轴等间距采样
      - 路径模式: 有障碍地图，沿 topomap_waypoints 定义的路径等距采样
    """
    env = MuJoCoLite3Env(gui=not args.no_gui)

    sc = SCENE_CONFIG
    gx, gy = sc["goal_pos"]
    rx, ry = sc["robot_start"]
    num_nodes = args.topomap_nodes
    stand_z = 0.28  # 站立高度

    # 确定保存目录
    topo_dir = args.topomap_dir if args.topomap_dir else get_default_topomap_dir(args.map)
    os.makedirs(topo_dir, exist_ok=True)

    # 计算采样位置（支持直线和路径两种模式）
    waypoints = sc.get("topomap_waypoints")
    if waypoints:
        wps = np.array(waypoints, dtype=float)
        seg_lens = np.linalg.norm(np.diff(wps, axis=0), axis=1)
        cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
        total_len = cum_len[-1]
        sample_dists = np.linspace(0, total_len, num_nodes)
        positions = np.zeros((num_nodes, 2))
        for dim in range(2):
            positions[:, dim] = np.interp(sample_dists, cum_len, wps[:, dim])
        print(f"\n[GenerateTopomap] Map '{args.map}': path mode, {num_nodes} nodes, path_len={total_len:.1f}m")
    else:
        x_positions = np.linspace(rx + 0.5, gx, num_nodes)
        positions = np.column_stack([x_positions, np.full(num_nodes, ry)])
        print(f"\n[GenerateTopomap] Map '{args.map}': straight mode, {num_nodes} nodes, x=[{x_positions[0]:.1f}, {x_positions[-1]:.1f}]")

    print(f"[GenerateTopomap] Save dir: {topo_dir}")

    for i in range(num_nodes):
        px, py = positions[i]
        env.data.qpos[:3] = [px, py, stand_z]
        env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 朝向 +X
        env.data.qpos[7:19] = JOINT_INIT
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        if env.viewer:
            env.viewer.sync()

        img = env.render_camera()
        img.save(os.path.join(topo_dir, f"{i}.png"))

        if i % 5 == 0 or i == num_nodes - 1:
            print(f"  Node {i:3d}/{num_nodes}: pos=({px:.2f}, {py:.2f})")

    print(f"\n[GenerateTopomap] Done! Saved {num_nodes} images to {topo_dir}")
    print(f"  Navigate: --mode navigate --map {args.map}")
    env.close()


# ══════════════════════════════════════════════════════════
#  运动测试模式（不加载 NoMaD）
# ══════════════════════════════════════════════════════════
def run_walk_test(args):
    """纯运动测试：RL 策略控制 Lite3 行走，验证四足运动正确性。"""
    env = MuJoCoLite3Env(gui=not args.no_gui)

    # 设置行走指令（策略会同时完成站立和行走的过渡）
    walk_speed = 0.5  # m/s
    print(f"\n[WalkTest] Walking forward at {walk_speed} m/s (with simultaneous standup)...")
    env.set_command(walk_speed, 0.0, 0.0)

    trajectory = []
    total_rl_steps = args.max_steps * RL_STEPS_PER_NOMAD

    for step_i in range(total_rl_steps):
        env.step_policy()

        if not env.viewer_alive():
            print("[WalkTest] Viewer closed, stopping.")
            break

        if step_i % 50 == 0:
            pos, yaw = env.get_pose()
            z = env.get_height()
            trajectory.append(pos.copy())
            print(
                f"  step {step_i:5d}/{total_rl_steps} | "
                f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}) z={z:.3f} "
                f"yaw={math.degrees(yaw):+.1f}°"
            )

        if env.is_fallen():
            print("[WalkTest] Robot fallen!")
            break

    trajectory = np.array(trajectory)
    if len(trajectory) > 1:
        total_dist = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        print(f"\n[WalkTest] Total distance: {total_dist:.2f} m")

    _save_results(trajectory, [], "walk_test", args)
    env.close()


# ══════════════════════════════════════════════════════════
#  探索模式
# ══════════════════════════════════════════════════════════
def run_explore(args):
    """NoMaD 探索模式：视觉 + 扩散策略 + RL 运动策略联合仿真。

    闭环速度控制: 检测本体实际速度 → 碰撞卡住时进入 RECOVERY →
    后退 + 转向 → 恢复 NAVIGATE 状态。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")

    model = build_nomad_model(device)
    env = MuJoCoLite3Env(gui=not args.no_gui)

    # 设置初始前进指令让策略同时站立和启动
    env.set_command(0.2, 0.0, 0.0)
    env.standup(duration=args.standup_time)
    env.reset_wall_clock()  # warmup 后重置实时时钟

    # FPV 头部视角保存
    fpv_dir = None
    if args.save_fpv:
        fpv_dir = os.path.join(PROJECT_ROOT, f"results/nomad_mujoco/{RUN_TAG}_explore_mujoco/fpv")
        os.makedirs(fpv_dir, exist_ok=True)
        print(f"[Explore] FPV frames will be saved to {fpv_dir}")

    # 闭环: 卡住检测器
    stuck_detector = StuckDetector()
    state_machine = "NAVIGATE"  # NAVIGATE | RECOVERY_BACK | RECOVERY_TURN
    recovery_counter = 0
    recovery_count_total = 0

    frame_buffer = deque(maxlen=CONTEXT_SIZE + 1)
    trajectory = []
    velocity_log = []

    pos, yaw = env.get_pose()
    trajectory.append(pos.copy())
    print(
        f"\n[Explore] Starting at ({pos[0]:.3f}, {pos[1]:.3f}), "
        f"yaw={math.degrees(yaw):.1f}°"
    )
    print(f"[Explore] Max steps: {args.max_steps}")
    print(f"[Explore] Closed-loop: stuck_threshold={STUCK_VEL_THRESHOLD} m/s, window={STUCK_CHECK_WINDOW}")
    print("=" * 60)

    for step_i in range(args.max_steps):
        t0 = time.time()

        # 1. 渲染头部 FPV 相机
        cam_img = env.render_camera()
        if fpv_dir and step_i % 5 == 0:
            cam_img.save(os.path.join(fpv_dir, f"{step_i:04d}.png"))

        # ── 实时 FPV 显示 ──
        actual_vel = env.get_body_speed_forward()
        status_text = f"State: {state_machine} | v_body={actual_vel:.3f} m/s"
        show_fpv_realtime(cam_img, step_i, status_text)

        # ── 状态机: 碰撞恢复 ──
        if state_machine == "RECOVERY_BACK":
            env.set_command(RECOVERY_BACK_VEL, 0.0, 0.0)
            env.step_nomad_period()
            recovery_counter += 1
            pos, yaw = env.get_pose()
            trajectory.append(pos.copy())
            velocity_log.append((RECOVERY_BACK_VEL, 0.0))
            if step_i % 2 == 0:
                print(f"  Step {step_i:4d} | RECOVERY_BACK {recovery_counter}/{RECOVERY_BACK_STEPS} | "
                      f"v_body={actual_vel:.3f}")
            if recovery_counter >= RECOVERY_BACK_STEPS:
                state_machine = "RECOVERY_TURN"
                recovery_counter = 0
                # 随机选择转向方向
                turn_dir = 1.0 if np.random.random() > 0.5 else -1.0
                print(f"  [Recovery] Switching to TURN ({'left' if turn_dir > 0 else 'right'})")
            continue

        if state_machine == "RECOVERY_TURN":
            env.set_command(0.05, 0.0, turn_dir * RECOVERY_TURN_VEL)
            env.step_nomad_period()
            recovery_counter += 1
            pos, yaw = env.get_pose()
            trajectory.append(pos.copy())
            velocity_log.append((0.05, turn_dir * RECOVERY_TURN_VEL))
            if step_i % 2 == 0:
                print(f"  Step {step_i:4d} | RECOVERY_TURN {recovery_counter}/{RECOVERY_TURN_STEPS}")
            if recovery_counter >= RECOVERY_TURN_STEPS:
                state_machine = "NAVIGATE"
                recovery_counter = 0
                stuck_detector.reset()
                print(f"  [Recovery] Done! Resuming NAVIGATE")
            continue

        # ── 正常导航状态 ──
        frame_tensor = pil_to_tensor(cam_img)
        frame_buffer.append(frame_tensor)

        if len(frame_buffer) < 2:
            env.step_nomad_period()
            continue

        # 2. NoMaD 视觉编码（探索模式: mask=1）
        obs_tensor = build_obs_tensor(frame_buffer).to(device)
        fake_goal = torch.randn((1, 3, *IMAGE_SIZE)).to(device)
        mask = torch.ones(1).long().to(device)

        with torch.no_grad():
            obs_cond = model(
                "vision_encoder",
                obs_img=obs_tensor,
                goal_img=fake_goal,
                input_goal_mask=mask,
            )

        # 3. 扩散推理 → 路径点
        sched = getattr(args, "scheduler", "ddpm")
        ddim_steps = getattr(args, "ddim_steps", NUM_DIFFUSION_ITERS)
        n_steps = ddim_steps if sched == "ddim" else NUM_DIFFUSION_ITERS
        actions = diffusion_inference(model, obs_cond, device,
                                      scheduler_kind=sched, num_steps=n_steps)
        mean_action = actions.mean(axis=0)
        wp_idx = min(args.waypoint, LEN_TRAJ_PRED - 1)
        chosen_wp = mean_action[wp_idx]

        # 4. PD 速度控制器 → (v, ω)
        v, w = pd_controller(chosen_wp)

        # 5. 设置命令 → 执行一个 NoMaD 周期的 RL 策略步
        env.set_command(v, 0.0, w)
        env.step_nomad_period()

        # 6. 闭环: 读取本体实际速度，检测卡住
        actual_vel = env.get_body_speed_forward()
        stuck_detector.update(actual_vel, v)

        # 6. 记录
        pos, yaw = env.get_pose()
        trajectory.append(pos.copy())
        velocity_log.append((v, w))

        elapsed = time.time() - t0
        if step_i % 5 == 0:
            z = env.get_height()
            print(
                f"  Step {step_i:4d} | pos=({pos[0]:+.3f}, {pos[1]:+.3f}) "
                f"z={z:.3f} yaw={math.degrees(yaw):+.1f}° | "
                f"wp=({chosen_wp[0]:+.3f}, {chosen_wp[1]:+.3f}) | "
                f"v_cmd={v:.3f} v_body={actual_vel:.3f} w={w:.3f} | {elapsed:.3f}s"
            )

        # 闭环: 检测卡住 → 进入恢复
        if stuck_detector.is_stuck():
            recovery_count_total += 1
            print(f"  [!] STUCK detected at step {step_i}! "
                  f"v_body={actual_vel:.3f} << v_cmd={v:.3f} | "
                  f"Recovery #{recovery_count_total}")
            state_machine = "RECOVERY_BACK"
            recovery_counter = 0
            turn_dir = 1.0  # 初始化，RECOVERY_TURN 阶段会覆盖

        if env.is_fallen():
            print(f"  [!] Robot fallen at step {step_i}!")
            break

        if not env.viewer_alive():
            print("[Explore] Viewer closed, stopping.")
            break

    print(f"\n[Explore] Total recovery events: {recovery_count_total}")
    _save_results(trajectory, velocity_log, "explore_mujoco", args)
    env.close()


# ══════════════════════════════════════════════════════════
#  导航模式
# ══════════════════════════════════════════════════════════
def run_navigate(args):
    """NoMaD 导航模式：沿 topomap 目标导向导航 + RL 运动策略。

    闭环速度控制: 检测本体实际速度 → 碰撞卡住时进入 RECOVERY →
    后退 + 转向 → 恢复 NAVIGATE 状态。
    支持 DDPM/DDIM、随机出生/目标点、三画面可视化。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")
    print(f"[Main] Scheduler: {args.scheduler.upper()}"
          + (f" ({args.ddim_steps} steps)" if args.scheduler == "ddim" else " (10 steps)"))

    model = build_nomad_model(device)
    env = MuJoCoLite3Env(gui=not args.no_gui)

    # ── 随机出生/目标点 ──
    goal_view_img = None  # 目标点摄像头画面
    if args.random:
        spawn_pos, goal_pos = generate_random_spawn_goal(SCENE_CONFIG)
        print(f"[Random] Spawn: ({spawn_pos[0]:.2f}, {spawn_pos[1]:.2f})")
        print(f"[Random] Goal:  ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
        print(f"[Random] Distance: {np.hypot(goal_pos[0]-spawn_pos[0], goal_pos[1]-spawn_pos[1]):.2f}m")

        # 更新场景目标位置
        SCENE_CONFIG["goal_pos"] = goal_pos
        SCENE_CONFIG["robot_start"] = spawn_pos

        # 移动目标标记
        env.move_goal_marker(goal_pos[0], goal_pos[1])

        # 捕获目标点摄像头画面
        goal_view_img = capture_goal_view(env, goal_pos)
        print("[Random] Goal camera view captured")

        # 在线生成 topomap
        topomap = generate_topomap_online(
            env, spawn_pos, goal_pos, num_nodes=args.topomap_nodes)

        # 将机器人传送到出生点并重新落地
        env.set_robot_position(spawn_pos[0], spawn_pos[1])
    else:
        # 非随机模式: 不在此阶段捕获 goal view，等加载 topomap 后用最后一张
        pass

    # 设置初始前进指令让策略同时站立和启动
    env.set_command(0.2, 0.0, 0.0)
    env.standup(duration=args.standup_time)
    env.reset_wall_clock()

    if not args.random:
        if args.topomap_dir:
            topomap = load_topomap_from_dir(args.topomap_dir)
        else:
            topomap = load_topomap_from_dataset(args.topomap_traj, step=args.topomap_step)
        # 使用 topomap 最后一张图像作为目标点摄像头画面
        goal_view_img = topomap[-1]
    num_nodes = len(topomap)
    goal_node = num_nodes - 1
    closest_node = 0

    # FPV 头部视角保存
    fpv_dir = None
    if args.save_fpv:
        fpv_dir = os.path.join(PROJECT_ROOT, f"results/nomad_mujoco/{RUN_TAG}_navigate_mujoco/fpv")
        os.makedirs(fpv_dir, exist_ok=True)
        print(f"[Navigate] FPV frames will be saved to {fpv_dir}")

    # 闭环: 卡住检测器
    stuck_detector = StuckDetector()
    state_machine = "NAVIGATE"
    recovery_counter = 0
    recovery_count_total = 0
    turn_dir = 1.0

    frame_buffer = deque(maxlen=CONTEXT_SIZE + 1)
    trajectory = []
    velocity_log = []

    pos, yaw = env.get_pose()
    trajectory.append(pos.copy())
    print(
        f"\n[Navigate] Starting at ({pos[0]:.3f}, {pos[1]:.3f})"
    )
    print(f"[Navigate] Topomap: {num_nodes} nodes, goal_node={goal_node}")
    print(f"[Navigate] Closed-loop: stuck_threshold={STUCK_VEL_THRESHOLD} m/s, window={STUCK_CHECK_WINDOW}")
    print("=" * 60)

    reached_goal = False

    for step_i in range(args.max_steps):
        t0 = time.time()

        cam_img = env.render_camera()
        if fpv_dir and step_i % 5 == 0:
            cam_img.save(os.path.join(fpv_dir, f"{step_i:04d}.png"))

        # ── 实时 FPV + 目标点视角显示 ──
        actual_vel = env.get_body_speed_forward()
        status_text = f"{state_machine} | node={closest_node}/{goal_node} | v_body={actual_vel:.3f}"
        show_fpv_realtime(cam_img, step_i, status_text, goal_view=goal_view_img)

        # ── 状态机: 碰撞恢复 ──
        if state_machine == "RECOVERY_BACK":
            env.set_command(RECOVERY_BACK_VEL, 0.0, 0.0)
            env.step_nomad_period()
            recovery_counter += 1
            pos, yaw = env.get_pose()
            trajectory.append(pos.copy())
            velocity_log.append((RECOVERY_BACK_VEL, 0.0))
            if step_i % 2 == 0:
                print(f"  Step {step_i:4d} | RECOVERY_BACK {recovery_counter}/{RECOVERY_BACK_STEPS}")
            if recovery_counter >= RECOVERY_BACK_STEPS:
                state_machine = "RECOVERY_TURN"
                recovery_counter = 0
                turn_dir = 1.0 if np.random.random() > 0.5 else -1.0
                print(f"  [Recovery] Switching to TURN ({'left' if turn_dir > 0 else 'right'})")
            continue

        if state_machine == "RECOVERY_TURN":
            env.set_command(0.05, 0.0, turn_dir * RECOVERY_TURN_VEL)
            env.step_nomad_period()
            recovery_counter += 1
            pos, yaw = env.get_pose()
            trajectory.append(pos.copy())
            velocity_log.append((0.05, turn_dir * RECOVERY_TURN_VEL))
            if step_i % 2 == 0:
                print(f"  Step {step_i:4d} | RECOVERY_TURN {recovery_counter}/{RECOVERY_TURN_STEPS}")
            if recovery_counter >= RECOVERY_TURN_STEPS:
                state_machine = "NAVIGATE"
                recovery_counter = 0
                stuck_detector.reset()
                print(f"  [Recovery] Done! Resuming NAVIGATE")
            continue

        # ── 正常导航状态 ──
        frame_tensor = pil_to_tensor(cam_img)
        frame_buffer.append(frame_tensor)

        if len(frame_buffer) < 2:
            env.step_nomad_period()
            continue

        obs_tensor = build_obs_tensor(frame_buffer).to(device)
        mask = torch.zeros(1).long().to(device)

        # 搜索窗口内 topomap 节点
        radius = args.radius
        start = max(closest_node - radius, 0)
        end = min(closest_node + radius + 1, goal_node)

        goal_tensors = []
        for g_img in topomap[start : end + 1]:
            g_t = pil_to_tensor(g_img).unsqueeze(0).to(device)
            goal_tensors.append(g_t)
        goal_batch = torch.cat(goal_tensors, dim=0)
        K = goal_batch.shape[0]

        with torch.no_grad():
            obsgoal_cond = model(
                "vision_encoder",
                obs_img=obs_tensor.repeat(K, 1, 1, 1),
                goal_img=goal_batch,
                input_goal_mask=mask.repeat(K),
            )
            dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dists = dists.cpu().numpy().flatten()

        min_idx = np.argmin(dists)
        new_closest = min_idx + start
        # 限制 closest_node 跳变：最多前进 3 步，后退 1 步，防止定位错乱
        closest_node = int(np.clip(new_closest, closest_node - 1,
                                    closest_node + 3))
        closest_node = min(closest_node, goal_node)

        # 子目标推进（与原版 deploy/src/navigate.py 一致）:
        # 当最近节点的时间距离 < close_threshold 时，子目标前进一步
        # 注意: sg_idx 是相对于搜索窗口的局部索引
        local_idx = closest_node - start
        sg_idx = min(local_idx + int(dists[min_idx] < args.close_threshold),
                     len(obsgoal_cond) - 1)
        obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

        # CFG: 计算无条件编码（goal masked）
        uncond_cond = None
        if args.cfg_weight and args.cfg_weight != 0.0:
            mask_explore = torch.ones(1).long().to(device)
            with torch.no_grad():
                uncond_cond = model(
                    "vision_encoder",
                    obs_img=obs_tensor,
                    goal_img=goal_batch[sg_idx:sg_idx+1],
                    input_goal_mask=mask_explore,
                )

        sched = args.scheduler
        n_steps = args.ddim_steps if sched == "ddim" else NUM_DIFFUSION_ITERS
        actions = diffusion_inference(model, obs_cond, device,
                                      scheduler_kind=sched, num_steps=n_steps,
                                      guidance_scale=args.cfg_weight if args.cfg_weight else None,
                                      uncond=uncond_cond)
        mean_action = actions.mean(axis=0)
        wp_idx = min(args.waypoint, LEN_TRAJ_PRED - 1)
        chosen_wp = mean_action[wp_idx]

        v, w = pd_controller(chosen_wp)

        env.set_command(v, 0.0, w)
        env.step_nomad_period()

        # 闭环: 读取本体实际速度
        actual_vel = env.get_body_speed_forward()
        stuck_detector.update(actual_vel, v)

        pos, yaw = env.get_pose()
        trajectory.append(pos.copy())
        velocity_log.append((v, w))

        elapsed = time.time() - t0

        # 到达判定: 以物理距离为主要标准（仿真中有地面真值位置）
        # topomap 节点匹配仅作为辅助信息（仿真场景视觉特征与训练数据差异大，存在误匹配）
        goal_xy = np.array(SCENE_CONFIG["goal_pos"])
        phys_dist_to_goal = np.linalg.norm(pos - goal_xy)
        reached_goal = phys_dist_to_goal < GOAL_REACH_DIST

        if step_i % 5 == 0 or reached_goal:
            z = env.get_height()
            print(
                f"  Step {step_i:4d} | pos=({pos[0]:+.3f}, {pos[1]:+.3f}) "
                f"z={z:.3f} | node={closest_node}/{goal_node} "
                f"dist={dists[min_idx]:.3f} d_goal={phys_dist_to_goal:.2f}m | "
                f"v_cmd={v:.3f} v_body={actual_vel:.3f} | {elapsed:.3f}s"
            )

        if reached_goal:
            print(f"\n  *** GOAL REACHED (physical distance {phys_dist_to_goal:.2f}m < {GOAL_REACH_DIST}m)! ***")
            break

        # 闭环: 检测卡住 → 进入恢复
        if stuck_detector.is_stuck():
            recovery_count_total += 1
            print(f"  [!] STUCK detected at step {step_i}! "
                  f"v_body={actual_vel:.3f} << v_cmd={v:.3f} | "
                  f"Recovery #{recovery_count_total}")
            state_machine = "RECOVERY_BACK"
            recovery_counter = 0

        if env.is_fallen():
            print(f"  [!] Robot fallen at step {step_i}!")
            break

        if not env.viewer_alive():
            print("[Navigate] Viewer closed, stopping.")
            break

    print(f"\n[Navigate] Total recovery events: {recovery_count_total}")
    print(f"[Navigate] Result: {'SUCCESS' if reached_goal else 'FAILED'} in {step_i+1} steps")
    _save_results(trajectory, velocity_log, "navigate_mujoco", args, reached_goal)
    env.close()
    return reached_goal


# ══════════════════════════════════════════════════════════
#  结果保存
# ══════════════════════════════════════════════════════════
def _save_results(trajectory, velocity_log, mode, args, reached_goal=None):
    """保存轨迹和速度数据。"""
    out_dir = os.path.join(PROJECT_ROOT, f"results/nomad_mujoco/{RUN_TAG}_{mode}")
    os.makedirs(out_dir, exist_ok=True)

    trajectory = np.array(trajectory) if len(trajectory) > 0 else np.zeros((1, 2))
    np.savetxt(os.path.join(out_dir, "trajectory.txt"), trajectory, fmt="%.6f")

    if velocity_log:
        np.savetxt(
            os.path.join(out_dir, "velocities.txt"),
            np.array(velocity_log),
            fmt="%.6f",
        )

    total_dist = (
        np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        if len(trajectory) > 1
        else 0.0
    )

    summary = (
        f"Mode: {mode}\n"
        f"Steps: {len(trajectory)}\n"
        f"Total distance: {total_dist:.3f} m\n"
    )
    if reached_goal is not None:
        summary += f"Reached goal: {reached_goal}\n"

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary)

    print(f"\n[Results] Saved to {out_dir}")
    print(f"  Path distance: {total_dist:.3f} m")


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════
def main():
    global SCENE_CONFIG

    parser = argparse.ArgumentParser(
        description="NoMaD + Lite3 MuJoCo 集成仿真导航"
    )
    parser.add_argument(
        "--mode",
        choices=["explore", "navigate", "walk-test", "generate-topomap"],
        default="walk-test",
    )
    parser.add_argument("--map", choices=["easy", "medium", "hard"], default="easy",
                        help="地图: easy(直线5m)/medium(单障碍7m)/hard(S弯8m)")
    parser.add_argument("--no-gui", action="store_true", help="无头模式")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="最大 NoMaD 步数 (默认 200 ≈ 50s 仿真时间)")
    parser.add_argument("--waypoint", type=int, default=2,
                        help="选择扩散轨迹第几个路径点 (默认 2, 范围 0~7)")
    parser.add_argument(
        "--standup-time", type=float, default=3.0, help="站立预热时间 (s)"
    )
    parser.add_argument("--save-fpv", action="store_true",
                        help="保存头部 FPV 相机图像到 results 目录")

    # 导航/topomap 参数
    parser.add_argument("--topomap-traj", type=str, default="no10vc_10_0",
                        help="GoStanford 轨迹名（--topomap-dir 未指定时回退使用）")
    parser.add_argument("--topomap-dir", type=str, default=None,
                        help="自定义 topomap 目录（覆盖 --map 默认路径）")
    parser.add_argument("--topomap-step", type=int, default=5)
    parser.add_argument("--topomap-nodes", type=int, default=20,
                        help="generate-topomap 生成节点数 (默认 20)")
    parser.add_argument("--radius", type=int, default=4,
                        help="topomap 定位搜索半径 (默认 4)")
    parser.add_argument("--close-threshold", type=float, default=3.0,
                        help="时间距离阈值: <阈值则推进子目标 (默认 3.0)")

    # 扩散调度器参数
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddpm",
                        help="扩散推理调度器: ddpm(默认 10步) / ddim(可加速)")
    parser.add_argument("--ddim-steps", type=int, default=10,
                        help="DDIM 去噪步数 (默认 10, 可设 5/3/2/1 加速)")
    parser.add_argument("--cfg-weight", type=float, default=0.0,
                        help="CFG 引导强度 (0.0=不使用, 1.0=标准引导, 2.0=强引导)")

    # 随机模式
    parser.add_argument("--random", action="store_true",
                        help="随机生成出生点和目标点（自动生成在线 topomap）")

    args = parser.parse_args()

    # ── 设置活动场景配置 ──
    SCENE_CONFIG = SCENE_MAPS[args.map]
    print(f"\n{'='*60}")
    print(f"[Config] Map: {args.map} - {SCENE_CONFIG['name']}")
    print(f"[Config] Goal: {SCENE_CONFIG['goal_pos']}, "
          f"Obstacles: {len(SCENE_CONFIG.get('obstacles', []))}")
    print(f"{'='*60}")

    if args.mode == "walk-test":
        run_walk_test(args)
    elif args.mode == "explore":
        run_explore(args)
    elif args.mode == "navigate":
        # 随机模式: 在线生成 topomap，无需预生成
        if args.random:
            print(f"[Config] Random mode: spawn/goal will be randomized")
        else:
            # 自动检测 topomap 目录
            if not args.topomap_dir:
                default_dir = get_default_topomap_dir(args.map)
                if os.path.isdir(default_dir) and any(
                    f.endswith(".png") for f in os.listdir(default_dir)
                ):
                    args.topomap_dir = default_dir
                    print(f"[Config] Topomap: {default_dir}")
                else:
                    print(f"[Error] Topomap not found at: {default_dir}")
                    print(f"  请先运行: python {sys.argv[0]} --mode generate-topomap --map {args.map}")
                    sys.exit(1)
        success = run_navigate(args)
        sys.exit(0 if success else 1)
    elif args.mode == "generate-topomap":
        run_generate_topomap(args)


if __name__ == "__main__":
    main()
