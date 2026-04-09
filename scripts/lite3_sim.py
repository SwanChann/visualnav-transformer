#!/usr/bin/env python3
"""
Day 5: DeepRobotics Lite3 Quadruped Simulation Adapter (Ubuntu 20.04)

将 NoMaD 路径点转换为速度指令，通过 PD 控制器在 PyBullet 中执行。
同时支持轮式（Husky）和足式（Lite3）平台的对比实验。

接口链:
  NoMaD 路径点 (dx, dy)
    -> waypoint_to_velocity -> (v, ω)
    -> PyBullet -> 机器人运动

平台: DeepRobotics Lite3
  - 体重: ~12.5 kg
  - 站立高度: ~0.40 m
  - 相机安装高度: ~0.32 m
  - 自由度: 12 (每腿 3 × 4 条腿)
  - 最大线速度: ~1.5 m/s
  - 步态频率: ~2.5 Hz (行走), ~3.5 Hz (小跑)
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pybullet as p
import pybullet_data
from project_paths import REPO_ROOT, repo_path

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_ROOT = str(REPO_ROOT)


class Lite3Simulator:
    """DeepRobotics Lite3 / Husky（回退）仿真控制器。

    若 Lite3 URDF 不存在，自动回退到 PyBullet 内置的 Husky 轮式底盘。
    """

    def __init__(self, urdf_path: str = None, gui: bool = True):
        mode = p.GUI if gui else p.DIRECT
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.plane = p.loadURDF("plane.urdf")

        if urdf_path and os.path.exists(urdf_path):
            self.robot = p.loadURDF(urdf_path, [0, 0, 0.4],
                                    useFixedBase=False)
            print(f"[Lite3Sim] URDF loaded: {urdf_path}")
        else:
            # 回退到 PyBullet 内置的 Husky
            self.robot = p.loadURDF("husky/husky.urdf", [0, 0, 0.1],
                                    useFixedBase=False)
            print("[Lite3Sim] WARNING: Lite3 URDF not found, using Husky as fallback")

        self.dt = 1.0 / 240.0

        # 打印关节信息（调试用）
        n_joints = p.getNumJoints(self.robot)
        print(f"[Lite3Sim] Robot joints: {n_joints}")
        for j in range(min(n_joints, 12)):
            info = p.getJointInfo(self.robot, j)
            print(f"  Joint {j}: {info[1].decode()} type={info[2]}")

    def get_pose(self):
        """返回 (x, y) 坐标和 yaw 偏航角"""
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return np.array(pos[:2]), yaw

    def set_velocity(self, v: float, w: float):
        """设置线速度 v (m/s) 和角速度 w (rad/s)"""
        p.resetBaseVelocity(
            self.robot,
            linearVelocity=[v, 0, 0],
            angularVelocity=[0, 0, w],
        )

    def step(self, n: int = 1):
        for _ in range(n):
            p.stepSimulation()

    def close(self):
        p.disconnect()


def waypoint_to_velocity(
    waypoint: np.ndarray,
    dt: float = 0.25,    # NoMaD 帧率=4 → dt=0.25s
    max_v: float = 0.5,
    max_w: float = 1.0,
) -> tuple:
    """
    将 NoMaD 路径点 (dx, dy) 转换为速度指令 (v, ω)。
    逻辑与 deployment/src/pd_controller.py 对齐。
    """
    dx, dy = float(waypoint[0]), float(waypoint[1])
    EPS = 1e-8

    if abs(dx) < EPS and abs(dy) < EPS:
        return 0.0, 0.0
    elif abs(dx) < EPS:
        v = 0.0
        w = np.sign(dy) * np.pi / (2 * dt)
    else:
        v = dx / dt
        w = np.arctan(dy / dx) / dt

    v = float(np.clip(v, 0, max_v))
    w = float(np.clip(w, -max_w, max_w))
    return v, w


def demo_open_loop():
    """开环演示: 使用预设路径点序列驱动机器人。"""
    sim = Lite3Simulator(gui=True)

    waypoints = np.array([
        [0.05,  0.01], [0.10,  0.02], [0.15,  0.01], [0.20, -0.01],
        [0.25, -0.02], [0.30,  0.00], [0.35,  0.01], [0.40,  0.00],
    ])

    positions = []
    pos, yaw = sim.get_pose()
    positions.append(pos.copy())

    print("Starting simulation (open-loop)...")
    for i, wp in enumerate(waypoints):
        v, w = waypoint_to_velocity(wp)
        print(f"  WP {i}: ({wp[0]:.3f}, {wp[1]:.3f}) -> v={v:.3f}, w={w:.3f}")

        for _ in range(100):       # 100 steps ≈ 0.42 s
            sim.set_velocity(v, w)
            sim.step()
            time.sleep(sim.dt)

        pos, yaw = sim.get_pose()
        positions.append(pos.copy())

    positions = np.array(positions)
    total_disp = np.linalg.norm(positions[-1] - positions[0])

    print(f"\nStart: ({positions[0][0]:.3f}, {positions[0][1]:.3f})")
    print(f"End:   ({positions[-1][0]:.3f}, {positions[-1][1]:.3f})")
    print(f"Total Displacement: {total_disp:.3f} m")

    # ── 保存结果到 txt ──
    out_dir = os.path.join(PROJECT_ROOT, "results/day5")
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"{RUN_TAG}_lite3_open_loop.txt")
    with open(txt_path, "w") as f:
        f.write(f"Run Time: {RUN_TAG}\n")
        f.write(f"Script: lite3_sim.py (demo_open_loop)\n")
        f.write(f"Platform: DeepRobotics Lite3 (or Husky fallback)\n\n")
        f.write("# 路径点执行日志\n")
        f.write(f"# 列说明: step, wp_dx, wp_dy, cmd_v, cmd_w, pos_x, pos_y\n")
        f.write(f"# wp_dx/wp_dy: NoMaD 路径点位移 (m)\n")
        f.write(f"# cmd_v: 线速度指令 (m/s)\n")
        f.write(f"# cmd_w: 角速度指令 (rad/s)\n")
        f.write(f"# pos_x/pos_y: 执行后机器人世界坐标 (m)\n\n")
        f.write(f"{'step':<6} {'wp_dx':<8} {'wp_dy':<8} {'cmd_v':<8} {'cmd_w':<8} {'pos_x':<10} {'pos_y':<10}\n")
        f.write("-" * 58 + "\n")
        for i, wp in enumerate(waypoints):
            v, w_ = waypoint_to_velocity(wp)
            f.write(f"{i:<6d} {wp[0]:<8.3f} {wp[1]:<8.3f} {v:<8.3f} {w_:<8.3f} "
                    f"{positions[i+1][0]:<10.4f} {positions[i+1][1]:<10.4f}\n")
        f.write(f"\nTotal Displacement: {total_disp:.4f} m\n")
        f.write(f"Start: ({positions[0][0]:.4f}, {positions[0][1]:.4f})\n")
        f.write(f"End:   ({positions[-1][0]:.4f}, {positions[-1][1]:.4f})\n")
    print(f"Results saved: {txt_path}")

    input("Press Enter to close simulation...")
    sim.close()


def compare_wheeled_vs_legged():
    """
    对比轮式 (Husky) 与足式 (Lite3) 平台的路径跟踪性能。
    使用相同的路径点序列，对比两种平台的路径跟踪精度和运动特性。
    """
    lite3_urdf = str(repo_path("assets", "lite3", "lite3.urdf"))

    waypoints = np.array([
        [0.10, 0.00], [0.10, 0.05], [0.10, 0.00],
        [0.10, -0.05], [0.10, 0.00], [0.15, 0.02],
    ])

    all_results = {}
    for platform, urdf in [("Wheeled (Husky)", None),
                           ("Legged (Lite3)", lite3_urdf)]:
        sim = Lite3Simulator(urdf_path=urdf, gui=False)
        positions = []
        pos, _ = sim.get_pose()
        positions.append(pos.copy())

        for wp in waypoints:
            v, w = waypoint_to_velocity(wp)
            for _ in range(100):
                sim.set_velocity(v, w)
                sim.step()
            pos, _ = sim.get_pose()
            positions.append(pos.copy())

        sim.close()
        positions = np.array(positions)

        # 计算跟踪指标
        path_length = sum(np.linalg.norm(positions[i+1] - positions[i])
                          for i in range(len(positions) - 1))
        final_disp = np.linalg.norm(positions[-1] - positions[0])
        # 轨迹偏差：每段的侧向偏差
        lateral_devs = []
        for i in range(1, len(positions)):
            desired_dir = waypoints[min(i-1, len(waypoints)-1)]
            actual_dir = positions[i] - positions[i-1]
            if np.linalg.norm(desired_dir) > 1e-8:
                lateral = np.abs(np.cross(desired_dir, actual_dir)) / np.linalg.norm(desired_dir)
                lateral_devs.append(lateral)
        mean_lateral = np.mean(lateral_devs) if lateral_devs else 0.0

        all_results[platform] = dict(
            positions=positions, path_length=path_length,
            final_disp=final_disp, mean_lateral_dev=mean_lateral,
        )
        print(f"  {platform}: path={path_length:.4f}m, "
              f"disp={final_disp:.4f}m, lat_dev={mean_lateral:.4f}m")

    # ── 保存对比结果到 txt ──
    out_dir = os.path.join(PROJECT_ROOT, "results/day5")
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"{RUN_TAG}_wheeled_vs_legged.txt")
    with open(txt_path, "w") as f:
        f.write(f"Run Time: {RUN_TAG}\n")
        f.write(f"Script: lite3_sim.py (compare_wheeled_vs_legged)\n\n")
        f.write("# 轮式 vs 足式平台对比\n")
        f.write("# 指标说明:\n")
        f.write("#   path_length   - 沿轨迹的总行驶距离 (m)\n")
        f.write("#   final_disp    - 从起点到终点的欧氏位移 (m)\n")
        f.write("#   mean_lateral  - 相对期望方向的平均侧向偏差 (m)\n")
        f.write("#     侧向偏差越低 = 路径跟踪精度越高\n\n")
        f.write(f"{'Platform':<22} {'Path Length':<14} {'Final Disp.':<14} {'Lat. Dev.':<12}\n")
        f.write("-" * 62 + "\n")
        for name, r in all_results.items():
            f.write(f"{name:<22} {r['path_length']:<14.4f} "
                    f"{r['final_disp']:<14.4f} {r['mean_lateral_dev']:<12.4f}\n")
        f.write(f"\n# 位置轨迹 (step, x, y):\n")
        for name, r in all_results.items():
            f.write(f"\n## {name}\n")
            for i, pos in enumerate(r['positions']):
                f.write(f"  {i}: ({pos[0]:.4f}, {pos[1]:.4f})\n")
    print(f"\nComparison results saved: {txt_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lite3 Simulation")
    parser.add_argument("--compare", action="store_true",
                        help="运行轮式 vs 足式对比实验")
    args = parser.parse_args()

    if args.compare:
        compare_wheeled_vs_legged()
    else:
        demo_open_loop()
