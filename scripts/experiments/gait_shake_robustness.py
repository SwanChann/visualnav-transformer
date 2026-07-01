#!/usr/bin/env python3
# Allow direct execution from any scripts/<category>/ path.
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve()
while SCRIPTS_ROOT.name != "scripts" and SCRIPTS_ROOT.parent != SCRIPTS_ROOT:
    SCRIPTS_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
"""
Day 3: 步态抖动鲁棒性实验

研究问题：四足机器人运动时，相机会随身体运动产生周期性抖动。
这种抖动在轮式平台上不存在，是足式部署的独特挑战。

实验方法：
  1. 对输入图像施加模拟步态周期性变换（旋转±α、平移±d）
  2. 对比干净图像与抖动图像的路径点预测差异
  3. 测试 CFG 能否缓解抖动干扰（结合 Day 3 上午结果）
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import transforms as T
from PIL import Image as PILImage
from tooling.project_paths import REPO_ROOT, add_repo_paths

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

PROJECT_ROOT = str(REPO_ROOT)
add_repo_paths()

from offline_inference import (
    build_model, prepare_goal,
    diffusion_inference, _find_image,
    DATASET_ROOT, CONTEXT_SIZE, LEN_TRAJ_PRED, IMAGE_SIZE,
)

# ── 步态抖动模拟 ─────────────────────────────────────────
# 云深处 Lite3 典型步态参数：
#   俯仰振荡: ±2–3°（步态频率 ≈2.5 Hz 走 / 3.5 Hz 小跑）
#   垂直振荡: ±5–10 mm
#   横滚振荡: ±1–2°
GAIT_CONFIGS = {
    "no_shake": {"angle": 0.0, "shift_x": 0, "shift_y": 0},
    "mild_shake (walk)": {"angle": 1.5, "shift_x": 2, "shift_y": 3},
    "moderate_shake (fast_walk)": {"angle": 3.0, "shift_x": 4, "shift_y": 5},
    "severe_shake (trot)": {"angle": 5.0, "shift_x": 6, "shift_y": 8},
}


def apply_gait_shake(
    img_pil: PILImage.Image,
    angle_deg: float,
    shift_x: int, shift_y: int,
    phase: float = 0.0,
) -> PILImage.Image:
    """
    模拟四足步态引起的相机周期性抖动。
    在 PIL 空间操作，避免归一化后的张量被 ToPILImage 截断破坏。

    参数:
        img_pil: 原始 PIL 图像（未归一化）
        angle_deg: 最大旋转角度（俯仰）
        shift_x, shift_y: 最大平移像素
        phase: 步态相位 (0–2π)
    返回:
        抖动后的 PIL 图像（未归一化）
    """
    if angle_deg == 0 and shift_x == 0 and shift_y == 0:
        return img_pil                          # 无抖动时直接返回，零开销

    actual_angle = angle_deg * np.sin(phase)
    actual_sx = int(shift_x * np.sin(phase))
    actual_sy = int(shift_y * np.cos(phase * 0.5))

    img_pil = img_pil.rotate(actual_angle, fillcolor=(128, 128, 128))

    arr = np.array(img_pil)
    arr = np.roll(arr, actual_sx, axis=1)
    arr = np.roll(arr, actual_sy, axis=0)
    return PILImage.fromarray(arr)


def apply_shake_to_observation(
    traj_path: str, t: int,
    angle_deg: float, shift_x: int, shift_y: int,
) -> torch.Tensor:
    """
    为 context_size+1 观测帧施加不同相位的步态抖动。

    修正：抖动在 PIL 空间施加（归一化之前），ImageNet 归一化仅在最后
    执行一次。这保证 no_shake 条件与 prepare_observation() 完全等价，
    天然成为基线。
    """
    from offline_inference import _TRANSFORM

    frames = []
    indices = list(range(max(0, t - CONTEXT_SIZE), t + 1))
    while len(indices) < CONTEXT_SIZE + 1:
        indices.insert(0, indices[0])
    indices = indices[-(CONTEXT_SIZE + 1):]

    for i, idx in enumerate(indices):
        # 1) 加载原始 PIL 图像（未归一化）
        img_pil = PILImage.open(_find_image(traj_path, idx)).convert("RGB")
        img_pil = img_pil.resize(IMAGE_SIZE)

        # 2) 在 PIL 空间施加步态抖动
        phase = 2 * np.pi * i / (CONTEXT_SIZE + 1)  # 模拟步态周期
        img_pil = apply_gait_shake(img_pil, angle_deg, shift_x, shift_y, phase)

        # 3) 统一归一化（ToTensor + ImageNet Normalize，仅此一次）
        img = _TRANSFORM(img_pil)
        frames.append(img)

    return torch.cat(frames, dim=0).unsqueeze(0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两条轨迹之间的余弦相似度"""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return float(np.dot(a_flat, b_flat) / (
        np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8
    ))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(device)

    trajs     = sorted(os.listdir(DATASET_ROOT))
    traj_path = os.path.join(DATASET_ROOT, trajs[0])
    obs_time  = CONTEXT_SIZE + 2
    goal_time = obs_time + 15

    goal_img = prepare_goal(traj_path, goal_time).to(device)
    mask = torch.zeros(1).long().to(device)

    # ── 所有条件（含 no_shake）走同一条处理流水线 ────
    # no_shake (angle=0, shift=0) 的处理路径与 prepare_observation() 完全等价，
    # 天然成为基线，无需单独计算 baseline。
    raw_results = {}
    print("=" * 70)
    print("Gait Shake Robustness Experiment")
    print("=" * 70)

    for name, cfg in GAIT_CONFIGS.items():
        print(f"\n--- {name} (angle=±{cfg['angle']}°, "
              f"shift=±{cfg['shift_x']}/{cfg['shift_y']}px) ---")

        obs = apply_shake_to_observation(
            traj_path, obs_time,
            cfg["angle"], cfg["shift_x"], cfg["shift_y"],
        ).to(device)

        with torch.no_grad():
            cond = model("vision_encoder", obs_img=obs,
                         goal_img=goal_img, input_goal_mask=mask)
        trajs_out = diffusion_inference(model, cond, device)
        raw_results[name] = trajs_out.mean(dim=0).cpu().numpy()

    # ── no_shake 即为基线 ──────────────────────────
    baseline = raw_results["no_shake"]

    # ── 计算各条件相对基线的指标 ──────────────────
    results = {}
    for name in GAIT_CONFIGS:
        mean_traj = raw_results[name]
        cos_sim = cosine_similarity(baseline, mean_traj)
        mse     = np.mean((baseline - mean_traj) ** 2)
        heading_err = np.abs(np.arctan2(mean_traj[0, 1], mean_traj[0, 0])
                            - np.arctan2(baseline[0, 1], baseline[0, 0]))

        results[name] = dict(cos_sim=cos_sim, mse=mse,
                             heading_err=np.degrees(heading_err),
                             mean_traj=mean_traj)
        print(f"  {name}: cos_sim={cos_sim:.4f} | MSE={mse:.6f} | "
              f"Heading Dev={np.degrees(heading_err):.2f}°")

    # ── 可视化 ───────────────────────────────────
    out_dir = os.path.join(PROJECT_ROOT, f"results/day3/{RUN_TAG}_gait_shake")
    os.makedirs(out_dir, exist_ok=True)
    names = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图表 1: 轨迹对比
    for name, r in results.items():
        t = r["mean_traj"]
        axes[0].plot(t[:, 1], t[:, 0], "o-", label=name, markersize=3)
    axes[0].plot(0, 0, "r*", markersize=15, label="robot")
    axes[0].set_title("Mean Trajectories Under Different Shake Intensities")
    axes[0].set_xlabel("y"); axes[0].set_ylabel("x")
    axes[0].legend(fontsize=8); axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    # 图表 2: 鲁棒性指标（仅展示有抖动的条件，no_shake 恒为 1.0）
    shake_names = [n for n in names if n != "no_shake"]
    cos_vals = [results[n]["cos_sim"] for n in shake_names]
    colors = ["#3498db", "#e67e22", "#e74c3c"][:len(shake_names)]
    axes[1].bar(range(len(shake_names)), cos_vals, color=colors)
    axes[1].set_xticks(range(len(shake_names)))
    axes[1].set_xticklabels(shake_names, rotation=20, ha="right", fontsize=9)
    axes[1].axhline(y=0.95, color="green", linestyle="--", alpha=0.7, label="robust threshold")
    axes[1].set_ylabel("Cosine Similarity vs no_shake")
    axes[1].set_title("Visual Navigation Robustness to Gait Shake")
    axes[1].set_ylim(0.8, 1.01)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gait_shake_robustness.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── 单独保存每个抖动级别的图表 ───────────────
    for name, r in results.items():
        fig_g, ax_g = plt.subplots(1, 1, figsize=(6, 6))
        t = r["mean_traj"]
        ax_g.plot(t[:, 1], t[:, 0], "o-", color="#e67e22", markersize=4, label=name)
        ax_g.plot(baseline[:, 1], baseline[:, 0], "--", color="#2ecc71",
                  linewidth=1.5, label="no_shake (baseline)")
        ax_g.plot(0, 0, "r*", markersize=15)
        ax_g.set_title(f"{name} | cos_sim={r['cos_sim']:.4f}")
        ax_g.set_aspect("equal"); ax_g.grid(True, alpha=0.3)
        ax_g.legend(fontsize=8); plt.tight_layout()
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(out_dir, f"shake_{safe}.png"), dpi=150)
        plt.close()

    # ── 保存结果到 txt ─────────────────────────
    txt_path = os.path.join(out_dir, "results.txt")
    with open(txt_path, "w") as f:
        f.write(f"Run Time: {RUN_TAG}\nScript: gait_shake_robustness.py\n\n")
        f.write(f"{'Shake Level':<20} {'cos_sim':<10} {'MSE':<12} {'Heading Dev.':<10}\n")
        f.write("-" * 52 + "\n")
        for n in names:
            r = results[n]
            f.write(f"{n:<20} {r['cos_sim']:<10.4f} {r['mse']:<12.6f} "
                    f"{r['heading_err']:<10.2f}\n")
        f.write("\nRobustness Assessment:\n")
        for n in names:
            r = results[n]
            if r["cos_sim"] > 0.95:
                f.write(f"  {n}: ✅ Navigation policy is robust to this shake level\n")
            else:
                f.write(f"  {n}: ❌ Needs additional compensation (CFG or image stabilization)\n")
    print(f"\nResults saved: {txt_path}")

    # ── 总结 ───────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Shake Level':<20} {'cos_sim':<10} {'MSE':<12} {'Heading Dev.':<10}")
    print("-" * 70)
    for n in names:
        r = results[n]
        print(f"{n:<20} {r['cos_sim']:<10.4f} {r['mse']:<12.6f} "
              f"{r['heading_err']:<10.2f}°")
    print("=" * 70)
    print("\n预期观察:")
    print("  cos_sim > 0.95 -> 导航策略对该抖动级别具有鲁棒性")
    print("  cos_sim < 0.90 -> 足式部署需要额外补偿（如 CFG 增强）")


if __name__ == "__main__":
    main()
