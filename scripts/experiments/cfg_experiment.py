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
Day 3: 无分类器引导 (CFG) 实验
探究不同引导强度 (w) 对轨迹目标导向性和多样性的影响。

CFG 公式:  ε_final = (1 + w) · ε_cond − w · ε_uncond
  - w = −1  → 纯无条件生成（完全忽略目标 — 探索模式）
  - w = 0   → 标准条件生成（使用目标导航，无额外增强）
  - w > 0   → 引导增强（放大目标方向信号）
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tooling.project_paths import REPO_ROOT, add_repo_paths

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

PROJECT_ROOT = str(REPO_ROOT)
add_repo_paths()

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from offline_inference import (
    build_model, prepare_observation, prepare_goal,
    DATASET_ROOT, CONTEXT_SIZE, IMAGE_SIZE, LEN_TRAJ_PRED,
)


def cfg_inference(
    model, obs_img, goal_img, device,
    w: float = 0.0,
    num_samples: int = 8,
    num_steps: int = 10,
):
    """
    带无分类器引导的扩散推理。

    每个去噪步骤需要 **两次** 前向传播：
      1. mask=0 → ε_cond    （导航方向）
      2. mask=1 → ε_uncond  （探索方向）
    然后: ε_final = (1+w)·ε_cond − w·ε_uncond
    """
    scheduler = DDPMScheduler(
        num_train_timesteps=num_steps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # ── 视觉编码（仅各执行一次，不在去噪循环内重复）──
    mask_nav     = torch.zeros(1).long().to(device)
    mask_explore = torch.ones(1).long().to(device)

    with torch.no_grad():
        cond_enc   = model("vision_encoder", obs_img=obs_img,
                           goal_img=goal_img, input_goal_mask=mask_nav)
        uncond_enc = model("vision_encoder", obs_img=obs_img,
                           goal_img=goal_img, input_goal_mask=mask_explore)

    cond_rep   = cond_enc.repeat(num_samples, 1)
    uncond_rep = uncond_enc.repeat(num_samples, 1)

    naction = torch.randn((num_samples, LEN_TRAJ_PRED, 2), device=device)
    scheduler.set_timesteps(num_steps)

    with torch.no_grad():
        for k in scheduler.timesteps:
            eps_cond = model("noise_pred_net", sample=naction,
                             timestep=k, global_cond=cond_rep)

            if w != 0.0:
                eps_uncond = model("noise_pred_net", sample=naction,
                                   timestep=k, global_cond=uncond_rep)
                # CFG 核心公式: ε_final = (1+w)·ε_cond − w·ε_uncond
                # w=-1 → 纯探索(ε_uncond); w>0 → 增强引导
                eps_final = (1.0 + w) * eps_cond - w * eps_uncond
            else:
                # w=0: 标准条件生成，跳过无条件前向传播以节省计算
                eps_final = eps_cond

            naction = scheduler.step(
                model_output=eps_final, timestep=k, sample=naction,
            ).prev_sample

    return naction


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(device)

    # 准备测试数据
    trajs     = sorted(os.listdir(DATASET_ROOT))
    traj_path = os.path.join(DATASET_ROOT, trajs[0])
    obs_time  = CONTEXT_SIZE + 2
    goal_time = obs_time + 15

    obs_img  = prepare_observation(traj_path, obs_time).to(device)
    goal_img = prepare_goal(traj_path, goal_time).to(device)

    # ── 实验 ──────────────────────────────────────────────
    w_values = [-1.0, 0.0, 0.5, 1.0, 2.0, 4.0]
    results  = {}

    print("=" * 70)
    print("Classifier-Free Guidance Experiment")
    print("=" * 70)

    for w in w_values:
        print(f"\n--- w = {w} ---")
        t0 = time.time()
        out = cfg_inference(model, obs_img, goal_img, device, w=w)
        elapsed = time.time() - t0

        trajs_np  = out.cpu().numpy()
        mean_traj = trajs_np.mean(axis=0)
        diversity = trajs_np.std(axis=0).mean()
        mean_dist = np.sqrt((mean_traj ** 2).sum(axis=1)).mean()

        results[w] = dict(
            trajs=trajs_np, mean_traj=mean_traj,
            diversity=diversity, mean_dist=mean_dist, time=elapsed,
        )
        print(f"  Time: {elapsed*1000:.1f} ms | "
              f"Diversity: {diversity:.4f} | Mean Displacement: {mean_dist:.4f}")

    # ── 可视化 1: 各 w 值下的轨迹分布 ────────
    out_dir = os.path.join(PROJECT_ROOT, "results/day3")
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 12))
    gs  = GridSpec(2, 3, figure=fig)

    for idx, w in enumerate(w_values):
        ax   = fig.add_subplot(gs[idx // 3, idx % 3])
        data = results[w]
        for i in range(data["trajs"].shape[0]):
            t = data["trajs"][i]
            ax.plot(t[:, 1], t[:, 0], "o-", alpha=0.4, markersize=2)
        mt = data["mean_traj"]
        ax.plot(mt[:, 1], mt[:, 0], "k-", linewidth=2.5, label="mean")
        ax.plot(0, 0, "r*", markersize=12)
        ax.set_title(f"w = {w} | div = {data['diversity']:.3f}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle("Classifier-Free Guidance: Trajectory Comparison for Different w", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cfg_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── 可视化 2: 趋势图 ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ws = list(results.keys())

    axes[0].plot(ws, [results[w]["diversity"]  for w in ws], "bo-")
    axes[0].set_xlabel("w")
    axes[0].set_ylabel("Diversity (std)")
    axes[0].set_title("Diversity vs w")

    axes[1].plot(ws, [results[w]["mean_dist"]  for w in ws], "ro-")
    axes[1].set_xlabel("w")
    axes[1].set_ylabel("Mean Displacement")
    axes[1].set_title("Goal-Directedness vs w")

    axes[2].plot(ws, [results[w]["time"] * 1000 for w in ws], "go-")
    axes[2].set_xlabel("w")
    axes[2].set_ylabel("Inference Time (ms)")
    axes[2].set_title("Inference Time vs w")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cfg_trends.png"), dpi=150)
    plt.close()

    # ── 可视化 3: 单独保存每个 w 值的图表 ─────────
    for w_val in w_values:
        fig_w, ax_w = plt.subplots(1, 1, figsize=(6, 6))
        data = results[w_val]
        for i in range(data["trajs"].shape[0]):
            t = data["trajs"][i]
            ax_w.plot(t[:, 1], t[:, 0], "o-", alpha=0.4, markersize=2)
        mt = data["mean_traj"]
        ax_w.plot(mt[:, 1], mt[:, 0], "k-", linewidth=2.5, label="mean")
        ax_w.plot(0, 0, "r*", markersize=12)
        ax_w.set_title(f"CFG w={w_val} | div={data['diversity']:.3f}")
        ax_w.set_aspect("equal"); ax_w.grid(True, alpha=0.3)
        ax_w.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cfg_w{w_val:.1f}.png"), dpi=150)
        plt.close()

    # ── 保存结果到 txt ─────────────────────────
    txt_path = os.path.join(out_dir, "results.txt")
    with open(txt_path, "w") as f:
        f.write(f"Run Time: {RUN_TAG}\nScript: cfg_experiment.py\n\n")
        f.write(f"{'w':<8} {'Diversity':<12} {'Mean Disp.':<12} {'Time(ms)':<12}\n")
        f.write("-" * 44 + "\n")
        for w_val in ws:
            r = results[w_val]
            f.write(f"{w_val:<8.1f} {r['diversity']:<12.4f} "
                    f"{r['mean_dist']:<12.4f} {r['time']*1000:<12.1f}\n")
    print(f"\nResults saved: {txt_path}")

    # ── 总结 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'w':<8} {'Diversity':<12} {'Mean Disp.':<12} {'Time(ms)':<12}")
    print("-" * 70)
    for w in ws:
        r = results[w]
        print(f"{w:<8.1f} {r['diversity']:<12.4f} "
              f"{r['mean_dist']:<12.4f} {r['time']*1000:<12.1f}")
    print("=" * 70)
    print("\n预期观察:")
    print("  w=-1    : 纯探索模式，最高多样性，无目标导向")
    print("  w=0     : 标准条件生成（目标导航基线）")
    print("  w=1~2   : 引导增强，多样性降低，轨迹更集中于目标方向（最佳范围）")
    print("  w=4     : 过度引导，可能过于激进，轨迹质量下降")


if __name__ == "__main__":
    main()
