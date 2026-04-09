#!/usr/bin/env python3
"""
Day 6: 综合消融实验（面向足式部署）

系统性消融实验，排列组合所有模块改进维度，量化每项改进对足式部署的贡献。

实验矩阵 (3 × 3 × 4 = 36 个配置/编码器):
  采样器:    DDPM-10 (基线), DDIM-5 (平衡), DDIM-3 (最快)
  引导强度:  w=0 (无 CFG), w=1.0 (中等), w=2.0 (激进)
  步态抖动: 无, 轻微 (行走), 中等 (快走), 严重 (小跑)

编码器维度（若 DINOv2 权重可用）:
  EfficientNet-B0 (原始) vs DINOv2-Small (Day 4)
  → 最多 72 个总配置

每个配置的指标:
  - inference_ms: 推理延迟 (ms)，越低越适合实时足式控制
  - diversity:    轨迹多样性 (std)，反映策略的探索能力
  - mse_vs_clean: 相比无抖动基线的均方误差，衡量抖动鲁棒性
  - cos_sim:      相比无抖动基线的余弦相似度，衡量方向一致性
  - heading_dev:  航向偏差 (degrees)，衡量第一个路径点的方向准确度
  - mean_disp:    平均位移量，衡量目标导向性

Output files:
  results/day6/ablation_full_results.txt    — 完整数值结果表
  results/day6/ablation_summary.txt         — 关键发现摘要
  results/day6/ablation_heatmap.png         — 采样器×引导强度 热力图
  results/day6/ablation_shake_robustness.png — 抖动鲁棒性对比图
  results/day6/ablation_pareto.png          — 速度-质量 Pareto 前沿
  results/day6/ablation_radar.png           — 最佳配置雷达图

Usage:
  python scripts/ablation_experiment.py [--dinov2-weights PATH] [--num-runs 5]
"""

import os
import sys
import time
import argparse
import itertools
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from matplotlib.gridspec import GridSpec
from project_paths import REPO_ROOT, add_repo_paths

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_ROOT = str(REPO_ROOT)
add_repo_paths()

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from offline_inference import (
    build_model, prepare_observation, prepare_goal,
    _find_image, _TRANSFORM,
    DATASET_ROOT, CONTEXT_SIZE, IMAGE_SIZE, LEN_TRAJ_PRED,
)

# ─────────────────────────────────────────────────────────────
# 实验维度定义
# ─────────────────────────────────────────────────────────────
SAMPLER_CONFIGS = [
    ("DDPM-10", "ddpm", 10),
    ("DDIM-5",  "ddim", 5),
    ("DDIM-3",  "ddim", 3),
]

GUIDANCE_CONFIGS = [
    ("w=0",   0.0),
    ("w=1.0", 1.0),
    ("w=2.0", 2.0),
]

# DeepRobotics Lite3 步态抖动参数
SHAKE_CONFIGS = [
    ("no_shake",       0.0, 0, 0),
    ("mild (walk)",    1.5, 2, 3),
    ("moderate (fast)", 3.0, 4, 5),
    ("severe (trot)",  5.0, 6, 8),
]

NUM_SAMPLES = 8     # 每次推理生成的轨迹数量
NUM_RUNS    = 5     # 每个配置重复次数（用于统计均值/标准差）


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────
def make_scheduler(kind: str, num_train_timesteps: int = 10):
    """创建 DDPM / DDIM 调度器"""
    common = dict(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    if kind == "ddpm":
        return DDPMScheduler(**common)
    return DDIMScheduler(**common)


def apply_gait_shake(img_pil, angle_deg, shift_x, shift_y, phase=0.0):
    """
    模拟四足步态引起的相机周期性抖动。
    在 PIL 空间操作，避免归一化后的张量被 ToPILImage 截断破坏。
    """
    from PIL import Image as PILImage

    if angle_deg == 0 and shift_x == 0 and shift_y == 0:
        return img_pil

    actual_angle = angle_deg * np.sin(phase)
    actual_sx = int(shift_x * np.sin(phase))
    actual_sy = int(shift_y * np.cos(phase * 0.5))

    img_pil = img_pil.rotate(actual_angle, fillcolor=(128, 128, 128))
    arr = np.array(img_pil)
    arr = np.roll(arr, actual_sx, axis=1)
    arr = np.roll(arr, actual_sy, axis=0)
    return PILImage.fromarray(arr)


def prepare_shaken_observation(traj_path, t, angle, sx, sy):
    """
    为 context_size+1 帧应用不同相位的步态抖动。
    抖动在 PIL 空间施加（归一化之前），ImageNet 归一化仅在最后执行一次。
    """
    from PIL import Image as PILImage

    frames = []
    indices = list(range(max(0, t - CONTEXT_SIZE), t + 1))
    while len(indices) < CONTEXT_SIZE + 1:
        indices.insert(0, indices[0])
    indices = indices[-(CONTEXT_SIZE + 1):]

    for i, idx in enumerate(indices):
        # 加载原始 PIL 图像（未归一化）
        img_pil = PILImage.open(_find_image(traj_path, idx)).convert("RGB")
        img_pil = img_pil.resize(IMAGE_SIZE)
        if angle > 0:
            phase = 2 * np.pi * i / (CONTEXT_SIZE + 1)
            img_pil = apply_gait_shake(img_pil, angle, sx, sy, phase)
        # 统一归一化（仅此一次）
        img = _TRANSFORM(img_pil)
        frames.append(img)
    return torch.cat(frames, dim=0).unsqueeze(0)


def cfg_inference(model, obs_cond_nav, obs_cond_explore, device,
                  scheduler, w, num_steps, num_samples=NUM_SAMPLES):
    """
    通用推理函数：支持 DDPM/DDIM + CFG。
    当 w=0 时为标准条件推理（单次前向，使用目标导航），
    当 w≠0 时执行 CFG 双路推理（w>0 增强引导，w=-1 纯探索）。
    """
    cond_rep   = obs_cond_nav.repeat(num_samples, 1)
    uncond_rep = obs_cond_explore.repeat(num_samples, 1)

    naction = torch.randn((num_samples, LEN_TRAJ_PRED, 2), device=device)
    scheduler.set_timesteps(num_steps)

    with torch.no_grad():
        for k in scheduler.timesteps:
            eps_cond = model("noise_pred_net", sample=naction,
                             timestep=k, global_cond=cond_rep)
            if w != 0.0:
                eps_uncond = model("noise_pred_net", sample=naction,
                                   timestep=k, global_cond=uncond_rep)
                eps_final = (1.0 + w) * eps_cond - w * eps_uncond
            else:
                eps_final = eps_cond

            naction = scheduler.step(
                model_output=eps_final, timestep=k, sample=naction,
            ).prev_sample
    return naction


def cosine_similarity(a, b):
    """两个轨迹之间的余弦相似度"""
    a_flat, b_flat = a.flatten(), b.flatten()
    return float(np.dot(a_flat, b_flat) / (
        np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8))


# ─────────────────────────────────────────────────────────────
# 主实验逻辑
# ─────────────────────────────────────────────────────────────
def run_ablation(model, device, traj_path, obs_time, goal_time, num_runs):
    """
    运行完整消融实验矩阵。
    返回 results: dict[config_key] -> dict of metrics
    """
    goal_img = prepare_goal(traj_path, goal_time).to(device)

    # 预计算各抖动级别的观测条件向量
    print("\n[Phase 1] Pre-computing vision encodings for all shake levels...")
    shake_encodings = {}
    for shake_name, angle, sx, sy in SHAKE_CONFIGS:
        obs = prepare_shaken_observation(traj_path, obs_time, angle, sx, sy).to(device)

        mask_nav     = torch.zeros(1).long().to(device)
        mask_explore = torch.ones(1).long().to(device)

        with torch.no_grad():
            cond_nav = model("vision_encoder", obs_img=obs,
                             goal_img=goal_img, input_goal_mask=mask_nav)
            cond_explore = model("vision_encoder", obs_img=obs,
                                 goal_img=goal_img, input_goal_mask=mask_explore)
        shake_encodings[shake_name] = (cond_nav, cond_explore)
        print(f"  {shake_name}: encoding shape = {cond_nav.shape}")

    # 获取 clean baseline 轨迹（DDPM-10, w=0, no_shake）
    baseline_sched = make_scheduler("ddpm")
    cond_nav_clean, cond_explore_clean = shake_encodings["no_shake"]
    baseline_trajs = cfg_inference(
        model, cond_nav_clean, cond_explore_clean, device,
        baseline_sched, w=0.0, num_steps=10,
    )
    baseline_mean = baseline_trajs.mean(dim=0).cpu().numpy()

    # 执行消融矩阵
    print(f"\n[Phase 2] Running ablation matrix: "
          f"{len(SAMPLER_CONFIGS)}×{len(GUIDANCE_CONFIGS)}×{len(SHAKE_CONFIGS)} "
          f"= {len(SAMPLER_CONFIGS)*len(GUIDANCE_CONFIGS)*len(SHAKE_CONFIGS)} configs "
          f"× {num_runs} runs each")
    print("=" * 80)

    results = {}
    total = len(SAMPLER_CONFIGS) * len(GUIDANCE_CONFIGS) * len(SHAKE_CONFIGS)
    done = 0

    for samp_name, samp_kind, samp_steps in SAMPLER_CONFIGS:
        for guid_name, w_val in GUIDANCE_CONFIGS:
            for shake_name, angle, sx, sy in SHAKE_CONFIGS:
                config_key = f"{samp_name} | {guid_name} | {shake_name}"
                done += 1
                print(f"\n[{done}/{total}] {config_key}")

                cond_nav, cond_explore = shake_encodings[shake_name]
                scheduler = make_scheduler(samp_kind)

                times = []
                all_trajs = []
                for r in range(num_runs):
                    t0 = time.time()
                    out = cfg_inference(
                        model, cond_nav, cond_explore, device,
                        scheduler, w=w_val, num_steps=samp_steps,
                    )
                    times.append(time.time() - t0)
                    all_trajs.append(out.cpu().numpy())

                all_trajs = np.array(all_trajs)               # [runs, 8, 8, 2]
                mean_traj = all_trajs.mean(axis=(0, 1))       # [8, 2]

                # 计算所有指标
                inference_ms = np.mean(times) * 1000
                diversity    = all_trajs.std(axis=(0, 1)).mean()
                mse_vs_clean = np.mean((mean_traj - baseline_mean) ** 2)
                cos_sim      = cosine_similarity(mean_traj, baseline_mean)
                heading_dev  = np.degrees(np.abs(
                    np.arctan2(mean_traj[0, 1], mean_traj[0, 0])
                    - np.arctan2(baseline_mean[0, 1], baseline_mean[0, 0])
                ))
                mean_disp    = np.sqrt((mean_traj ** 2).sum(axis=1)).mean()

                results[config_key] = dict(
                    sampler=samp_name, guidance=guid_name, shake=shake_name,
                    samp_steps=samp_steps, w=w_val,
                    inference_ms=inference_ms,
                    diversity=diversity,
                    mse_vs_clean=mse_vs_clean,
                    cos_sim=cos_sim,
                    heading_dev=heading_dev,
                    mean_disp=mean_disp,
                    mean_traj=mean_traj,
                )
                print(f"  Time: {inference_ms:.1f}ms | Div: {diversity:.4f} | "
                      f"MSE: {mse_vs_clean:.6f} | cos_sim: {cos_sim:.4f} | "
                      f"Heading Dev: {heading_dev:.2f} deg")

    return results, baseline_mean


# ─────────────────────────────────────────────────────────────
# 结果保存与可视化
# ─────────────────────────────────────────────────────────────
def save_results_txt(results, baseline_mean, out_dir, encoder_name="EfficientNet-B0"):
    """保存完整数值结果到 txt"""
    txt_path = os.path.join(out_dir, "ablation_full_results.txt")
    with open(txt_path, "w") as f:
        f.write(f"Run Time: {RUN_TAG}\n")
        f.write(f"Script: ablation_experiment.py\n")
        f.write(f"Encoder: {encoder_name}\n")
        f.write(f"Configs: {len(results)}\n")
        f.write(f"Runs per config: {NUM_RUNS}\n\n")
        f.write("# ============================================================\n")
        f.write("# 列说明:\n")
        f.write("# ============================================================\n")
        f.write("#   Sampler      - 扩散调度器及步数\n")
        f.write("#                  DDPM-10: 标准 10 步 DDPM (基线)\n")
        f.write("#                  DDIM-5:  5 步 DDIM (2x 更快)\n")
        f.write("#                  DDIM-3:  3 步 DDIM (3.3x 更快)\n")
        f.write("#   Guidance     - 无分类器引导权重 w\n")
        f.write("#                  w=0: 标准条件生成（使用目标导航，无额外增强）\n")
        f.write("#                  w=1.0: 中等目标导向性\n")
        f.write("#                  w=2.0: 激进目标导向性\n")
        f.write("#   Shake        - 模拟步态相机抖动强度\n")
        f.write("#                  模拟 DeepRobotics Lite3 机体运动\n")
        f.write("#   inference_ms - 平均推理延迟 (ms)\n")
        f.write("#                  越低越适合实时控制 (目标 < 67ms = 15Hz)\n")
        f.write("#   diversity    - 轨迹多样性 (路径点标准差)\n")
        f.write("#                  越高 = 更多探索; 越低 = 更集中\n")
        f.write("#   mse_vs_clean - 与 DDPM-10/w=0/无抖动基线的均方误差\n")
        f.write("#                  0 = 与基线完全相同; 衡量退化程度\n")
        f.write("#   cos_sim      - 与干净基线轨迹的余弦相似度\n")
        f.write("#                  1.0 = 完美对齐; < 0.9 = 显著偏差\n")
        f.write("#   heading_dev  - 第一个路径点的航向偏差 (度)\n")
        f.write("#                  对足式机器人至关重要: > 15° = 危险\n")
        f.write("#   mean_disp    - 路径点相对机器人原点的平均位移\n")
        f.write("#                  衡量目标导向性（前进量）\n\n")

        header = (f"{'Sampler':<12} {'Guidance':<10} {'Shake':<18} "
                  f"{'ms':<8} {'div':<8} {'MSE':<12} "
                  f"{'cos_sim':<8} {'hdg_dev':<8} {'disp':<8}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for key in sorted(results.keys()):
            r = results[key]
            f.write(f"{r['sampler']:<12} {r['guidance']:<10} {r['shake']:<18} "
                    f"{r['inference_ms']:<8.1f} {r['diversity']:<8.4f} "
                    f"{r['mse_vs_clean']:<12.6f} {r['cos_sim']:<8.4f} "
                    f"{r['heading_dev']:<8.2f} {r['mean_disp']:<8.4f}\n")

    print(f"Full results: {txt_path}")
    return txt_path


def save_summary_txt(results, out_dir):
    """保存关键发现摘要"""
    txt_path = os.path.join(out_dir, "ablation_summary.txt")

    # 按维度聚合
    by_sampler = {}
    by_guidance = {}
    by_shake = {}
    for key, r in results.items():
        by_sampler.setdefault(r["sampler"], []).append(r)
        by_guidance.setdefault(r["guidance"], []).append(r)
        by_shake.setdefault(r["shake"], []).append(r)

    with open(txt_path, "w") as f:
        f.write(f"Run Time: {RUN_TAG}\n")
        f.write(f"Script: ablation_experiment.py (Summary)\n\n")

        # 按 sampler 聚合
        f.write("=" * 60 + "\n")
        f.write("1. SAMPLER COMPARISON (averaged over guidance × shake)\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Sampler':<12} {'Avg ms':<10} {'Avg Div':<10} {'Avg MSE':<12} {'Avg cos':<10}\n")
        f.write("-" * 54 + "\n")
        for name, rs in sorted(by_sampler.items()):
            avg_ms  = np.mean([r["inference_ms"] for r in rs])
            avg_div = np.mean([r["diversity"] for r in rs])
            avg_mse = np.mean([r["mse_vs_clean"] for r in rs])
            avg_cos = np.mean([r["cos_sim"] for r in rs])
            f.write(f"{name:<12} {avg_ms:<10.1f} {avg_div:<10.4f} "
                    f"{avg_mse:<12.6f} {avg_cos:<10.4f}\n")

        # 按 guidance 聚合
        f.write(f"\n{'='*60}\n")
        f.write("2. GUIDANCE COMPARISON (averaged over sampler × shake)\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Guidance':<12} {'Avg ms':<10} {'Avg Div':<10} {'Avg Disp':<10}\n")
        f.write("-" * 42 + "\n")
        for name, rs in sorted(by_guidance.items()):
            avg_ms   = np.mean([r["inference_ms"] for r in rs])
            avg_div  = np.mean([r["diversity"] for r in rs])
            avg_disp = np.mean([r["mean_disp"] for r in rs])
            f.write(f"{name:<12} {avg_ms:<10.1f} {avg_div:<10.4f} {avg_disp:<10.4f}\n")

        # 按 shake 聚合
        f.write(f"\n{'='*60}\n")
        f.write("3. GAIT SHAKE ROBUSTNESS (averaged over sampler × guidance)\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Shake':<18} {'Avg cos_sim':<12} {'Avg Hdg Dev':<12} {'Avg MSE':<12}\n")
        f.write("-" * 54 + "\n")
        for name, rs in sorted(by_shake.items(), key=lambda x: x[1][0]["mse_vs_clean"]):
            avg_cos = np.mean([r["cos_sim"] for r in rs])
            avg_hdg = np.mean([r["heading_dev"] for r in rs])
            avg_mse = np.mean([r["mse_vs_clean"] for r in rs])
            f.write(f"{name:<18} {avg_cos:<12.4f} {avg_hdg:<12.2f} {avg_mse:<12.6f}\n")

        # 最佳配置推荐
        f.write(f"\n{'='*60}\n")
        f.write("4. RECOMMENDED CONFIGURATIONS\n")
        f.write("=" * 60 + "\n")

        # 最佳实时性能（抖动鲁棒）
        noshake = {k: v for k, v in results.items() if "no_shake" in k}
        best_rt = min(noshake.items(),
                      key=lambda x: x[1]["inference_ms"])
        f.write(f"\n  [最佳实时性能]: {best_rt[0]}\n")
        f.write(f"    Latency: {best_rt[1]['inference_ms']:.1f} ms "
                f"({1000/best_rt[1]['inference_ms']:.0f} Hz)\n")

        # 最佳质量（抖动鲁棒）
        severe = {k: v for k, v in results.items() if "severe" in k}
        best_robust = max(severe.items(),
                          key=lambda x: x[1]["cos_sim"])
        f.write(f"\n  [最佳鲁棒性 (严重抖动)]: {best_robust[0]}\n")
        f.write(f"    cos_sim: {best_robust[1]['cos_sim']:.4f}, "
                f"Heading Dev: {best_robust[1]['heading_dev']:.2f} deg\n")

        # 最佳综合（Pareto 点）
        # 归一化打分：速度(越低越好)×0.3 + 抖动鲁棒(cos_sim越高越好)×0.4 + 目标导向×0.3
        all_ms = [r["inference_ms"] for r in results.values()]
        all_cos = [r["cos_sim"] for r in results.values()]
        all_disp = [r["mean_disp"] for r in results.values()]
        ms_range = max(all_ms) - min(all_ms) + 1e-8
        cos_range = max(all_cos) - min(all_cos) + 1e-8
        disp_range = max(all_disp) - min(all_disp) + 1e-8

        best_score, best_key = -1, None
        for key, r in results.items():
            score = (0.3 * (1 - (r["inference_ms"] - min(all_ms)) / ms_range)
                     + 0.4 * ((r["cos_sim"] - min(all_cos)) / cos_range)
                     + 0.3 * ((r["mean_disp"] - min(all_disp)) / disp_range))
            if score > best_score:
                best_score, best_key = score, key
        f.write(f"\n  [最佳综合 (加权评分)]: {best_key}\n")
        f.write(f"    Score: {best_score:.4f} "
                f"(延迟×0.3 + 鲁棒性×0.4 + 目标导向性×0.3)\n")
        r = results[best_key]
        f.write(f"    Latency: {r['inference_ms']:.1f}ms, "
                f"cos_sim: {r['cos_sim']:.4f}, disp: {r['mean_disp']:.4f}\n")

    print(f"Summary: {txt_path}")
    return txt_path


def plot_heatmaps(results, out_dir):
    """绘制 采样器 × 引导强度 热力图（以 no_shake 为例）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    samplers = [s[0] for s in SAMPLER_CONFIGS]
    guidances = [g[0] for g in GUIDANCE_CONFIGS]

    for ax_idx, (metric, title, cmap) in enumerate([
        ("inference_ms", "Inference Time (ms)", "YlOrRd"),
        ("diversity",    "Trajectory Diversity", "YlGnBu"),
        ("mean_disp",   "Mean Displacement", "RdYlGn"),
    ]):
        matrix = np.zeros((len(samplers), len(guidances)))
        for i, s in enumerate(samplers):
            for j, g in enumerate(guidances):
                key = f"{s} | {g} | no_shake"
                if key in results:
                    matrix[i, j] = results[key][metric]

        im = axes[ax_idx].imshow(matrix, cmap=cmap, aspect="auto")
        axes[ax_idx].set_xticks(range(len(guidances)))
        axes[ax_idx].set_xticklabels(guidances)
        axes[ax_idx].set_yticks(range(len(samplers)))
        axes[ax_idx].set_yticklabels(samplers)
        axes[ax_idx].set_xlabel("Guidance")
        axes[ax_idx].set_ylabel("Sampler")
        axes[ax_idx].set_title(title)
        # 在每个格子中显示数值
        for i in range(len(samplers)):
            for j in range(len(guidances)):
                val = matrix[i, j]
                fmt = f"{val:.1f}" if metric == "inference_ms" else f"{val:.4f}"
                axes[ax_idx].text(j, i, fmt, ha="center", va="center",
                                  fontsize=9, color="black")
        plt.colorbar(im, ax=axes[ax_idx], shrink=0.8)

    plt.suptitle("Ablation Heatmap: Sampler x Guidance (no_shake)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap: {path}")


def plot_shake_robustness(results, out_dir):
    """绘制各配置在不同抖动级别下的鲁棒性"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    shakes = [s[0] for s in SHAKE_CONFIGS]
    # 选择代表性配置
    configs_to_show = [
        ("DDPM-10", "w=0"),
        ("DDIM-5",  "w=0"),
        ("DDIM-5",  "w=1.0"),
        ("DDIM-5",  "w=2.0"),
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    for idx, (samp, guid) in enumerate(configs_to_show):
        cos_vals = []
        hdg_vals = []
        for shake in shakes:
            key = f"{samp} | {guid} | {shake}"
            if key in results:
                cos_vals.append(results[key]["cos_sim"])
                hdg_vals.append(results[key]["heading_dev"])
            else:
                cos_vals.append(0)
                hdg_vals.append(0)

        label = f"{samp}/{guid}"
        axes[0].plot(range(len(shakes)), cos_vals, "o-",
                     color=colors[idx], label=label, linewidth=2)
        axes[1].plot(range(len(shakes)), hdg_vals, "s-",
                     color=colors[idx], label=label, linewidth=2)

    axes[0].set_xticks(range(len(shakes)))
    axes[0].set_xticklabels(shakes, rotation=15, fontsize=9)
    axes[0].set_ylabel("Cosine Similarity vs Baseline")
    axes[0].set_title("Trajectory Direction Robustness")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.7, 1.05)

    axes[1].set_xticks(range(len(shakes)))
    axes[1].set_xticklabels(shakes, rotation=15, fontsize=9)
    axes[1].set_ylabel("Heading Deviation (deg)")
    axes[1].set_title("First Waypoint Heading Error")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Gait Shake Robustness Analysis (Lite3)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_shake_robustness.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Shake robustness: {path}")


def plot_pareto(results, out_dir):
    """速度-质量 Pareto 前沿图"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # 仅取 no_shake 配置来展示速度-质量权衡
    noshake = {k: v for k, v in results.items() if "no_shake" in k}

    for key, r in noshake.items():
        color = "#e74c3c" if "DDPM" in key else "#3498db"
        marker = "o" if "w=0" in key else ("s" if "w=1" in key else "^")
        ax.scatter(r["inference_ms"], r["diversity"],
                   c=color, marker=marker, s=100, zorder=5)
        ax.annotate(f"{r['sampler']}/{r['guidance']}",
                    (r["inference_ms"], r["diversity"]),
                    fontsize=7, textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("Inference Time (ms) — lower is better")
    ax.set_ylabel("Diversity (std) — higher = more exploration")
    ax.set_title("Speed-Quality Pareto Front (no_shake)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_pareto.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Pareto: {path}")


def plot_radar(results, out_dir):
    """最佳配置的雷达图"""
    # 选择 DDIM-5/w=1.0/no_shake 作为推荐配置
    target_key = "DDIM-5 | w=1.0 | no_shake"
    baseline_key = "DDPM-10 | w=0 | no_shake"

    if target_key not in results or baseline_key not in results:
        print("  [SKIP] Radar chart: target config not found")
        return

    categories = ["Speed", "Diversity", "Robustness", "Directedness", "Quality"]
    N = len(categories)

    def normalize(val, lo, hi):
        return (val - lo) / (hi - lo + 1e-8)

    all_ms   = [r["inference_ms"] for r in results.values()]
    all_div  = [r["diversity"] for r in results.values()]
    all_cos  = [r["cos_sim"] for r in results.values()]
    all_disp = [r["mean_disp"] for r in results.values()]
    all_mse  = [r["mse_vs_clean"] for r in results.values()]

    def compute_radar(r):
        return [
            1 - normalize(r["inference_ms"], min(all_ms), max(all_ms)),  # 速度（越低越好 → 反转）
            normalize(r["diversity"], min(all_div), max(all_div)),
            normalize(r["cos_sim"], min(all_cos), max(all_cos)),
            normalize(r["mean_disp"], min(all_disp), max(all_disp)),
            1 - normalize(r["mse_vs_clean"], min(all_mse), max(all_mse)),  # 质量（越低越好 → 反转）
        ]

    values_target   = compute_radar(results[target_key])
    values_baseline = compute_radar(results[baseline_key])

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values_target   += values_target[:1]
    values_baseline += values_baseline[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, values_baseline, "o-", color="#e74c3c", linewidth=2, label="Baseline (DDPM-10/w=0)")
    ax.fill(angles, values_baseline, alpha=0.1, color="#e74c3c")
    ax.plot(angles, values_target, "s-", color="#2ecc71", linewidth=2, label="Recommended (DDIM-5/w=1.0)")
    ax.fill(angles, values_target, alpha=0.1, color="#2ecc71")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Ablation Radar: Baseline vs Recommended", y=1.08)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Radar: {path}")


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Day 6: 综合消融实验")
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS,
                        help="每个配置的重复次数 (默认 5)")
    parser.add_argument("--dinov2-weights", type=str, default=None,
                        help="DINOv2 训练模型权重路径 (可选)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── EfficientNet 基线模型 ──
    print("\n[加载] EfficientNet-B0 基线模型...")
    model = build_model(device)

    # 准备测试数据
    trajs = sorted(os.listdir(DATASET_ROOT))
    traj_path = os.path.join(DATASET_ROOT, trajs[0])
    obs_time  = CONTEXT_SIZE + 2
    goal_time = obs_time + 15

    out_dir = os.path.join(PROJECT_ROOT, f"results/day6/{RUN_TAG}_ablation")
    os.makedirs(out_dir, exist_ok=True)

    # ── 运行 EfficientNet 消融 ──
    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT: EfficientNet-B0 Encoder")
    print("=" * 80)
    results_eff, baseline_mean = run_ablation(
        model, device, traj_path, obs_time, goal_time, args.num_runs)

    # 保存结果
    save_results_txt(results_eff, baseline_mean, out_dir, "EfficientNet-B0")
    save_summary_txt(results_eff, out_dir)
    plot_heatmaps(results_eff, out_dir)
    plot_shake_robustness(results_eff, out_dir)
    plot_pareto(results_eff, out_dir)
    plot_radar(results_eff, out_dir)

    # ── 可选: DINOv2 对比 ──
    if args.dinov2_weights and os.path.exists(args.dinov2_weights):
        print("\n" + "=" * 80)
        print("ABLATION EXPERIMENT: DINOv2-Small Encoder")
        print("=" * 80)

        # TODO: 加载 DINOv2 模型并运行相同的消融矩阵
        # 需要 train_dinov2.py 训练完成后的权重
        dinov2_dir = os.path.join(out_dir, "dinov2")
        os.makedirs(dinov2_dir, exist_ok=True)
        print(f"  [INFO] DINOv2 ablation would be saved to: {dinov2_dir}")
        print(f"  [INFO] DINOv2 weights: {args.dinov2_weights}")
        print(f"  [TODO] 实现 DINOv2 模型加载 + 相同的消融矩阵")

    # ── 打印最终摘要 ──
    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {out_dir}")
    print(f"Files generated:")
    for f in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, f)
        size = os.path.getsize(fpath)
        print(f"  {f:<40} ({size:,} bytes)")
    print(f"\nTotal configurations tested: {len(results_eff)}")
    print(f"Runs per configuration: {args.num_runs}")


if __name__ == "__main__":
    main()
