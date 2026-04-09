#!/usr/bin/env python3
"""
Day 2: DDIM 加速采样对比实验
对比 DDPM-10 步 vs DDIM-{10,5,3,2,1} 步的推理时间、轨迹质量和多样性。
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from project_paths import REPO_ROOT, add_repo_paths

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

PROJECT_ROOT = str(REPO_ROOT)
add_repo_paths()

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from offline_inference import (
    build_model, prepare_observation, prepare_goal,
    DATASET_ROOT, CONTEXT_SIZE, IMAGE_SIZE, LEN_TRAJ_PRED,
)

NUM_RUNS = 10   # 每个配置的重复次数（用于求平均）


def inference_with_scheduler(model, obs_cond, device, scheduler,
                             num_samples=8, num_steps=10):
    """通用扩散推理，支持任意调度器和步数"""
    cond = obs_cond.repeat(num_samples, 1)
    naction = torch.randn((num_samples, LEN_TRAJ_PRED, 2), device=device)

    scheduler.set_timesteps(num_steps)
    with torch.no_grad():
        for k in scheduler.timesteps:
            noise_pred = model(
                "noise_pred_net", sample=naction,
                timestep=k, global_cond=cond,
            )
            naction = scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction,
            ).prev_sample
    return naction


def make_scheduler(kind: str, num_train_timesteps: int = 10):
    common = dict(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    if kind == "ddpm":
        return DDPMScheduler(**common)
    return DDIMScheduler(**common)


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

    mask = torch.zeros(1).long().to(device)
    with torch.no_grad():
        obsgoal_cond = model(
            "vision_encoder", obs_img=obs_img,
            goal_img=goal_img, input_goal_mask=mask,
        )

    # ── Experiment Matrix ──────────────────────────────────────────
    experiments = [
        ("DDPM-10 (baseline)", "ddpm", 10),
        ("DDIM-10",            "ddim", 10),
        ("DDIM-5",             "ddim", 5),
        ("DDIM-3",             "ddim", 3),
        ("DDIM-2",             "ddim", 2),
        ("DDIM-1",             "ddim", 1),
    ]

    # 基线轨迹（用于计算 MSE 偏差）
    baseline_trajs = inference_with_scheduler(
        model, obsgoal_cond, device,
        make_scheduler("ddpm"), num_samples=8, num_steps=10,
    )
    baseline_mean = baseline_trajs.mean(dim=0).cpu().numpy()

    results = {}
    print("=" * 70)
    print("DDIM Accelerated Sampling Experiment")
    print("=" * 70)

    for name, kind, steps in experiments:
        print(f"\n--- {name} ---")
        scheduler = make_scheduler(kind)

        times, all_trajs = [], []
        for _ in range(NUM_RUNS):
            t0 = time.time()
            out = inference_with_scheduler(
                model, obsgoal_cond, device, scheduler,
                num_samples=8, num_steps=steps,
            )
            times.append(time.time() - t0)
            all_trajs.append(out.cpu().numpy())

        all_trajs = np.array(all_trajs)              # [runs, 8, 8, 2]
        mean_traj = all_trajs.mean(axis=(0, 1))      # [8, 2]
        mse       = np.mean((mean_traj - baseline_mean) ** 2)
        diversity = all_trajs.std(axis=(0, 1)).mean()

        results[name] = dict(
            avg_ms=np.mean(times) * 1000,
            std_ms=np.std(times) * 1000,
            mse=mse, diversity=diversity,
            mean_traj=mean_traj, steps=steps,
        )
        print(f"  Time: {results[name]['avg_ms']:.2f} ± {results[name]['std_ms']:.2f} ms")
        print(f"  MSE:  {mse:.6f}    Diversity: {diversity:.4f}")

    # ── Visualization ─────────────────────────────────────
    out_dir = os.path.join(PROJECT_ROOT, "results/day2")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names  = list(results.keys())
    colors = ["#e74c3c" if "DDPM" in n else "#3498db" for n in names]

    # 图表 1: 推理时间
    axes[0].bar(range(len(names)),
                [results[n]["avg_ms"] for n in names], color=colors)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Inference Time (ms)")
    axes[0].set_title("Inference Time Comparison")

    # 图表 2: 相对基线的 MSE
    axes[1].bar(range(len(names)),
                [results[n]["mse"] for n in names], color=colors)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("MSE vs Baseline")
    axes[1].set_title("Trajectory Quality Deviation (lower is better)")

    # 图表 3: 轨迹可视化
    for n in names:
        tr = results[n]["mean_traj"]
        ls = "--" if "DDPM" in n else "-"
        axes[2].plot(tr[:, 1], tr[:, 0], ls + "o", label=n,
                     markersize=3, alpha=0.7)
    axes[2].plot(0, 0, "r*", markersize=15)
    axes[2].legend(fontsize=7)
    axes[2].set_title("Mean Trajectory Comparison")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("x")
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ddim_comparison.png"), dpi=150)
    plt.close()

    # ── 保存结果到 txt ─────────────────────────
    txt_path = os.path.join(out_dir, f"{RUN_TAG}_ddim_results.txt")
    with open(txt_path, "w") as f:
        f.write(f"Run Time: {RUN_TAG}\n")
        f.write(f"Script: ddim_experiment.py\n")
        f.write(f"Repeat runs per config: {NUM_RUNS}\n\n")
        f.write("# 列说明:\n")
        f.write("#   Config      - 调度器类型及去噪步数\n")
        f.write("#   Time(ms)    - 平均推理延迟 (ms)，越低越好\n")
        f.write("#   Std(ms)     - 推理时间标准差 (ms)\n")
        f.write("#   MSE         - 相对 DDPM-10 基线轨迹的均方误差\n")
        f.write("#                 衡量轨迹质量退化程度；0 = 完全一致\n")
        f.write("#   Diversity   - 采样轨迹路径点的标准差\n")
        f.write("#                 越高 = 轨迹越多样\n")
        f.write("#   Steps       - 使用的去噪步数\n")
        f.write("#   Speedup     - 相对 DDPM 基线的加速比\n\n")
        baseline_ms = results[names[0]]["avg_ms"]
        f.write(f"{'Config':<22} {'Time(ms)':<12} {'Std(ms)':<10} {'MSE':<14} "
                f"{'Diversity':<12} {'Steps':<8} {'Speedup':<8}\n")
        f.write("-" * 86 + "\n")
        for n in names:
            r = results[n]
            speedup = baseline_ms / r["avg_ms"] if r["avg_ms"] > 0 else 0
            f.write(f"{n:<22} {r['avg_ms']:<12.2f} {r['std_ms']:<10.2f} "
                    f"{r['mse']:<14.6f} {r['diversity']:<12.4f} "
                    f"{r['steps']:<8d} {speedup:<8.2f}x\n")
        f.write(f"\n# 关键发现:\n")
        # 自动总结最佳配置
        best_tradeoff = min(
            [(n, r) for n, r in results.items() if "DDIM" in n],
            key=lambda x: x[1]["avg_ms"] + x[1]["mse"] * 10000,
        )
        f.write(f"#   Best speed-quality tradeoff: {best_tradeoff[0]}\n")
        f.write(f"#     Time: {best_tradeoff[1]['avg_ms']:.2f} ms, "
                f"MSE: {best_tradeoff[1]['mse']:.6f}\n")
        fastest = min(results.items(), key=lambda x: x[1]["avg_ms"])
        f.write(f"#   Fastest config: {fastest[0]} ({fastest[1]['avg_ms']:.2f} ms)\n")
    print(f"\nResults saved: {txt_path}")

    # ── Summary Table ─────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Config':<22} {'Time(ms)':<14} {'MSE':<14} {'Diversity':<10}")
    print("-" * 70)
    for n in names:
        r = results[n]
        print(f"{n:<22} {r['avg_ms']:<14.2f} {r['mse']:<14.6f} {r['diversity']:<10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
