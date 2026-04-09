#!/usr/bin/env python3
"""
DDIM × CFG 联合优化实验 —— 确定 NoMaD 扩散策略的最优部署配置。

实验设计:
  - DDIM 步数: {10(DDPM基线), 10(DDIM), 5, 3, 2}
  - CFG 引导强度: {0.0(无引导), 0.5, 1.0, 2.0}
  - 共 20 种组合, 每种在多个测试 case 上重复多次采样
  - 输出: 所有指标的均值/标准差/CI95, 综合评分排名, 推荐最优配置

指标体系:
  1. latency_ms        — 推理延迟 (越低越好)
  2. forward_progress   — 轨迹前进距离 (越大越好)
  3. diversity          — 采样多样性 (适中为佳, 太低=过拟合, 太高=发散)
  4. smoothness         — 轨迹平滑度 (越低越好)
  5. lateral_abs        — 横向偏移 (越低越好)
  6. endpoint_norm      — 终点距离 (越大通常越好)
  7. composite_score    — 综合加权评分

运行:
  python scripts/joint_ddim_cfg_experiment.py --cases-per-suite 8 --num-runs 8
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from project_paths import add_repo_paths
add_repo_paths()

from nomad_eval_common import (
    aggregate_case_records,
    build_eval_cases,
    compute_sample_metrics,
    encode_cfg_conditions,
    encode_navigation_condition,
    load_case_tensors,
    load_nomad_model,
    make_output_dir,
    parse_suite_offsets,
    run_diffusion_sampling,
    save_cases_json,
    save_markdown_table,
    save_rows_csv,
    save_rows_json,
    set_seed,
)


# ── 实验配置网格 ──────────────────────────────────────────────
@dataclass(frozen=True)
class ConfigCombo:
    """一种 DDIM 步数 × CFG 强度的组合。"""
    name: str
    scheduler_kind: str   # "ddpm" 或 "ddim"
    num_steps: int
    guidance_scale: float  # 0.0 = 无 CFG


# 默认实验网格
DEFAULT_GRID: List[ConfigCombo] = [
    # DDPM 基线 (10 steps)
    ConfigCombo("ddpm10_w0.0", "ddpm", 10, 0.0),
    ConfigCombo("ddpm10_w0.5", "ddpm", 10, 0.5),
    ConfigCombo("ddpm10_w1.0", "ddpm", 10, 1.0),
    ConfigCombo("ddpm10_w2.0", "ddpm", 10, 2.0),
    # DDIM-10
    ConfigCombo("ddim10_w0.0", "ddim", 10, 0.0),
    ConfigCombo("ddim10_w0.5", "ddim", 10, 0.5),
    ConfigCombo("ddim10_w1.0", "ddim", 10, 1.0),
    ConfigCombo("ddim10_w2.0", "ddim", 10, 2.0),
    # DDIM-5
    ConfigCombo("ddim5_w0.0",  "ddim",  5, 0.0),
    ConfigCombo("ddim5_w0.5",  "ddim",  5, 0.5),
    ConfigCombo("ddim5_w1.0",  "ddim",  5, 1.0),
    ConfigCombo("ddim5_w2.0",  "ddim",  5, 2.0),
    # DDIM-3
    ConfigCombo("ddim3_w0.0",  "ddim",  3, 0.0),
    ConfigCombo("ddim3_w0.5",  "ddim",  3, 0.5),
    ConfigCombo("ddim3_w1.0",  "ddim",  3, 1.0),
    ConfigCombo("ddim3_w2.0",  "ddim",  3, 2.0),
    # DDIM-2
    ConfigCombo("ddim2_w0.0",  "ddim",  2, 0.0),
    ConfigCombo("ddim2_w0.5",  "ddim",  2, 0.5),
    ConfigCombo("ddim2_w1.0",  "ddim",  2, 1.0),
    ConfigCombo("ddim2_w2.0",  "ddim",  2, 2.0),
]

METRIC_FIELDS = [
    "latency_ms",
    "diversity",
    "path_length_mean",
    "endpoint_norm_mean",
    "forward_progress_mean",
    "lateral_abs_mean",
    "smoothness_mean",
]


def evaluate_combo(
    combo: ConfigCombo,
    model,
    cases,
    device: torch.device,
    num_runs: int,
    num_samples: int,
    seed: int,
) -> List[Dict[str, object]]:
    """对一种配置组合, 在所有 case 上做重复采样评估。"""
    records: List[Dict[str, object]] = []
    use_cfg = combo.guidance_scale != 0.0

    for case_idx, case in enumerate(cases):
        obs_img, goal_img = load_case_tensors(case, device)

        # 编码视觉条件
        if use_cfg:
            cond, uncond = encode_cfg_conditions(model, obs_img, goal_img, device)
        else:
            cond = encode_navigation_condition(model, obs_img, goal_img, device)
            uncond = None

        latency_values: List[float] = []
        sample_batches: List[np.ndarray] = []

        for run_idx in range(num_runs):
            set_seed(seed + case_idx * 1000 + run_idx)
            t0 = time.perf_counter()
            samples = run_diffusion_sampling(
                model=model,
                cond=cond,
                device=device,
                scheduler_kind=combo.scheduler_kind,
                num_steps=combo.num_steps,
                num_samples=num_samples,
                guidance_scale=combo.guidance_scale if use_cfg else None,
                uncond=uncond,
            )
            latency_values.append((time.perf_counter() - t0) * 1000.0)
            sample_batches.append(samples)

        all_samples = np.concatenate(sample_batches, axis=0)
        metrics = compute_sample_metrics(all_samples)

        record: Dict[str, object] = {
            "config_name": combo.name,
            "scheduler": combo.scheduler_kind,
            "num_steps": combo.num_steps,
            "guidance_scale": combo.guidance_scale,
            "dataset_name": case.dataset_name,
            "suite_name": case.suite_name,
            "traj_name": case.traj_name,
            "latency_ms": float(np.mean(latency_values)),
            "latency_std_ms": float(np.std(latency_values)),
        }
        record.update(metrics)
        records.append(record)

    return records


def compute_composite_score(row: Dict[str, object]) -> float:
    """
    计算综合评分。越高越好。

    评分策略:
      - 延迟分: 基线 30ms 归一化, 越低越好 → 30 / latency
      - 前进分: forward_progress 直接用, 越大越好
      - 平滑分: 基线 0.15 归一化, 越低越好 → 0.15 / smoothness
      - 横向分: 惩罚横向偏移 → max(0, 5 - lateral_abs)
      - 多样性分: 适中为佳 → 衰减过高或过低的值

    权重: 延迟30%, 前进30%, 平滑15%, 横向15%, 多样性10%
    """
    latency = float(row.get("latency_ms_mean", 30.0))
    forward = float(row.get("forward_progress_mean_mean", 0.0))
    smooth = float(row.get("smoothness_mean_mean", 0.15))
    lateral = float(row.get("lateral_abs_mean_mean", 3.0))
    diversity = float(row.get("diversity_mean", 0.3))

    # 延迟分: 30ms 基准, 最高 10 分
    latency_score = min(10.0, 30.0 / max(latency, 1.0))
    # 前进分: 直接归一化到 0-10
    forward_score = min(10.0, forward / 1.0)
    # 平滑分: 0.15 基准, 越低越好
    smooth_score = min(10.0, 0.15 / max(smooth, 0.01))
    # 横向分: 越小越好
    lateral_score = max(0.0, min(10.0, (5.0 - lateral) / 0.5))
    # 多样性分: 0.2-0.4 是理想区间
    if diversity < 0.1:
        div_score = diversity / 0.1 * 5.0
    elif diversity < 0.5:
        div_score = 10.0 - abs(diversity - 0.3) / 0.2 * 5.0
    else:
        div_score = max(0.0, 5.0 - (diversity - 0.5) * 10.0)

    composite = (
        0.30 * latency_score +
        0.30 * forward_score +
        0.15 * smooth_score +
        0.15 * lateral_score +
        0.10 * div_score
    )
    return round(composite, 4)


def plot_heatmap(overall_rows: List[Dict[str, object]], out_dir: Path) -> None:
    """绘制 DDIM 步数 × CFG 强度的综合评分热力图。"""
    # 提取唯一的步数和引导强度
    steps_set = sorted(set(int(r["num_steps"]) for r in overall_rows))
    w_set = sorted(set(float(r["guidance_scale"]) for r in overall_rows))

    # 构造评分矩阵
    score_matrix = np.full((len(steps_set), len(w_set)), np.nan)
    scheduler_labels = {}
    for row in overall_rows:
        si = steps_set.index(int(row["num_steps"]))
        wi = w_set.index(float(row["guidance_scale"]))
        score_matrix[si, wi] = row.get("composite_score", 0.0)
        scheduler_labels[(si, wi)] = str(row.get("scheduler", ""))

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(score_matrix, cmap="YlOrRd", aspect="auto", origin="lower")

    ax.set_xticks(range(len(w_set)))
    ax.set_xticklabels([f"w={w}" for w in w_set])
    ax.set_yticks(range(len(steps_set)))
    y_labels = []
    for si, steps in enumerate(steps_set):
        # 标注哪些是 ddpm
        schedulers_at = set(scheduler_labels.get((si, wi), "") for wi in range(len(w_set)))
        prefix = "DDPM" if "ddpm" in schedulers_at else "DDIM"
        y_labels.append(f"{prefix}-{steps}")
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("CFG Guidance Scale (w)")
    ax.set_ylabel("Scheduler / Steps")
    ax.set_title("DDIM × CFG Joint Optimization — Composite Score")

    # 在每个格子里标注数值
    for si in range(len(steps_set)):
        for wi in range(len(w_set)):
            val = score_matrix[si, wi]
            if not np.isnan(val):
                color = "white" if val > score_matrix[~np.isnan(score_matrix)].mean() else "black"
                ax.text(wi, si, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Composite Score")
    plt.tight_layout()
    plt.savefig(out_dir / "joint_heatmap.png", dpi=160)
    plt.close(fig)


def plot_pareto(overall_rows: List[Dict[str, object]], out_dir: Path) -> None:
    """绘制 延迟 vs 前进距离 的帕累托前沿图。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    latencies = [float(r["latency_ms_mean"]) for r in overall_rows]
    forwards = [float(r["forward_progress_mean_mean"]) for r in overall_rows]
    names = [str(r["config_name"]) for r in overall_rows]
    scores = [float(r.get("composite_score", 0)) for r in overall_rows]

    scatter = ax.scatter(latencies, forwards, c=scores, cmap="YlOrRd", s=100, edgecolors="black", zorder=5)
    for i, name in enumerate(names):
        ax.annotate(name, (latencies[i], forwards[i]), fontsize=6, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points", rotation=30)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Forward Progress")
    ax.set_title("Latency vs Forward Progress (color = composite score)")
    plt.colorbar(scatter, ax=ax, label="Composite Score")
    plt.tight_layout()
    plt.savefig(out_dir / "joint_pareto.png", dpi=160)
    plt.close(fig)


def plot_metric_bars(overall_rows: List[Dict[str, object]], out_dir: Path) -> None:
    """绘制各配置的关键指标柱状对比图。"""
    names = [str(r["config_name"]) for r in overall_rows]
    n = len(names)
    x = np.arange(n)

    metrics_to_plot = [
        ("latency_ms_mean", "Latency (ms)", "#3498db"),
        ("forward_progress_mean_mean", "Forward Progress", "#2ecc71"),
        ("smoothness_mean_mean", "Smoothness", "#e74c3c"),
        ("diversity_mean", "Diversity", "#9b59b6"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (metric_key, label, color) in zip(axes.flatten(), metrics_to_plot):
        values = [float(r.get(metric_key, 0)) for r in overall_rows]
        ax.bar(x, values, color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel(label)
        ax.set_title(label)
    plt.tight_layout()
    plt.savefig(out_dir / "joint_metrics_bars.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="DDIM × CFG 联合优化实验")
    parser.add_argument("--dataset-root", action="append", default=None)
    parser.add_argument("--suite-offsets", default="short:8,medium:16,long:24")
    parser.add_argument("--cases-per-suite", type=int, default=8)
    parser.add_argument("--num-runs", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 加载模型
    print("Loading NoMaD model...")
    model, weights_path = load_nomad_model(device=device)
    print(f"  Weights: {weights_path}")
    print(f"  Device: {device}")

    # 2) 构造评估 case
    suite_offsets = parse_suite_offsets(args.suite_offsets)
    cases = build_eval_cases(
        dataset_roots=args.dataset_root,
        suite_offsets=suite_offsets,
        cases_per_suite=args.cases_per_suite,
        seed=args.seed,
    )
    print(f"  Eval cases: {len(cases)}")

    # 3) 逐配置评估
    all_records: List[Dict[str, object]] = []
    total = len(DEFAULT_GRID)

    for idx, combo in enumerate(DEFAULT_GRID):
        print(f"\n[{idx+1}/{total}] {combo.name}  (scheduler={combo.scheduler_kind}, "
              f"steps={combo.num_steps}, w={combo.guidance_scale})")
        records = evaluate_combo(
            combo=combo,
            model=model,
            cases=cases,
            device=device,
            num_runs=args.num_runs,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        all_records.extend(records)
        # 实时进度
        mean_lat = np.mean([float(r["latency_ms"]) for r in records])
        mean_fwd = np.mean([float(r["forward_progress_mean"]) for r in records])
        print(f"  -> latency={mean_lat:.1f}ms, forward={mean_fwd:.2f}")

    # 4) 聚合统计
    print("\n" + "=" * 70)
    print("Aggregating results...")

    overall_rows = aggregate_case_records(
        records=all_records,
        group_fields=["config_name", "scheduler", "num_steps", "guidance_scale"],
        metric_fields=METRIC_FIELDS,
    )

    # 5) 计算综合评分
    for row in overall_rows:
        row["composite_score"] = compute_composite_score(row)

    # 6) 按综合评分排序
    overall_rows.sort(key=lambda r: float(r.get("composite_score", 0)), reverse=True)

    # 7) 输出排名
    print("\n" + "=" * 70)
    print("RANKING (by composite score)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Config':<18} {'Score':<8} {'Latency(ms)':<14} "
          f"{'Forward':<10} {'Smooth':<10} {'Lateral':<10} {'Diversity':<10}")
    print("-" * 95)
    for rank, row in enumerate(overall_rows, 1):
        print(f"{rank:<5} {str(row['config_name']):<18} "
              f"{float(row['composite_score']):<8.4f} "
              f"{float(row.get('latency_ms_mean', 0)):<14.2f} "
              f"{float(row.get('forward_progress_mean_mean', 0)):<10.3f} "
              f"{float(row.get('smoothness_mean_mean', 0)):<10.4f} "
              f"{float(row.get('lateral_abs_mean_mean', 0)):<10.3f} "
              f"{float(row.get('diversity_mean', 0)):<10.4f}")

    best = overall_rows[0]
    print(f"\n*** BEST CONFIG: {best['config_name']} (score={best['composite_score']}) ***")
    print(f"    Scheduler: {best['scheduler']}, Steps: {best['num_steps']}, "
          f"CFG w={best['guidance_scale']}")
    print(f"    Latency: {float(best.get('latency_ms_mean', 0)):.2f} ms")
    print(f"    Forward Progress: {float(best.get('forward_progress_mean_mean', 0)):.3f}")
    print(f"    Smoothness: {float(best.get('smoothness_mean_mean', 0)):.4f}")

    # 8) 保存结果
    out_dir = make_output_dir("day4", "joint_ddim_cfg_experiment")
    save_cases_json(out_dir / "eval_cases.json", cases)
    save_rows_csv(out_dir / "case_records.csv", all_records)
    save_rows_csv(out_dir / "overall_summary.csv", overall_rows)
    save_rows_json(out_dir / "overall_summary.json", overall_rows)
    save_markdown_table(out_dir / "overall_summary.md", overall_rows)

    # 保存最优配置到单独文件
    import json
    best_config = {
        "config_name": str(best["config_name"]),
        "scheduler": str(best["scheduler"]),
        "num_steps": int(best["num_steps"]),
        "guidance_scale": float(best["guidance_scale"]),
        "composite_score": float(best["composite_score"]),
        "latency_ms": float(best.get("latency_ms_mean", 0)),
        "forward_progress": float(best.get("forward_progress_mean_mean", 0)),
        "smoothness": float(best.get("smoothness_mean_mean", 0)),
        "diversity": float(best.get("diversity_mean", 0)),
    }
    with open(out_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # README
    with open(out_dir / "README.txt", "w") as f:
        f.write(f"Joint DDIM × CFG Optimization Experiment\n")
        f.write(f"Weights: {weights_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Cases: {len(cases)}\n")
        f.write(f"Runs per config: {args.num_runs}\n")
        f.write(f"Samples per run: {args.num_samples}\n")
        f.write(f"Total configs: {len(DEFAULT_GRID)}\n")
        f.write(f"\nBest config: {best_config['config_name']}\n")
        f.write(f"  Score: {best_config['composite_score']}\n")
        f.write(f"  Latency: {best_config['latency_ms']:.2f} ms\n")
        f.write(f"  Forward: {best_config['forward_progress']:.3f}\n")

    # 9) 绘图
    plot_heatmap(overall_rows, out_dir)
    plot_pareto(overall_rows, out_dir)
    plot_metric_bars(overall_rows, out_dir)

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
