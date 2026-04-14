#!/usr/bin/env python3
"""MuJoCo 闭环导航基准实验：比较不同视觉编码器的导航性能。

实验矩阵：
  4 个视觉编码器 × 3 张地图 × N 次重复运行
  可选：额外 DDIM/CFG 消融

指标：
  - success_rate: 是否到达目标（二元）
  - steps: NoMaD 决策步数
  - path_distance: 机器人行走总路径长度 (m)
  - final_goal_dist: 最终位置到目标的欧氏距离 (m)
  - wall_time: 总运行时间 (s)

用法：
  python scripts/mujoco_encoder_benchmark.py --runs 3 --max-steps 300
  python scripts/mujoco_encoder_benchmark.py --runs 5 --maps easy hard --schedulers ddpm ddim
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# ────────────────────────────────────────────────────────────
# 模型配置：编码器名称 → (训练 config, checkpoint) 的映射
# ────────────────────────────────────────────────────────────
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "efficientnet_b0": {
        "config": "scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml",
        "checkpoint": "deployment/model_weights/nomad_efficientb0.pth",
        "label": "EfficientNet-B0 (baseline)",
    },
    "efficientnet_b0_suite": {
        "config": "scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0_suite.yaml",
        "checkpoint": "deployment/model_weights/nomad_efficientnet_b0_suite.pth",
        "label": "EfficientNet-B0 (suite fair control)",
    },
    "dinov2_small": {
        "config": "scripts/configs/vision_encoder/nomad_encoder_dinov2_small.yaml",
        "checkpoint": "deployment/model_weights/nomad_dinvo2_small.pth",
        "label": "DINOv2-Small",
    },
    "resnet50": {
        "config": "scripts/configs/vision_encoder/nomad_encoder_resnet50.yaml",
        "checkpoint": "deployment/model_weights/nomad_resnet50.pth",
        "label": "ResNet-50",
    },
    "convnext_tiny": {
        "config": "scripts/configs/vision_encoder/nomad_encoder_convnext_tiny.yaml",
        "checkpoint": "deployment/model_weights/nomad_convnext_tiny.pth",
        "label": "ConvNeXt-Tiny",
    },
}

# 地图信息：目标位置
MAP_GOALS = {
    "easy": np.array([5.0, 0.0]),
    "medium": np.array([7.0, 0.0]),
    "hard": np.array([8.0, 0.0]),
}


@dataclass
class RunResult:
    """单次运行结果记录"""
    encoder: str
    map_name: str
    scheduler: str
    ddim_steps: int
    cfg_weight: float
    run_index: int
    success: bool
    steps: int
    path_distance: float
    final_goal_dist: float
    wall_time: float
    exit_code: int


def parse_summary(summary_path: Path) -> dict:
    """解析 summary.txt 文件"""
    result = {"steps": 0, "path_distance": 0.0, "reached_goal": False}
    if not summary_path.exists():
        return result
    text = summary_path.read_text(encoding="utf-8")
    for line in text.strip().splitlines():
        if line.startswith("Steps:"):
            result["steps"] = int(line.split(":")[1].strip())
        elif line.startswith("Total distance:"):
            result["path_distance"] = float(line.split(":")[1].strip().replace(" m", ""))
        elif line.startswith("Reached goal:"):
            result["reached_goal"] = line.split(":")[1].strip().lower() == "true"
    return result


def parse_trajectory_final_position(traj_path: Path) -> Optional[np.ndarray]:
    """从 trajectory.txt 中读取最后一个位置"""
    if not traj_path.exists():
        return None
    lines = traj_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return None
    # 格式: x, y 或 [x, y] 或 x y
    last_line = lines[-1].strip().strip("[]")
    parts = [p.strip() for p in last_line.replace(",", " ").split()]
    if len(parts) >= 2:
        return np.array([float(parts[0]), float(parts[1])])
    return None


def find_latest_result_dir(base_dir: Path, prefix: str) -> Optional[Path]:
    """查找最新的结果目录"""
    if not base_dir.exists():
        return None
    candidates = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and prefix in d.name],
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_single_experiment(
    encoder_name: str,
    map_name: str,
    scheduler: str,
    ddim_steps: int,
    cfg_weight: float,
    max_steps: int,
    run_index: int,
    close_threshold: float,
    cpu_cores: str = "0-3",
) -> RunResult:
    """运行一次闭环导航实验"""
    model_info = MODEL_REGISTRY[encoder_name]
    config_path = str(REPO_ROOT / model_info["config"])
    checkpoint_path = str(REPO_ROOT / model_info["checkpoint"])

    # 使用 taskset 限制 CPU 亲和性，nice 降低优先级
    cmd = [
        "taskset", "-c", cpu_cores,
        "nice", "-n", "10",
        sys.executable,
        str(REPO_ROOT / "scripts" / "simulation" / "nomad_mujoco_lite3_state_machine.py"),
        "--mode", "navigate",
        "--map", map_name,
        "--no-gui",
        "--camera", "off",
        "--policy-config", config_path,
        "--policy-checkpoint", checkpoint_path,
        "--scheduler", scheduler,
        "--ddim-steps", str(ddim_steps),
        "--cfg-weight", str(cfg_weight),
        "--max-steps", str(max_steps),
        "--close-threshold", str(close_threshold),
    ]

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["QT_QPA_PLATFORM"] = "offscreen"

    print(f"  [{encoder_name}] {map_name} run={run_index+1} sched={scheduler} "
          f"ddim={ddim_steps} cfg={cfg_weight} ...", end="", flush=True)

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=600,
    )
    wall_time = time.time() - t0

    # 解析结果
    result_base = REPO_ROOT / "results" / "nomad_mujoco"
    result_dir = find_latest_result_dir(result_base, "lite3_state_machine_navigate")

    summary = {"steps": max_steps, "path_distance": 0.0, "reached_goal": False}
    final_pos = None
    if result_dir is not None:
        summary = parse_summary(result_dir / "summary.txt")
        final_pos = parse_trajectory_final_position(result_dir / "trajectory.txt")

    goal_pos = MAP_GOALS[map_name]
    if final_pos is not None:
        final_goal_dist = float(np.linalg.norm(final_pos - goal_pos))
    else:
        final_goal_dist = float(np.linalg.norm(goal_pos))

    status = "✓" if summary["reached_goal"] else "✗"
    print(f" {status} steps={summary['steps']} dist={summary['path_distance']:.1f}m "
          f"goal_dist={final_goal_dist:.2f}m t={wall_time:.1f}s")

    return RunResult(
        encoder=encoder_name,
        map_name=map_name,
        scheduler=scheduler,
        ddim_steps=ddim_steps,
        cfg_weight=cfg_weight,
        run_index=run_index,
        success=summary["reached_goal"],
        steps=summary["steps"],
        path_distance=summary["path_distance"],
        final_goal_dist=final_goal_dist,
        wall_time=wall_time,
        exit_code=proc.returncode,
    )


def aggregate_results(results: List[RunResult]) -> List[dict]:
    """按 (encoder, map, scheduler) 分组聚合统计结果"""
    groups: Dict[tuple, List[RunResult]] = {}
    for r in results:
        key = (r.encoder, r.map_name, r.scheduler, r.ddim_steps, r.cfg_weight)
        groups.setdefault(key, []).append(r)

    rows = []
    for key, runs in sorted(groups.items()):
        encoder, map_name, sched, ddim, cfg = key
        n = len(runs)
        successes = sum(1 for r in runs if r.success)
        steps_arr = np.array([r.steps for r in runs], dtype=float)
        dist_arr = np.array([r.path_distance for r in runs])
        goal_dist_arr = np.array([r.final_goal_dist for r in runs])
        time_arr = np.array([r.wall_time for r in runs])

        rows.append({
            "encoder": encoder,
            "encoder_label": MODEL_REGISTRY[encoder]["label"],
            "map": map_name,
            "scheduler": f"{sched}-{ddim}" if sched == "ddim" else sched,
            "cfg_weight": cfg,
            "runs": n,
            "success_rate": f"{successes}/{n}",
            "success_pct": round(100.0 * successes / n, 1),
            "steps_mean": round(float(np.mean(steps_arr)), 1),
            "steps_std": round(float(np.std(steps_arr)), 1),
            "path_dist_mean": round(float(np.mean(dist_arr)), 2),
            "path_dist_std": round(float(np.std(dist_arr)), 2),
            "goal_dist_mean": round(float(np.mean(goal_dist_arr)), 2),
            "goal_dist_std": round(float(np.std(goal_dist_arr)), 2),
            "wall_time_mean": round(float(np.mean(time_arr)), 1),
        })
    return rows


def render_markdown_report(rows: List[dict], all_results: List[RunResult], output_dir: Path) -> str:
    """生成 Markdown 格式的实验报告"""
    lines = [
        "# MuJoCo 闭环导航基准实验：多视觉编码器对比",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 实验设计",
        "",
        "### 1.1 实验目标",
        "",
        "在 MuJoCo 仿真环境中，使用统一的闭环导航流程（NoMaD + Lite3 四足机器人），",
        "对比 4 种不同视觉编码器训练的策略在导航任务中的表现差异。",
        "",
        "### 1.2 编码器配置",
        "",
        "| 编码器 | 类型 | 参数量级 | 图像尺寸 | 训练配置 |",
        "|---|---|---|---|---|",
        "| EfficientNet-B0 | CNN (baseline) | 5.3M | 96×96 | bs=32, lr=1e-4 |",
        "| DINOv2-Small | ViT (自监督预训练) | 22M | 98×98 | bs=32, lr=5e-5, frozen backbone |",
        "| ResNet-50 | CNN (ImageNet 预训练) | 25.6M | 96×96 | bs=32, lr=5e-5, frozen backbone |",
        "| ConvNeXt-Tiny | 现代 CNN (ImageNet-12k 预训练) | 28.6M | 96×96 | bs=32, lr=5e-5, frozen backbone |",
        "",
        "### 1.3 实验场景",
        "",
        "| 地图 | 起点 | 终点 | 直线距离 | 障碍物 | 难度 |",
        "|---|---|---|---|---|---|",
        "| easy | (0,0) | (5,0) | 5.0m | 0 | 无障碍直行 |",
        "| medium | (0,0) | (7,0) | 7.0m | 1 | 单障碍绕行 |",
        "| hard | (0,0) | (8,0) | 8.0m | 2 | 双障碍 S 形绕行 |",
        "",
        "### 1.4 评价指标解释",
        "",
        "| 指标 | 含义 | 好的方向 |",
        "|---|---|---|",
        "| **Success Rate (成功率)** | 机器人最终位置距目标 < 0.5m 的比例 | ↑ 越高越好 |",
        "| **Steps (步数)** | NoMaD 决策循环次数；越少说明导航越高效 | ↓ 越少越好（成功前提下） |",
        "| **Path Distance (路径距离)** | 机器人实际行走的总路径长度 (m) | ↓ 越短越好（接近直线距离） |",
        "| **Final Goal Dist (终点到目标距离)** | 导航结束时机器人与目标的欧氏距离 (m) | ↓ 越小越好 |",
        "| **Wall Time (实际耗时)** | 单次运行的总墙钟时间 (s)，反映推理效率 | ↓ 越短越好 |",
        "",
        "## 2. 总体结果",
        "",
        "### 2.1 按编码器汇总（所有地图平均）",
        "",
    ]

    # 按编码器汇总
    encoder_summary = {}
    for r in all_results:
        encoder_summary.setdefault(r.encoder, []).append(r)

    lines.append("| 编码器 | 总成功率 | 平均步数 | 平均路径距离 | 平均终点距离 | 平均耗时 |")
    lines.append("|---|---|---|---|---|---|")
    for enc_name in MODEL_REGISTRY:
        if enc_name not in encoder_summary:
            continue
        runs = encoder_summary[enc_name]
        n = len(runs)
        succ = sum(1 for r in runs if r.success)
        steps_m = np.mean([r.steps for r in runs])
        dist_m = np.mean([r.path_distance for r in runs])
        gdist_m = np.mean([r.final_goal_dist for r in runs])
        time_m = np.mean([r.wall_time for r in runs])
        label = MODEL_REGISTRY[enc_name]["label"]
        lines.append(
            f"| {label} | {succ}/{n} ({100*succ/n:.0f}%) | {steps_m:.1f} | "
            f"{dist_m:.2f} m | {gdist_m:.2f} m | {time_m:.1f} s |"
        )

    lines.extend(["", "### 2.2 详细结果（按编码器 × 地图）", ""])
    lines.append("| 编码器 | 地图 | 调度器 | 成功率 | 步数 (mean±std) | 路径距离 (mean±std) | 终点距离 (mean±std) | 耗时 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['encoder_label']} | {row['map']} | {row['scheduler']} | "
            f"{row['success_rate']} ({row['success_pct']}%) | "
            f"{row['steps_mean']}±{row['steps_std']} | "
            f"{row['path_dist_mean']}±{row['path_dist_std']} m | "
            f"{row['goal_dist_mean']}±{row['goal_dist_std']} m | "
            f"{row['wall_time_mean']} s |"
        )

    lines.extend([
        "",
        "## 3. 结果分析",
        "",
        "### 3.1 指标解读",
        "",
        "- **成功率 (Success Rate)**：最核心的指标。100% 意味着该编码器在该地图上所有运行都能到达目标，",
        "  0% 意味着该编码器在该难度下无法完成导航。成功判定：机器人最终位置距目标 < 0.5m。",
        "",
        "- **步数 (Steps)**：在成功情况下，步数越少代表导航路径越优。如果失败（达到 max_steps 上限），",
        "  步数等于 max_steps，此时步数不具可比性。",
        "",
        "- **路径距离 (Path Distance)**：实际行走路径的长度。与直线距离之比（路径效率比 = 路径/直线）",
        "  越接近 1 越好。对于有障碍的地图，最优路径会略大于直线距离。",
        "",
        "- **终点到目标距离 (Final Goal Dist)**：成功时应 < 0.5m（到达阈值）；失败时数值越大，",
        "  说明离目标越远，导航偏离越严重。",
        "",
        "- **实际耗时 (Wall Time)**：包含 MuJoCo 纯仿真时间 + NoMaD 推理时间 + IO 开销。",
        "  较大的编码器（如 ResNet-50, ConvNeXt-Tiny）可能推理更慢。",
        "",
    ])

    # 动态分析
    lines.append("### 3.2 对比分析")
    lines.append("")

    # 找最优编码器
    if encoder_summary:
        best_encoder = max(encoder_summary.keys(),
                          key=lambda e: sum(1 for r in encoder_summary[e] if r.success) / len(encoder_summary[e]))
        best_rate = sum(1 for r in encoder_summary[best_encoder] if r.success) / len(encoder_summary[best_encoder])
        worst_encoder = min(encoder_summary.keys(),
                           key=lambda e: sum(1 for r in encoder_summary[e] if r.success) / len(encoder_summary[e]))
        worst_rate = sum(1 for r in encoder_summary[worst_encoder] if r.success) / len(encoder_summary[worst_encoder])

        lines.append(f"- **最佳编码器**：{MODEL_REGISTRY[best_encoder]['label']}，"
                     f"总成功率 {best_rate*100:.0f}%")
        lines.append(f"- **最差编码器**：{MODEL_REGISTRY[worst_encoder]['label']}，"
                     f"总成功率 {worst_rate*100:.0f}%")
        lines.append("")

        # 按地图分析
        for map_name in sorted(set(r.map_name for r in all_results)):
            map_runs = [r for r in all_results if r.map_name == map_name]
            map_enc_rates = {}
            for enc in MODEL_REGISTRY:
                enc_runs = [r for r in map_runs if r.encoder == enc]
                if enc_runs:
                    rate = sum(1 for r in enc_runs if r.success) / len(enc_runs)
                    map_enc_rates[enc] = rate
            if map_enc_rates:
                best_map_enc = max(map_enc_rates, key=map_enc_rates.get)
                lines.append(f"- **{map_name} 地图最佳**：{MODEL_REGISTRY[best_map_enc]['label']} "
                             f"({map_enc_rates[best_map_enc]*100:.0f}% 成功)")

    lines.extend([
        "",
        "### 3.3 关键发现",
        "",
        "（以下分析基于自动生成的定量结果，更深入的定性分析请结合轨迹可视化）",
        "",
        "1. **Baseline vs 预训练特征**：EfficientNet-B0 作为 baseline 从头训练，"
        "   与冻结 backbone 的预训练模型相比，各自有何优劣；",
        "",
        "2. **CNN vs ViT**：DINOv2 (ViT 架构) vs ConvNeXt/ResNet (CNN 架构) "
        "   在导航鲁棒性和推理效率上的差异；",
        "",
        "3. **难度梯度**：随着障碍物增加（easy→hard），不同编码器的性能退化程度不同。",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo 多编码器闭环导航基准实验")
    parser.add_argument("--runs", type=int, default=3, help="每个配置的重复运行次数")
    parser.add_argument("--max-steps", type=int, default=300, help="每次运行的最大 NoMaD 步数")
    parser.add_argument("--close-threshold", type=float, default=3.0, help="NoMaD 节点到达阈值")
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=["efficientnet_b0", "dinov2_small", "convnext_tiny", "resnet50"],
        choices=list(MODEL_REGISTRY.keys()),
        help="要测试的编码器列表",
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=["easy", "medium", "hard"],
        help="要测试的地图列表",
    )
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=["ddim"],
        choices=["ddpm", "ddim"],
        help="要测试的扩散调度器",
    )
    parser.add_argument("--ddim-steps", type=int, default=2, help="DDIM 采样步数")
    parser.add_argument(
        "--cfg-weights",
        nargs="+",
        type=float,
        default=[0.0],
        help="要测试的 Classifier-Free Guidance 权重列表 (0=禁用)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="结果输出目录")
    parser.add_argument("--cpu-cores", type=str, default="0-3",
                        help="限制子进程使用的 CPU 核心范围，如 '0-3' 表示仅用 4 个核心")
    args = parser.parse_args()

    # 确保 topomap 已生成
    for map_name in args.maps:
        topomap_dir = REPO_ROOT / "topomaps" / map_name
        if not topomap_dir.exists():
            print(f"[WARN] Topomap not found for {map_name}, generating...")
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "simulation" / "nomad_mujoco_lite3_state_machine.py"),
                "--mode", "generate-topomap", "--map", map_name, "--no-gui",
            ]
            env = os.environ.copy()
            env.update({"MUJOCO_GL": "egl", "QT_QPA_PLATFORM": "offscreen"})
            subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), check=True)

    # 实验矩阵
    total_runs = len(args.encoders) * len(args.maps) * len(args.schedulers) * len(args.cfg_weights) * args.runs
    print(f"\n{'='*60}")
    print(f"MuJoCo 闭环导航基准实验")
    print(f"编码器: {args.encoders}")
    print(f"地图: {args.maps}")
    print(f"调度器: {args.schedulers} (DDIM steps={args.ddim_steps})")
    print(f"CFG 权重: {args.cfg_weights}")
    print(f"每配置运行次数: {args.runs}")
    print(f"总运行次数: {total_runs}")
    print(f"max_steps: {args.max_steps}")
    print(f"{'='*60}\n")

    all_results: List[RunResult] = []
    run_counter = 0

    for encoder in args.encoders:
        print(f"\n--- {MODEL_REGISTRY[encoder]['label']} ---")
        for map_name in args.maps:
            for scheduler in args.schedulers:
                ddim_steps = args.ddim_steps if scheduler == "ddim" else 10
                for cfg_weight in args.cfg_weights:
                    for run_idx in range(args.runs):
                        run_counter += 1
                        print(f"[{run_counter}/{total_runs}]", end="")
                        result = run_single_experiment(
                            encoder_name=encoder,
                            map_name=map_name,
                            scheduler=scheduler,
                            ddim_steps=ddim_steps,
                            cfg_weight=cfg_weight,
                            max_steps=args.max_steps,
                            run_index=run_idx,
                            close_threshold=args.close_threshold,
                            cpu_cores=args.cpu_cores,
                        )
                        all_results.append(result)

    # 聚合与输出
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = REPO_ROOT / "results" / "benchmark" / f"{tag}_encoder_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始数据
    raw_data = [asdict(r) for r in all_results]
    (output_dir / "raw_results.json").write_text(
        json.dumps(raw_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 聚合统计
    agg_rows = aggregate_results(all_results)
    (output_dir / "aggregated_results.json").write_text(
        json.dumps(agg_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown 报告
    report = render_markdown_report(agg_rows, all_results, output_dir)
    (output_dir / "benchmark_report.md").write_text(report, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"实验完成！结果保存到: {output_dir}")
    print(f"{'='*60}")

    # 打印摘要表格
    print("\n" + "─" * 80)
    print(f"{'编码器':<20} {'地图':<8} {'成功率':<10} {'步数':<15} {'路径距离':<15} {'终点距离':<15}")
    print("─" * 80)
    for row in agg_rows:
        print(
            f"{row['encoder_label']:<20} {row['map']:<8} "
            f"{row['success_rate']:<10} "
            f"{row['steps_mean']:>6.1f}±{row['steps_std']:<6.1f} "
            f"{row['path_dist_mean']:>6.2f}±{row['path_dist_std']:<6.2f} "
            f"{row['goal_dist_mean']:>6.2f}±{row['goal_dist_std']:<6.2f}"
        )
    print("─" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
