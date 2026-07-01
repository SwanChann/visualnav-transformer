#!/usr/bin/env python3
"""
Quick TTS experiment for NoMaD diffusion inference.

TTS here means test-time scaling for diffusion policy inference:
we sample a larger candidate set at inference time and use a verifier
to keep the best trajectories.
"""

from __future__ import annotations

# Allow direct execution from any scripts/<category>/ path.
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve()
while SCRIPTS_ROOT.name != "scripts" and SCRIPTS_ROOT.parent != SCRIPTS_ROOT:
    SCRIPTS_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from tooling.project_paths import REPO_ROOT, add_repo_paths

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_ROOT = str(REPO_ROOT)
add_repo_paths()

from shared.nomad_eval_common import build_eval_cases, load_case_tensors, parse_suite_offsets, set_seed
from shared.nomad_inference import NoMaDInferenceModule, NoMaDInferenceSpec


QUICK_CONFIGS = [
    ("ddpm10_cfg0_tts0", "ddpm", 10, 0.0, 0),
    ("ddim2_cfg0_tts0", "ddim", 2, 0.0, 0),
    ("ddim2_cfg0_tts16", "ddim", 2, 0.0, 16),
    ("ddim2_cfg05_tts0", "ddim", 2, 0.5, 0),
    ("ddim2_cfg05_tts16", "ddim", 2, 0.5, 16),
    ("ddim5_cfg10_tts32", "ddim", 5, 1.0, 32),
]


def evaluate_config(
    inference: NoMaDInferenceModule,
    obs_img,
    goal_img,
    scheduler_kind: str,
    num_steps: int,
    cfg_weight: float,
    tts_budget: int,
    tts_topk: int,
    verifier: str,
    num_runs: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    times_ms = []
    sample_batches = []

    inference.scheduler_kind = scheduler_kind
    inference.ddim_steps = num_steps
    inference.cfg_weight = cfg_weight
    inference.tts_enabled = tts_budget > 0
    inference.tts_budget = max(tts_budget, inference.num_samples)
    inference.tts_topk = min(max(tts_topk, 1), inference.tts_budget)
    inference.tts_verifier = verifier

    uncond = None
    if cfg_weight != 0.0:
        uncond = inference.encode_condition(obs_img, goal_img, goal_mask_value=1)

    for run_idx in range(num_runs):
        set_seed(base_seed + run_idx)
        cond = inference.encode_condition(obs_img, goal_img, goal_mask_value=0)
        start_time = time.perf_counter()
        samples = inference.sample_actions(
            cond,
            uncond_condition=uncond,
            num_samples=inference.num_samples,
        )
        times_ms.append((time.perf_counter() - start_time) * 1000.0)
        sample_batches.append(samples)

    return np.asarray(times_ms, dtype=np.float64), np.concatenate(sample_batches, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick TTS experiment for NoMaD")
    parser.add_argument("--dataset-root", action="append", default=None)
    parser.add_argument("--policy-config", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--suite-offsets", default="short:8")
    parser.add_argument("--cases-per-suite", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--tts-topk", type=int, default=4)
    parser.add_argument("--tts-verifier", choices=["heuristic", "forward", "conservative"], default="heuristic")
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    inference = NoMaDInferenceModule(
        NoMaDInferenceSpec(
            policy_config=args.policy_config,
            policy_checkpoint=args.weights,
            num_samples=args.num_samples,
        )
    )
    suite_offsets = parse_suite_offsets(args.suite_offsets)
    cases = build_eval_cases(
        dataset_roots=args.dataset_root,
        suite_offsets=suite_offsets,
        cases_per_suite=args.cases_per_suite,
        seed=args.seed,
        context_size=inference.context_size,
    )
    case = cases[0]
    obs_img, goal_img = load_case_tensors(case, inference.device)

    results = {}
    baseline_key = QUICK_CONFIGS[0][0]
    baseline_mean = None

    for config_idx, (name, scheduler_kind, num_steps, cfg_weight, tts_budget) in enumerate(QUICK_CONFIGS):
        times_ms, samples = evaluate_config(
            inference=inference,
            obs_img=obs_img,
            goal_img=goal_img,
            scheduler_kind=scheduler_kind,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            tts_budget=tts_budget,
            tts_topk=args.tts_topk,
            verifier=args.tts_verifier,
            num_runs=args.num_runs,
            base_seed=args.seed + config_idx * 100,
        )
        mean_traj = samples.mean(axis=0)
        if baseline_mean is None:
            baseline_mean = mean_traj.copy()
        mse = float(np.mean((mean_traj - baseline_mean) ** 2))
        verifier_scores = inference.score_action_candidates(samples, verifier=args.tts_verifier)
        endpoint = mean_traj[-1]
        results[name] = {
            "latency_ms": float(times_ms.mean()),
            "latency_std_ms": float(times_ms.std(ddof=1)) if len(times_ms) > 1 else 0.0,
            "mse_vs_base": mse,
            "forward_progress": float(endpoint[0]),
            "lateral_abs": float(abs(endpoint[1])),
            "verifier_score": float(verifier_scores.mean()),
            "mean_traj": mean_traj,
            "scheduler_kind": scheduler_kind,
            "num_steps": num_steps,
            "cfg_weight": cfg_weight,
            "tts_budget": tts_budget,
        }

    out_dir = os.path.join(PROJECT_ROOT, "results", "day4")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    config_names = list(results.keys())
    axes[0].bar(
        range(len(config_names)),
        [results[name]["latency_ms"] for name in config_names],
        color=["#e74c3c" if results[name]["tts_budget"] == 0 else "#2ecc71" for name in config_names],
    )
    axes[0].set_title("Latency")
    axes[0].set_ylabel("ms")
    axes[0].set_xticks(range(len(config_names)))
    axes[0].set_xticklabels(config_names, rotation=30, ha="right", fontsize=8)

    axes[1].bar(
        range(len(config_names)),
        [results[name]["forward_progress"] for name in config_names],
        color="#3498db",
    )
    axes[1].set_title("Forward Progress")
    axes[1].set_xticks(range(len(config_names)))
    axes[1].set_xticklabels(config_names, rotation=30, ha="right", fontsize=8)

    for name in config_names:
        traj = results[name]["mean_traj"]
        axes[2].plot(traj[:, 1], traj[:, 0], marker="o", label=name, alpha=0.75)
    axes[2].set_title("Mean Trajectory")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("x")
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{RUN_TAG}_tts_quick_experiment.png"), dpi=160)
    plt.close(fig)

    txt_path = os.path.join(out_dir, f"{RUN_TAG}_tts_quick_experiment.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("Quick TTS experiment for NoMaD\n")
        handle.write(f"Case: {case.traj_name} | suite={case.suite_name}\n")
        handle.write(f"Verifier: {args.tts_verifier}, topk={args.tts_topk}\n\n")
        handle.write(
            f"{'Config':<22} {'Latency(ms)':<14} {'MSE':<12} {'Fwd':<10} {'LatAbs':<10} {'Score':<10}\n"
        )
        handle.write("-" * 84 + "\n")
        for name in config_names:
            row = results[name]
            handle.write(
                f"{name:<22} {row['latency_ms']:<14.2f} {row['mse_vs_base']:<12.6f} "
                f"{row['forward_progress']:<10.4f} {row['lateral_abs']:<10.4f} {row['verifier_score']:<10.4f}\n"
            )
        handle.write("\n")
        handle.write("# 说明:\n")
        handle.write("# TTS = test-time scaling, 通过扩大量化候选轨迹并用 verifier 选择更优轨迹。\n")
        handle.write("# tts_budget=0 表示关闭 TTS，仅保留原始扩散采样。\n")
    print(f"Saved quick TTS experiment to: {txt_path}")


if __name__ == "__main__":
    main()
