#!/usr/bin/env python3
"""Statistical CFG benchmark for NoMaD."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from nomad_eval_common import (
    aggregate_case_records,
    build_eval_cases,
    compute_sample_metrics,
    encode_cfg_conditions,
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


GUIDANCE_VALUES = [-1.0, 0.0, 0.5, 1.0, 2.0, 4.0]
METRIC_FIELDS = [
    "latency_ms",
    "latency_std_ms",
    "shift_vs_w0",
    "diversity",
    "path_length_mean",
    "endpoint_norm_mean",
    "forward_progress_mean",
    "lateral_abs_mean",
    "smoothness_mean",
]


def evaluate_guidance(
    model,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    device: torch.device,
    guidance_scale: float,
    num_steps: int,
    num_runs: int,
    num_samples: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run repeated CFG experiments for one guidance scale."""
    times_ms: List[float] = []
    sample_batches: List[np.ndarray] = []
    for run_index in range(num_runs):
        set_seed(base_seed + run_index)
        start_time = time.perf_counter()
        if guidance_scale == -1.0:
            samples = run_diffusion_sampling(
                model=model,
                cond=uncond,
                uncond=None,
                device=device,
                scheduler_kind="ddpm",
                num_steps=num_steps,
                num_samples=num_samples,
            )
        elif guidance_scale == 0.0:
            samples = run_diffusion_sampling(
                model=model,
                cond=cond,
                uncond=None,
                device=device,
                scheduler_kind="ddpm",
                num_steps=num_steps,
                num_samples=num_samples,
            )
        else:
            samples = run_diffusion_sampling(
                model=model,
                cond=cond,
                uncond=uncond,
                device=device,
                scheduler_kind="ddpm",
                num_steps=num_steps,
                num_samples=num_samples,
                guidance_scale=guidance_scale,
            )
        times_ms.append((time.perf_counter() - start_time) * 1000.0)
        sample_batches.append(samples)
    return np.asarray(times_ms, dtype=np.float64), np.concatenate(sample_batches, axis=0)


def build_case_record(
    case,
    guidance_scale: float,
    times_ms: np.ndarray,
    samples: np.ndarray,
    baseline_mean: np.ndarray,
) -> Dict[str, object]:
    """Create a case-level record for CFG evaluation."""
    metrics = compute_sample_metrics(samples)
    record: Dict[str, object] = {
        "dataset_name": case.dataset_name,
        "suite_name": case.suite_name,
        "traj_name": case.traj_name,
        "guidance_scale": guidance_scale,
        "latency_ms": float(times_ms.mean()),
        "latency_std_ms": float(times_ms.std(ddof=1)) if len(times_ms) > 1 else 0.0,
        "shift_vs_w0": float(np.mean((samples.mean(axis=0) - baseline_mean) ** 2)),
    }
    record.update(metrics)
    return record


def plot_cfg_trends(overall_rows: List[Dict[str, object]], out_dir: Path) -> None:
    """Plot the overall CFG trends."""
    scales = [float(row["guidance_scale"]) for row in overall_rows]
    diversity = [float(row["diversity_mean"]) for row in overall_rows]
    forward_progress = [float(row["forward_progress_mean_mean"]) for row in overall_rows]
    latency = [float(row["latency_ms_mean"]) for row in overall_rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(scales, diversity, marker="o")
    axes[0].set_title("Diversity vs Guidance")
    axes[0].set_xlabel("Guidance Scale")
    axes[0].set_ylabel("Diversity")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(scales, forward_progress, marker="o", color="#c0392b")
    axes[1].set_title("Forward Progress vs Guidance")
    axes[1].set_xlabel("Guidance Scale")
    axes[1].set_ylabel("Forward Progress")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(scales, latency, marker="o", color="#27ae60")
    axes[2].set_title("Latency vs Guidance")
    axes[2].set_xlabel("Guidance Scale")
    axes[2].set_ylabel("Latency (ms)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "cfg_stat_trends.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical CFG benchmark")
    parser.add_argument("--dataset-root", action="append", default=None, help="Repeatable dataset root")
    parser.add_argument("--weights", default=None, help="Optional NoMaD checkpoint path")
    parser.add_argument("--cases-per-suite", type=int, default=8, help="Cases sampled per suite")
    parser.add_argument("--suite-offsets", default="short:8,medium:16,long:24", help="Suite spec")
    parser.add_argument("--num-runs", type=int, default=8, help="Repeated runs per case and guidance scale")
    parser.add_argument("--num-samples", type=int, default=8, help="Diffusion samples per run")
    parser.add_argument("--num-steps", type=int, default=10, help="Diffusion denoising steps")
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, weights_path = load_nomad_model(device=device, weights_path=args.weights)
    suite_offsets = parse_suite_offsets(args.suite_offsets)
    cases = build_eval_cases(
        dataset_roots=args.dataset_root,
        suite_offsets=suite_offsets,
        cases_per_suite=args.cases_per_suite,
        seed=args.seed,
    )

    out_dir = make_output_dir("day3", "cfg_stat_experiment")
    save_cases_json(out_dir / "eval_cases.json", cases)

    records: List[Dict[str, object]] = []
    for case_index, case in enumerate(cases):
        obs_img, goal_img = load_case_tensors(case, device)
        cond, uncond = encode_cfg_conditions(model, obs_img, goal_img, device)

        baseline_times, baseline_samples = evaluate_guidance(
            model=model,
            cond=cond,
            uncond=uncond,
            device=device,
            guidance_scale=0.0,
            num_steps=args.num_steps,
            num_runs=args.num_runs,
            num_samples=args.num_samples,
            base_seed=args.seed + case_index * 1000,
        )
        baseline_mean = baseline_samples.mean(axis=0)
        records.append(
            build_case_record(
                case=case,
                guidance_scale=0.0,
                times_ms=baseline_times,
                samples=baseline_samples,
                baseline_mean=baseline_mean,
            )
        )

        for guide_index, guidance_scale in enumerate(GUIDANCE_VALUES):
            if guidance_scale == 0.0:
                continue
            times_ms, samples = evaluate_guidance(
                model=model,
                cond=cond,
                uncond=uncond,
                device=device,
                guidance_scale=guidance_scale,
                num_steps=args.num_steps,
                num_runs=args.num_runs,
                num_samples=args.num_samples,
                base_seed=args.seed + case_index * 1000 + guide_index * 100,
            )
            records.append(
                build_case_record(
                    case=case,
                    guidance_scale=guidance_scale,
                    times_ms=times_ms,
                    samples=samples,
                    baseline_mean=baseline_mean,
                )
            )

    overall_rows = aggregate_case_records(records, ["guidance_scale"], METRIC_FIELDS)
    suite_rows = aggregate_case_records(
        records,
        ["dataset_name", "suite_name", "guidance_scale"],
        METRIC_FIELDS,
    )

    save_rows_csv(out_dir / "cfg_case_records.csv", records)
    save_rows_csv(out_dir / "cfg_overall_summary.csv", overall_rows)
    save_rows_csv(out_dir / "cfg_suite_summary.csv", suite_rows)
    save_rows_json(out_dir / "cfg_overall_summary.json", overall_rows)
    save_markdown_table(out_dir / "cfg_overall_summary.md", overall_rows)
    save_markdown_table(out_dir / "cfg_suite_summary.md", suite_rows)
    plot_cfg_trends(overall_rows, out_dir)

    readme_lines = [
        f"Weights: {weights_path}",
        f"Device: {device}",
        f"Cases: {len(cases)}",
        f"Num runs per guidance scale: {args.num_runs}",
        f"Num samples per run: {args.num_samples}",
        f"Num steps: {args.num_steps}",
        f"Suites: {suite_offsets}",
    ]
    (out_dir / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"Saved CFG statistical benchmark to: {out_dir}")


if __name__ == "__main__":
    main()
