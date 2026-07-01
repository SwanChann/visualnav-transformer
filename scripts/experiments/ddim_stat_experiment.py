#!/usr/bin/env python3
"""Statistical DDIM benchmark for NoMaD."""

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
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from shared.nomad_eval_common import (
    aggregate_case_records,
    build_eval_cases,
    compute_sample_metrics,
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


CONFIGS = [
    ("ddpm_10", "ddpm", 10),
    ("ddim_10", "ddim", 10),
    ("ddim_5", "ddim", 5),
    ("ddim_3", "ddim", 3),
    ("ddim_2", "ddim", 2),
    ("ddim_1", "ddim", 1),
]

METRIC_FIELDS = [
    "latency_ms",
    "latency_std_ms",
    "mse_vs_baseline",
    "diversity",
    "path_length_mean",
    "endpoint_norm_mean",
    "forward_progress_mean",
    "lateral_abs_mean",
    "smoothness_mean",
]


def evaluate_config(
    model,
    cond: torch.Tensor,
    device: torch.device,
    scheduler_kind: str,
    num_steps: int,
    num_runs: int,
    num_samples: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a configuration repeatedly and return times plus samples."""
    times_ms: List[float] = []
    sample_batches: List[np.ndarray] = []
    for run_index in range(num_runs):
        set_seed(base_seed + run_index)
        start_time = time.perf_counter()
        samples = run_diffusion_sampling(
            model=model,
            cond=cond,
            device=device,
            scheduler_kind=scheduler_kind,
            num_steps=num_steps,
            num_samples=num_samples,
        )
        times_ms.append((time.perf_counter() - start_time) * 1000.0)
        sample_batches.append(samples)
    return np.asarray(times_ms, dtype=np.float64), np.concatenate(sample_batches, axis=0)


def build_case_record(
    case,
    config_name: str,
    scheduler_kind: str,
    num_steps: int,
    times_ms: np.ndarray,
    samples: np.ndarray,
    baseline_mean: np.ndarray,
) -> Dict[str, object]:
    """Create a case-level record."""
    metrics = compute_sample_metrics(samples)
    record: Dict[str, object] = {
        "dataset_name": case.dataset_name,
        "suite_name": case.suite_name,
        "traj_name": case.traj_name,
        "config_name": config_name,
        "scheduler_kind": scheduler_kind,
        "num_steps": num_steps,
        "latency_ms": float(times_ms.mean()),
        "latency_std_ms": float(times_ms.std(ddof=1)) if len(times_ms) > 1 else 0.0,
        "mse_vs_baseline": float(np.mean((samples.mean(axis=0) - baseline_mean) ** 2)),
    }
    record.update(metrics)
    return record


def plot_overall_tradeoff(overall_rows: List[Dict[str, object]], out_dir: Path) -> None:
    """Plot overall latency-quality tradeoff."""
    labels = [str(row["config_name"]) for row in overall_rows]
    latency = [float(row["latency_ms_mean"]) for row in overall_rows]
    latency_ci = [float(row["latency_ms_ci95"]) for row in overall_rows]
    quality = [float(row["mse_vs_baseline_mean"]) for row in overall_rows]
    quality_ci = [float(row["mse_vs_baseline_ci95"]) for row in overall_rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].errorbar(labels, latency, yerr=latency_ci, fmt="o", capsize=4)
    axes[0].set_title("Latency Across Configurations")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(labels, quality, yerr=quality_ci, fmt="o", capsize=4, color="#d35400")
    axes[1].set_title("MSE vs DDPM-10 Baseline")
    axes[1].set_ylabel("MSE")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "ddim_stat_overall.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical DDIM benchmark")
    parser.add_argument("--dataset-root", action="append", default=None, help="Repeatable dataset root")
    parser.add_argument("--weights", default=None, help="Optional NoMaD checkpoint path")
    parser.add_argument("--cases-per-suite", type=int, default=8, help="Cases sampled per suite")
    parser.add_argument("--suite-offsets", default="short:8,medium:16,long:24", help="Suite spec")
    parser.add_argument("--num-runs", type=int, default=8, help="Repeated runs per case and config")
    parser.add_argument("--num-samples", type=int, default=8, help="Diffusion samples per run")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
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

    out_dir = make_output_dir("day2", "ddim_stat_experiment")
    save_cases_json(out_dir / "eval_cases.json", cases)

    records: List[Dict[str, object]] = []
    for case_index, case in enumerate(cases):
        obs_img, goal_img = load_case_tensors(case, device)
        cond = encode_navigation_condition(model, obs_img, goal_img, device)

        baseline_times, baseline_samples = evaluate_config(
            model=model,
            cond=cond,
            device=device,
            scheduler_kind="ddpm",
            num_steps=10,
            num_runs=args.num_runs,
            num_samples=args.num_samples,
            base_seed=args.seed + case_index * 1000,
        )
        baseline_mean = baseline_samples.mean(axis=0)
        records.append(
            build_case_record(
                case=case,
                config_name="ddpm_10",
                scheduler_kind="ddpm",
                num_steps=10,
                times_ms=baseline_times,
                samples=baseline_samples,
                baseline_mean=baseline_mean,
            )
        )

        for config_index, (config_name, scheduler_kind, num_steps) in enumerate(CONFIGS[1:], start=1):
            times_ms, samples = evaluate_config(
                model=model,
                cond=cond,
                device=device,
                scheduler_kind=scheduler_kind,
                num_steps=num_steps,
                num_runs=args.num_runs,
                num_samples=args.num_samples,
                base_seed=args.seed + case_index * 1000 + config_index * 100,
            )
            records.append(
                build_case_record(
                    case=case,
                    config_name=config_name,
                    scheduler_kind=scheduler_kind,
                    num_steps=num_steps,
                    times_ms=times_ms,
                    samples=samples,
                    baseline_mean=baseline_mean,
                )
            )

    overall_rows = aggregate_case_records(records, ["config_name"], METRIC_FIELDS)
    baseline_latency = next(
        float(row["latency_ms_mean"]) for row in overall_rows if row["config_name"] == "ddpm_10"
    )
    for row in overall_rows:
        row["speedup_vs_ddpm10"] = baseline_latency / float(row["latency_ms_mean"])

    suite_rows = aggregate_case_records(
        records,
        ["dataset_name", "suite_name", "config_name"],
        METRIC_FIELDS,
    )

    save_rows_csv(out_dir / "ddim_case_records.csv", records)
    save_rows_csv(out_dir / "ddim_overall_summary.csv", overall_rows)
    save_rows_csv(out_dir / "ddim_suite_summary.csv", suite_rows)
    save_rows_json(out_dir / "ddim_overall_summary.json", overall_rows)
    save_markdown_table(out_dir / "ddim_overall_summary.md", overall_rows)
    save_markdown_table(out_dir / "ddim_suite_summary.md", suite_rows)
    plot_overall_tradeoff(overall_rows, out_dir)

    readme_lines = [
        f"Weights: {weights_path}",
        f"Device: {device}",
        f"Cases: {len(cases)}",
        f"Num runs per config: {args.num_runs}",
        f"Num samples per run: {args.num_samples}",
        f"Suites: {suite_offsets}",
    ]
    (out_dir / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"Saved DDIM statistical benchmark to: {out_dir}")


if __name__ == "__main__":
    main()
