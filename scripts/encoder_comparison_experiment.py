#!/usr/bin/env python3
"""Compare NoMaD checkpoints trained with different visual encoders."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nomad_eval_common import (
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
from nomad_vint_backbone_suite import build_backbone_nomad_model, get_backbone_names


METRIC_FIELDS = [
    "latency_ms",
    "dist_pred_mean",
    "diversity",
    "path_length_mean",
    "endpoint_norm_mean",
    "forward_progress_mean",
    "lateral_abs_mean",
    "smoothness_mean",
]


def parse_model_specs(raw_specs: List[str] | None) -> List[Tuple[str, str]]:
    """Parse repeated model specs in the form backbone=checkpoint_path."""
    if not raw_specs:
        return []
    pairs: List[Tuple[str, str]] = []
    for raw_spec in raw_specs:
        name, path = raw_spec.split("=", 1)
        pairs.append((name.strip(), path.strip()))
    return pairs


def load_model_for_spec(
    spec_name: str,
    checkpoint_path: str,
    device: torch.device,
):
    """Load one model for a checkpoint spec."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if spec_name == "efficientnet_b0":
        model, _ = load_nomad_model(device=device, weights_path=str(checkpoint))
        return model

    if spec_name not in get_backbone_names():
        raise KeyError(f"Unsupported model spec: {spec_name}")

    config = {
        "context_size": 3,
        "encoding_size": 256,
        "mha_num_attention_heads": 4,
        "mha_num_attention_layers": 4,
        "mha_ff_dim_factor": 4,
        "down_dims": [64, 128, 256],
        "cond_predict_scale": False,
    }
    model = build_backbone_nomad_model(
        config=config,
        backbone_name=spec_name,
        freeze_backbone=False,
        pretrained_backbone=False,
    )
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def evaluate_one_model(
    model_name: str,
    model,
    cases,
    device: torch.device,
    num_runs: int,
    num_samples: int,
    scheduler_kind: str,
    num_steps: int,
    seed: int,
) -> List[Dict[str, object]]:
    """Evaluate one model across all cases."""
    records: List[Dict[str, object]] = []
    for case_index, case in enumerate(cases):
        obs_img, goal_img = load_case_tensors(case, device)
        cond = encode_navigation_condition(model, obs_img, goal_img, device)

        latency_values: List[float] = []
        sample_batches: List[np.ndarray] = []

        with torch.no_grad():
            dist_pred = model("dist_pred_net", obsgoal_cond=cond)

        for run_index in range(num_runs):
            set_seed(seed + case_index * 1000 + run_index)
            start_time = time.perf_counter()
            samples = run_diffusion_sampling(
                model=model,
                cond=cond,
                device=device,
                scheduler_kind=scheduler_kind,
                num_steps=num_steps,
                num_samples=num_samples,
            )
            latency_values.append((time.perf_counter() - start_time) * 1000.0)
            sample_batches.append(samples)

        all_samples = np.concatenate(sample_batches, axis=0)
        metrics = compute_sample_metrics(all_samples)
        record: Dict[str, object] = {
            "model_name": model_name,
            "dataset_name": case.dataset_name,
            "suite_name": case.suite_name,
            "traj_name": case.traj_name,
            "latency_ms": float(np.mean(latency_values)),
            "dist_pred_mean": float(dist_pred.squeeze().item()),
        }
        record.update(metrics)
        records.append(record)
    return records


def plot_encoder_summary(overall_rows: List[Dict[str, object]], out_dir: Path) -> None:
    """Plot overall latency and forward progress for encoder comparison."""
    labels = [str(row["model_name"]) for row in overall_rows]
    latency = [float(row["latency_ms_mean"]) for row in overall_rows]
    forward_progress = [float(row["forward_progress_mean_mean"]) for row in overall_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(labels, latency, color="#3498db")
    axes[0].set_title("Latency by Visual Encoder")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(labels, forward_progress, color="#e67e22")
    axes[1].set_title("Forward Progress by Visual Encoder")
    axes[1].set_ylabel("Forward Progress")
    axes[1].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig(out_dir / "encoder_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="NoMaD encoder comparison benchmark")
    parser.add_argument("--model-spec", action="append", default=None, help="Repeatable spec backbone=checkpoint")
    parser.add_argument("--dataset-root", action="append", default=None, help="Repeatable dataset root")
    parser.add_argument("--suite-offsets", default="short:8,medium:16,long:24", help="Suite spec")
    parser.add_argument("--cases-per-suite", type=int, default=8, help="Cases per suite")
    parser.add_argument("--num-runs", type=int, default=4, help="Repeated runs per model and case")
    parser.add_argument("--num-samples", type=int, default=8, help="Diffusion samples per run")
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddpm", help="Scheduler kind")
    parser.add_argument("--num-steps", type=int, default=10, help="Sampling steps")
    parser.add_argument("--seed", type=int, default=19, help="Random seed")
    args = parser.parse_args()

    specs = parse_model_specs(args.model_spec)
    if not specs:
        raise ValueError("At least one --model-spec backbone=checkpoint_path is required.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite_offsets = parse_suite_offsets(args.suite_offsets)
    cases = build_eval_cases(
        dataset_roots=args.dataset_root,
        suite_offsets=suite_offsets,
        cases_per_suite=args.cases_per_suite,
        seed=args.seed,
    )
    out_dir = make_output_dir("day4", "encoder_comparison_experiment")
    save_cases_json(out_dir / "eval_cases.json", cases)

    all_records: List[Dict[str, object]] = []
    for spec_name, checkpoint_path in specs:
        model = load_model_for_spec(spec_name, checkpoint_path, device)
        all_records.extend(
            evaluate_one_model(
                model_name=spec_name,
                model=model,
                cases=cases,
                device=device,
                num_runs=args.num_runs,
                num_samples=args.num_samples,
                scheduler_kind=args.scheduler,
                num_steps=args.num_steps,
                seed=args.seed,
            )
        )

    overall_rows = aggregate_case_records(all_records, ["model_name"], METRIC_FIELDS)
    suite_rows = aggregate_case_records(
        all_records,
        ["dataset_name", "suite_name", "model_name"],
        METRIC_FIELDS,
    )

    save_rows_csv(out_dir / "encoder_case_records.csv", all_records)
    save_rows_csv(out_dir / "encoder_overall_summary.csv", overall_rows)
    save_rows_csv(out_dir / "encoder_suite_summary.csv", suite_rows)
    save_rows_json(out_dir / "encoder_overall_summary.json", overall_rows)
    save_markdown_table(out_dir / "encoder_overall_summary.md", overall_rows)
    save_markdown_table(out_dir / "encoder_suite_summary.md", suite_rows)
    plot_encoder_summary(overall_rows, out_dir)
    print(f"Saved encoder comparison benchmark to: {out_dir}")


if __name__ == "__main__":
    main()
