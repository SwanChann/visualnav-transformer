#!/usr/bin/env python3
"""Statistical TTS benchmark for NoMaD diffusion inference."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import add_repo_paths

add_repo_paths()

from nomad_eval_common import (
    aggregate_case_records,
    build_eval_cases,
    compute_sample_metrics,
    load_case_tensors,
    make_output_dir,
    parse_suite_offsets,
    save_cases_json,
    save_markdown_table,
    save_rows_csv,
    save_rows_json,
    set_seed,
)
from shared.nomad_inference import NoMaDInferenceModule, NoMaDInferenceSpec


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
    "verifier_score_mean",
]


def parse_int_list(spec: str) -> list[int]:
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def parse_float_list(spec: str) -> list[float]:
    return [float(item.strip()) for item in spec.split(",") if item.strip()]


def build_configs(
    ddim_steps_list: list[int],
    cfg_weights: list[float],
    tts_budgets: list[int],
    tts_topk: int,
    tts_verifier: str,
) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = [
        {
            "config_name": "ddpm10_cfg0.0_tts0",
            "scheduler_kind": "ddpm",
            "num_steps": 10,
            "cfg_weight": 0.0,
            "tts_budget": 0,
            "tts_topk": 0,
            "tts_verifier": tts_verifier,
        }
    ]
    for num_steps in ddim_steps_list:
        for cfg_weight in cfg_weights:
            for tts_budget in tts_budgets:
                configs.append(
                    {
                        "config_name": f"ddim{num_steps}_cfg{cfg_weight:.1f}_tts{tts_budget}",
                        "scheduler_kind": "ddim",
                        "num_steps": num_steps,
                        "cfg_weight": cfg_weight,
                        "tts_budget": tts_budget,
                        "tts_topk": min(max(tts_topk, 1), max(tts_budget, 1)) if tts_budget > 0 else 0,
                        "tts_verifier": tts_verifier,
                    }
                )
    return configs


def evaluate_config(
    inference: NoMaDInferenceModule,
    obs_img,
    goal_img,
    config: dict[str, object],
    num_runs: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times_ms: List[float] = []
    sample_batches: List[np.ndarray] = []
    score_batches: List[np.ndarray] = []

    inference.scheduler_kind = str(config["scheduler_kind"])
    inference.ddim_steps = int(config["num_steps"])
    inference.cfg_weight = float(config["cfg_weight"])
    inference.tts_enabled = int(config["tts_budget"]) > 0
    inference.tts_budget = max(int(config["tts_budget"]), inference.num_samples)
    inference.tts_topk = max(int(config["tts_topk"]), 1)
    inference.tts_verifier = str(config["tts_verifier"])

    uncond = None
    if float(config["cfg_weight"]) != 0.0:
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
        score_batches.append(inference.score_action_candidates(samples, verifier=inference.tts_verifier))

    return (
        np.asarray(times_ms, dtype=np.float64),
        np.concatenate(sample_batches, axis=0),
        np.concatenate(score_batches, axis=0),
    )


def build_case_record(
    case,
    config: dict[str, object],
    times_ms: np.ndarray,
    samples: np.ndarray,
    scores: np.ndarray,
    baseline_mean: np.ndarray,
) -> Dict[str, object]:
    metrics = compute_sample_metrics(samples)
    record: Dict[str, object] = {
        "dataset_name": case.dataset_name,
        "suite_name": case.suite_name,
        "traj_name": case.traj_name,
        "config_name": config["config_name"],
        "scheduler_kind": config["scheduler_kind"],
        "num_steps": int(config["num_steps"]),
        "cfg_weight": float(config["cfg_weight"]),
        "tts_budget": int(config["tts_budget"]),
        "tts_topk": int(config["tts_topk"]),
        "tts_verifier": config["tts_verifier"],
        "latency_ms": float(times_ms.mean()),
        "latency_std_ms": float(times_ms.std(ddof=1)) if len(times_ms) > 1 else 0.0,
        "mse_vs_baseline": float(np.mean((samples.mean(axis=0) - baseline_mean) ** 2)),
        "verifier_score_mean": float(scores.mean()),
    }
    record.update(metrics)
    return record


def add_rank_score(rows: list[Dict[str, object]]) -> None:
    for row in rows:
        row["rank_score"] = (
            float(row["forward_progress_mean_mean"])
            - 0.050 * float(row["latency_ms_mean"])
            - 0.350 * float(row["lateral_abs_mean_mean"])
            - 0.150 * float(row["smoothness_mean_mean"])
        )
    rows.sort(key=lambda item: float(item["rank_score"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank


def plot_tts_tradeoff(overall_rows: list[Dict[str, object]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in overall_rows:
        label = str(row["config_name"])
        ax.scatter(
            float(row["latency_ms_mean"]),
            float(row["forward_progress_mean_mean"]),
            s=30 + 3 * float(row["tts_budget"]),
            alpha=0.75,
        )
        ax.annotate(label, (float(row["latency_ms_mean"]), float(row["forward_progress_mean_mean"])), fontsize=7)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Forward Progress")
    ax.set_title("DDIM / CFG / TTS Tradeoff")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "tts_combo_tradeoff.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical TTS benchmark for NoMaD")
    parser.add_argument("--dataset-root", action="append", default=None, help="Repeatable dataset root")
    parser.add_argument("--policy-config", default=None, help="Optional NoMaD config path")
    parser.add_argument("--weights", default=None, help="Optional NoMaD checkpoint path")
    parser.add_argument("--cases-per-suite", type=int, default=8, help="Cases sampled per suite")
    parser.add_argument("--suite-offsets", default="short:8,medium:16,long:24", help="Suite spec")
    parser.add_argument("--num-runs", type=int, default=8, help="Repeated runs per configuration")
    parser.add_argument("--num-samples", type=int, default=8, help="Samples per diffusion call")
    parser.add_argument("--ddim-steps-list", default="5,2", help="Comma-separated DDIM step list")
    parser.add_argument("--cfg-weights", default="0.0,0.5,1.0", help="Comma-separated CFG weights")
    parser.add_argument("--tts-budgets", default="0,16,32", help="Comma-separated TTS budgets; 0 disables TTS")
    parser.add_argument("--tts-topk", type=int, default=4, help="How many top TTS candidates to keep")
    parser.add_argument("--tts-verifier", choices=["heuristic", "forward", "conservative"], default="heuristic")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
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
    configs = build_configs(
        ddim_steps_list=parse_int_list(args.ddim_steps_list),
        cfg_weights=parse_float_list(args.cfg_weights),
        tts_budgets=parse_int_list(args.tts_budgets),
        tts_topk=args.tts_topk,
        tts_verifier=args.tts_verifier,
    )

    out_dir = make_output_dir("day4", "tts_stat_experiment")
    save_cases_json(out_dir / "eval_cases.json", cases)

    records: list[Dict[str, object]] = []
    for case_index, case in enumerate(cases):
        obs_img, goal_img = load_case_tensors(case, inference.device)

        baseline_config = configs[0]
        baseline_times, baseline_samples, baseline_scores = evaluate_config(
            inference=inference,
            obs_img=obs_img,
            goal_img=goal_img,
            config=baseline_config,
            num_runs=args.num_runs,
            base_seed=args.seed + case_index * 1000,
        )
        baseline_mean = baseline_samples.mean(axis=0)
        records.append(
            build_case_record(
                case=case,
                config=baseline_config,
                times_ms=baseline_times,
                samples=baseline_samples,
                scores=baseline_scores,
                baseline_mean=baseline_mean,
            )
        )

        for config_index, config in enumerate(configs[1:], start=1):
            times_ms, samples, scores = evaluate_config(
                inference=inference,
                obs_img=obs_img,
                goal_img=goal_img,
                config=config,
                num_runs=args.num_runs,
                base_seed=args.seed + case_index * 1000 + config_index * 100,
            )
            records.append(
                build_case_record(
                    case=case,
                    config=config,
                    times_ms=times_ms,
                    samples=samples,
                    scores=scores,
                    baseline_mean=baseline_mean,
                )
            )

    overall_rows = aggregate_case_records(
        records,
        ["config_name", "scheduler_kind", "num_steps", "cfg_weight", "tts_budget", "tts_topk", "tts_verifier"],
        METRIC_FIELDS,
    )
    suite_rows = aggregate_case_records(
        records,
        ["dataset_name", "suite_name", "config_name"],
        METRIC_FIELDS,
    )
    add_rank_score(overall_rows)

    save_rows_csv(out_dir / "tts_case_records.csv", records)
    save_rows_csv(out_dir / "tts_overall_summary.csv", overall_rows)
    save_rows_csv(out_dir / "tts_suite_summary.csv", suite_rows)
    save_rows_json(out_dir / "tts_overall_summary.json", overall_rows)
    save_markdown_table(out_dir / "tts_overall_summary.md", overall_rows)
    save_markdown_table(out_dir / "tts_suite_summary.md", suite_rows)
    plot_tts_tradeoff(overall_rows, out_dir)

    readme_lines = [
        "TTS Statistical Experiment for NoMaD",
        f"Cases: {len(cases)}",
        f"Num runs per config: {args.num_runs}",
        f"Num samples per diffusion call: {args.num_samples}",
        f"DDIM steps list: {args.ddim_steps_list}",
        f"CFG weights: {args.cfg_weights}",
        f"TTS budgets: {args.tts_budgets}",
        f"TTS topk: {args.tts_topk}",
        f"TTS verifier: {args.tts_verifier}",
    ]
    (out_dir / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"Saved TTS statistical benchmark to: {out_dir}")


if __name__ == "__main__":
    main()
