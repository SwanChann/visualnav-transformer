#!/usr/bin/env python3
"""Joint ablation: visual encoder × {DDIM steps, CFG weight, TTS budget}.

Covers three sub-experiments from SJTUThesis/missing_experiments.md §1.2:
- 编码器 × DDIM 联合消融 (4 encoders × DDIM-2/3/5)
- 编码器 × TTS 联合消融 (4 encoders × TTS-0/8/16)
- 编码器 × CFG 小规模联合消融 (EfficientNet-B0, ResNet-50 × CFG-0/1/2)

All configurations repeat num_runs times per case and report mean / std / CI95.
Each encoder is loaded once and reused across configurations to limit GPU cost.
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shared.nomad_eval_common import (
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
from models.nomad_vint_backbone_suite import build_backbone_nomad_model, get_backbone_names


METRIC_FIELDS = [
    "latency_ms",
    "diversity",
    "path_length_mean",
    "endpoint_norm_mean",
    "forward_progress_mean",
    "lateral_abs_mean",
    "smoothness_mean",
]


DEFAULT_MODEL_SPECS = [
    ("efficientnet_b0", "deployment/model_weights/nomad_efficientb0.pth"),
    ("dinov2_small", "deployment/model_weights/nomad_dinvo2_small.pth"),
    ("convnext_tiny", "deployment/model_weights/nomad_convnext_tiny.pth"),
    ("resnet50", "deployment/model_weights/nomad_resnet50.pth"),
]


@dataclass(frozen=True)
class JointConfig:
    """One (encoder, scheduler, steps, cfg_weight, tts_budget) configuration."""

    experiment: str
    config_name: str
    scheduler: str
    num_steps: int
    cfg_weight: float
    tts_budget: int

    @property
    def uses_cfg(self) -> bool:
        return self.cfg_weight != 0.0

    @property
    def uses_tts(self) -> bool:
        return self.tts_budget > 0


def build_configs(experiments: List[str]) -> List[JointConfig]:
    """Build the JointConfig list for the requested sub-experiments."""
    configs: List[JointConfig] = []
    if "ddim" in experiments:
        for steps in (2, 3, 5):
            configs.append(
                JointConfig(
                    experiment="encoder_x_ddim",
                    config_name=f"ddim{steps}_cfg0_tts0",
                    scheduler="ddim",
                    num_steps=steps,
                    cfg_weight=0.0,
                    tts_budget=0,
                )
            )
    if "tts" in experiments:
        for budget in (0, 8, 16):
            configs.append(
                JointConfig(
                    experiment="encoder_x_tts",
                    config_name=f"ddim2_cfg0_tts{budget}",
                    scheduler="ddim",
                    num_steps=2,
                    cfg_weight=0.0,
                    tts_budget=budget,
                )
            )
    if "cfg" in experiments:
        for weight in (0.0, 1.0, 2.0):
            weight_tag = ("cfg0" if weight == 0.0 else f"cfg{weight:.1f}".replace(".", "p"))
            configs.append(
                JointConfig(
                    experiment="encoder_x_cfg",
                    config_name=f"ddim2_{weight_tag}_tts0",
                    scheduler="ddim",
                    num_steps=2,
                    cfg_weight=weight,
                    tts_budget=0,
                )
            )
    return configs


def parse_model_specs(raw_specs: List[str] | None) -> List[Tuple[str, str]]:
    """Parse optional explicit specs; fall back to the default 4-encoder suite."""
    if not raw_specs:
        return list(DEFAULT_MODEL_SPECS)
    parsed: List[Tuple[str, str]] = []
    for spec in raw_specs:
        name, path = spec.split("=", 1)
        parsed.append((name.strip(), path.strip()))
    return parsed


def load_model(spec_name: str, checkpoint_path: str, device: torch.device):
    """Load a NoMaD model for a given encoder spec / checkpoint."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_absolute():
        checkpoint = (Path(__file__).resolve().parents[1] / checkpoint_path).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if spec_name == "efficientnet_b0":
        model, _ = load_nomad_model(device=device, weights_path=str(checkpoint))
        return model

    if spec_name not in get_backbone_names():
        raise KeyError(f"Unsupported encoder spec: {spec_name}")

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


def sample_with_tts(
    model,
    cond: torch.Tensor,
    uncond: torch.Tensor | None,
    device: torch.device,
    config: JointConfig,
    num_samples: int,
) -> np.ndarray:
    """Run diffusion sampling with optional CFG and optional TTS selection."""
    if not config.uses_tts:
        return run_diffusion_sampling(
            model=model,
            cond=cond,
            device=device,
            scheduler_kind=config.scheduler,
            num_steps=config.num_steps,
            num_samples=num_samples,
            guidance_scale=config.cfg_weight if config.uses_cfg else None,
            uncond=uncond if config.uses_cfg else None,
        )

    budget = max(config.tts_budget, num_samples)
    topk = max(num_samples // 2, 1)
    generated = 0
    candidate_batches: List[np.ndarray] = []
    while generated < budget:
        batch = min(num_samples, budget - generated)
        candidate_batches.append(
            run_diffusion_sampling(
                model=model,
                cond=cond,
                device=device,
                scheduler_kind=config.scheduler,
                num_steps=config.num_steps,
                num_samples=batch,
                guidance_scale=config.cfg_weight if config.uses_cfg else None,
                uncond=uncond if config.uses_cfg else None,
            )
        )
        generated += batch

    candidates = np.concatenate(candidate_batches, axis=0)
    # 中文注释：默认 verifier：偏好“向前推进、横摆小、曲率小、路径效率高”的候选
    diffs = np.diff(candidates, axis=1)
    if candidates.shape[1] > 2:
        second_diffs = np.diff(candidates, n=2, axis=1)
        smoothness = np.linalg.norm(second_diffs, axis=-1).mean(axis=1)
    else:
        smoothness = np.zeros((candidates.shape[0],), dtype=np.float32)
    endpoints = candidates[:, -1, :]
    path_length = np.linalg.norm(diffs, axis=-1).sum(axis=1)
    endpoint_norm = np.linalg.norm(endpoints, axis=1)
    efficiency = endpoint_norm / np.maximum(path_length, 1e-6)
    scores = (
        endpoints[:, 0]
        + 0.35 * efficiency
        - 0.35 * np.abs(endpoints[:, 1])
        - 0.15 * smoothness
    )
    ranking = np.argsort(scores)[::-1]
    return candidates[ranking[:topk]]


def evaluate_one_config(
    model,
    model_name: str,
    cases,
    device: torch.device,
    config: JointConfig,
    num_runs: int,
    num_samples: int,
    seed: int,
) -> List[Dict[str, object]]:
    """Evaluate one (encoder, config) pair over all cases."""
    records: List[Dict[str, object]] = []
    for case_index, case in enumerate(cases):
        obs_img, goal_img = load_case_tensors(case, device)
        if config.uses_cfg:
            cond, uncond = encode_cfg_conditions(model, obs_img, goal_img, device)
        else:
            cond = encode_navigation_condition(model, obs_img, goal_img, device)
            uncond = None

        with torch.no_grad():
            dist_pred = model("dist_pred_net", obsgoal_cond=cond)

        # 中文注释：每个 case 的首次采样会因 GPU kernel JIT/cache 产生冷启动偏差，先跑一次 warm-up 不计入时延
        set_seed(seed + case_index * 1000 - 1)
        _ = sample_with_tts(
            model=model,
            cond=cond,
            uncond=uncond,
            device=device,
            config=config,
            num_samples=num_samples,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latency_values: List[float] = []
        sample_batches: List[np.ndarray] = []
        for run_index in range(num_runs):
            set_seed(seed + case_index * 1000 + run_index)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            samples = sample_with_tts(
                model=model,
                cond=cond,
                uncond=uncond,
                device=device,
                config=config,
                num_samples=num_samples,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_values.append((time.perf_counter() - start_time) * 1000.0)
            sample_batches.append(samples)

        all_samples = np.concatenate(sample_batches, axis=0)
        metrics = compute_sample_metrics(all_samples)
        record: Dict[str, object] = {
            "experiment": config.experiment,
            "model_name": model_name,
            "config_name": config.config_name,
            "scheduler": config.scheduler,
            "num_steps": config.num_steps,
            "cfg_weight": config.cfg_weight,
            "tts_budget": config.tts_budget,
            "dataset_name": case.dataset_name,
            "suite_name": case.suite_name,
            "traj_name": case.traj_name,
            "latency_ms": float(np.mean(latency_values)),
            "dist_pred_mean": float(dist_pred.squeeze().item()),
        }
        record.update(metrics)
        records.append(record)
    return records


def plot_grouped_bars(
    rows: List[Dict[str, object]],
    experiment: str,
    metric: str,
    out_path: Path,
) -> None:
    """Plot grouped bar chart for one experiment: encoder × config."""
    filtered = [row for row in rows if row["experiment"] == experiment]
    if not filtered:
        return
    encoders = sorted({str(row["model_name"]) for row in filtered})
    configs = sorted({str(row["config_name"]) for row in filtered})

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(configs)), 4.5))
    x = np.arange(len(configs))
    bar_width = 0.8 / max(len(encoders), 1)
    for index, encoder in enumerate(encoders):
        values = []
        errors = []
        for config_name in configs:
            match = [
                row
                for row in filtered
                if row["model_name"] == encoder and row["config_name"] == config_name
            ]
            if match:
                values.append(float(match[0][f"{metric}_mean"]))
                errors.append(float(match[0][f"{metric}_ci95"]))
            else:
                values.append(float("nan"))
                errors.append(0.0)
        ax.bar(
            x + index * bar_width - 0.4 + bar_width / 2,
            values,
            bar_width,
            yerr=errors,
            capsize=3,
            label=encoder,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_title(f"{experiment} — {metric}")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint encoder × (DDIM / TTS / CFG) ablation")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["ddim", "tts", "cfg", "all"],
        default=["all"],
        help="Which sub-experiments to run",
    )
    parser.add_argument(
        "--model-spec",
        action="append",
        default=None,
        help="Repeatable spec name=checkpoint_path (default: 4-encoder suite)",
    )
    parser.add_argument("--dataset-root", action="append", default=None, help="Repeatable dataset root")
    parser.add_argument("--suite-offsets", default="short:8,medium:16", help="Suite spec")
    parser.add_argument("--cases-per-suite", type=int, default=6, help="Cases per suite")
    parser.add_argument("--num-runs", type=int, default=3, help="Repeated runs per case / config")
    parser.add_argument("--num-samples", type=int, default=8, help="Diffusion samples per run")
    parser.add_argument("--seed", type=int, default=23, help="Random seed")
    parser.add_argument(
        "--cfg-encoders",
        nargs="+",
        default=["efficientnet_b0", "resnet50"],
        help="Encoders for the CFG mini-ablation",
    )
    args = parser.parse_args()

    experiments = args.experiments
    if "all" in experiments:
        experiments = ["ddim", "tts", "cfg"]
    configs_all = build_configs(experiments)
    if not configs_all:
        raise ValueError("No configurations selected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite_offsets = parse_suite_offsets(args.suite_offsets)
    specs = parse_model_specs(args.model_spec)
    cases = build_eval_cases(
        dataset_roots=args.dataset_root,
        suite_offsets=suite_offsets,
        cases_per_suite=args.cases_per_suite,
        seed=args.seed,
    )
    out_dir = make_output_dir("day5", "encoder_joint_ablation_experiment")
    save_cases_json(out_dir / "eval_cases.json", cases)

    all_records: List[Dict[str, object]] = []
    for spec_name, checkpoint_path in specs:
        print(f"[encoder_joint_ablation] loading {spec_name} from {checkpoint_path}")
        model = load_model(spec_name, checkpoint_path, device)
        for config in configs_all:
            if config.experiment == "encoder_x_cfg" and spec_name not in args.cfg_encoders:
                continue
            print(f"  -> {config.experiment} | {config.config_name}")
            records = evaluate_one_config(
                model=model,
                model_name=spec_name,
                cases=cases,
                device=device,
                config=config,
                num_runs=args.num_runs,
                num_samples=args.num_samples,
                seed=args.seed,
            )
            all_records.extend(records)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_rows_csv(out_dir / "joint_case_records.csv", all_records)

    overall_rows = aggregate_case_records(
        all_records,
        ["experiment", "model_name", "config_name"],
        METRIC_FIELDS,
    )
    suite_rows = aggregate_case_records(
        all_records,
        ["experiment", "dataset_name", "suite_name", "model_name", "config_name"],
        METRIC_FIELDS,
    )
    save_rows_csv(out_dir / "joint_overall_summary.csv", overall_rows)
    save_rows_csv(out_dir / "joint_suite_summary.csv", suite_rows)
    save_rows_json(out_dir / "joint_overall_summary.json", overall_rows)
    save_markdown_table(out_dir / "joint_overall_summary.md", overall_rows)
    save_markdown_table(out_dir / "joint_suite_summary.md", suite_rows)

    for experiment in {row["experiment"] for row in overall_rows}:
        for metric in ("latency_ms", "forward_progress_mean", "diversity", "smoothness_mean"):
            plot_grouped_bars(
                overall_rows,
                experiment=experiment,
                metric=metric,
                out_path=out_dir / f"{experiment}_{metric}.png",
            )

    readme_lines = [
        f"Experiments: {experiments}",
        f"Model specs: {specs}",
        f"CFG encoders: {args.cfg_encoders}",
        f"Device: {device}",
        f"Cases: {len(cases)}",
        f"Suites: {suite_offsets}",
        f"Num runs per config: {args.num_runs}",
        f"Num samples per run: {args.num_samples}",
    ]
    (out_dir / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"Saved joint ablation to: {out_dir}")


if __name__ == "__main__":
    main()
