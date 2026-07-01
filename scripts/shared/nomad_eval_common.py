#!/usr/bin/env python3
"""Common helpers for NoMaD offline evaluation experiments."""

from __future__ import annotations

# Allow direct execution from any scripts/<category>/ path.
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve()
while SCRIPTS_ROOT.name != "scripts" and SCRIPTS_ROOT.parent != SCRIPTS_ROOT:
    SCRIPTS_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch

from tooling.project_paths import add_repo_paths, repo_path

add_repo_paths()

import offline_inference as offline_baseline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vint_train.training.train_utils import get_action


DEFAULT_SUITE_OFFSETS: Dict[str, int] = {
    "short": 8,
    "medium": 16,
    "long": 24,
}


@dataclass(frozen=True)
class EvalCase:
    """Single offline evaluation case."""

    dataset_name: str
    suite_name: str
    traj_name: str
    traj_path: str
    obs_time: int
    goal_time: int


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    # 中文注释：统一固定随机种子，保证重复实验可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_nomad_weights(explicit_path: str | None = None) -> Path:
    """Resolve the NoMaD checkpoint path."""
    # 中文注释：兼容仓库里两种常见权重放置方式
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            repo_path("deployment", "model_weights", "nomad", "nomad.pth"),
            repo_path("deployment", "model_weights", "nomad.pth"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_nomad_model(
    device: torch.device,
    weights_path: str | None = None,
):
    """Load the baseline NoMaD checkpoint using the existing builder."""
    resolved = resolve_nomad_weights(weights_path)
    offline_baseline.MODEL_WEIGHTS = str(resolved)
    model = offline_baseline.build_model(device)
    return model, resolved


def parse_suite_offsets(spec: str | None) -> Dict[str, int]:
    """Parse a suite spec string like 'short:8,medium:16,long:24'."""
    if not spec:
        return dict(DEFAULT_SUITE_OFFSETS)
    suites: Dict[str, int] = {}
    for item in spec.split(","):
        name, raw_offset = item.split(":")
        suites[name.strip()] = int(raw_offset.strip())
    return suites


def list_traj_dirs(dataset_root: Path) -> List[Path]:
    """List valid trajectory directories under a dataset root."""
    return sorted([path for path in dataset_root.iterdir() if path.is_dir()])


def count_frames(traj_dir: Path) -> int:
    """Count image frames in a trajectory directory."""
    frame_count = 0
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        frame_count += len(list(traj_dir.glob(ext)))
    return frame_count


def choose_obs_time(
    num_frames: int,
    goal_offset: int,
    context_size: int,
    selector_index: int,
) -> int:
    """Choose a deterministic observation index for a trajectory."""
    # 中文注释：在轨迹不同位置采样，避免所有样本都集中在单一时间段
    min_obs = context_size + 2
    max_obs = num_frames - goal_offset - 1
    if max_obs <= min_obs:
        return min_obs
    fractions = (0.25, 0.5, 0.75)
    fraction = fractions[selector_index % len(fractions)]
    return int(round(min_obs + fraction * (max_obs - min_obs)))


def build_eval_cases(
    dataset_roots: Sequence[str] | None,
    suite_offsets: Mapping[str, int],
    cases_per_suite: int,
    context_size: int = offline_baseline.CONTEXT_SIZE,
    seed: int = 0,
) -> List[EvalCase]:
    """Build a balanced case list across datasets and suites."""
    roots = [Path(path) for path in (dataset_roots or [offline_baseline.DATASET_ROOT])]
    rng = random.Random(seed)
    cases: List[EvalCase] = []

    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        traj_dirs = list_traj_dirs(root)
        rng.shuffle(traj_dirs)

        for suite_name, goal_offset in suite_offsets.items():
            eligible = [
                traj_dir
                for traj_dir in traj_dirs
                if count_frames(traj_dir) > context_size + goal_offset + 3
            ]
            selected = eligible[:cases_per_suite]
            for index, traj_dir in enumerate(selected):
                frame_count = count_frames(traj_dir)
                obs_time = choose_obs_time(
                    num_frames=frame_count,
                    goal_offset=goal_offset,
                    context_size=context_size,
                    selector_index=index,
                )
                cases.append(
                    EvalCase(
                        dataset_name=root.name,
                        suite_name=suite_name,
                        traj_name=traj_dir.name,
                        traj_path=str(traj_dir),
                        obs_time=obs_time,
                        goal_time=obs_time + goal_offset,
                    )
                )
    if not cases:
        raise RuntimeError("No evaluation cases were created. Check dataset paths and suite offsets.")
    return cases


def load_case_tensors(case: EvalCase, device: torch.device):
    """Load observation and goal tensors for a case."""
    obs_img = offline_baseline.prepare_observation(case.traj_path, case.obs_time).to(device)
    goal_img = offline_baseline.prepare_goal(case.traj_path, case.goal_time).to(device)
    return obs_img, goal_img


def encode_navigation_condition(
    model,
    obs_img: torch.Tensor,
    goal_img: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Encode the goal-conditioned NoMaD context."""
    mask = torch.zeros(1).long().to(device)
    with torch.no_grad():
        return model(
            "vision_encoder",
            obs_img=obs_img,
            goal_img=goal_img,
            input_goal_mask=mask,
        )


def encode_cfg_conditions(
    model,
    obs_img: torch.Tensor,
    goal_img: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode both goal-conditioned and goal-masked contexts."""
    mask_nav = torch.zeros(1).long().to(device)
    mask_explore = torch.ones(1).long().to(device)
    with torch.no_grad():
        cond = model(
            "vision_encoder",
            obs_img=obs_img,
            goal_img=goal_img,
            input_goal_mask=mask_nav,
        )
        uncond = model(
            "vision_encoder",
            obs_img=obs_img,
            goal_img=goal_img,
            input_goal_mask=mask_explore,
        )
    return cond, uncond


def make_scheduler(kind: str, num_train_timesteps: int = offline_baseline.NUM_DIFFUSION_ITERS):
    """Build a diffusion scheduler."""
    common_kwargs = dict(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    if kind == "ddpm":
        return DDPMScheduler(**common_kwargs)
    if kind == "ddim":
        return DDIMScheduler(**common_kwargs)
    raise ValueError(f"Unsupported scheduler kind: {kind}")


def run_diffusion_sampling(
    model,
    cond: torch.Tensor,
    device: torch.device,
    scheduler_kind: str,
    num_steps: int,
    num_samples: int,
    guidance_scale: float | None = None,
    uncond: torch.Tensor | None = None,
) -> np.ndarray:
    """Run diffusion sampling and return decoded trajectories."""
    scheduler = make_scheduler(scheduler_kind, offline_baseline.NUM_DIFFUSION_ITERS)
    cond_rep = cond.repeat(num_samples, 1)
    uncond_rep = None if uncond is None else uncond.repeat(num_samples, 1)

    naction = torch.randn(
        (num_samples, offline_baseline.LEN_TRAJ_PRED, 2),
        device=device,
    )
    scheduler.set_timesteps(num_steps)

    with torch.no_grad():
        for timestep in scheduler.timesteps:
            eps_cond = model(
                "noise_pred_net",
                sample=naction,
                timestep=timestep,
                global_cond=cond_rep,
            )
            if guidance_scale is None or guidance_scale == 0.0 or uncond_rep is None:
                eps_final = eps_cond
            else:
                eps_uncond = model(
                    "noise_pred_net",
                    sample=naction,
                    timestep=timestep,
                    global_cond=uncond_rep,
                )
                # 中文注释：使用 CFG 将条件分支与无条件分支融合
                eps_final = (1.0 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
            naction = scheduler.step(
                model_output=eps_final,
                timestep=timestep,
                sample=naction,
            ).prev_sample

    actions = get_action(naction).detach().cpu().numpy()
    return actions


def compute_sample_metrics(samples: np.ndarray) -> Dict[str, float]:
    """Compute aggregate metrics for a batch of sampled trajectories."""
    if samples.ndim != 3:
        raise ValueError(f"Expected [N, T, 2], got shape={samples.shape}")

    diffs = np.diff(samples, axis=1)
    second_diffs = np.diff(samples, n=2, axis=1) if samples.shape[1] > 2 else np.zeros((samples.shape[0], 1, 2))
    endpoints = samples[:, -1, :]

    path_lengths = np.linalg.norm(diffs, axis=-1).sum(axis=1)
    endpoint_norms = np.linalg.norm(endpoints, axis=1)
    smoothness = np.linalg.norm(second_diffs, axis=-1).mean(axis=1)

    return {
        "diversity": float(samples.std(axis=0).mean()),
        "path_length_mean": float(path_lengths.mean()),
        "endpoint_norm_mean": float(endpoint_norms.mean()),
        "forward_progress_mean": float(endpoints[:, 0].mean()),
        "lateral_abs_mean": float(np.abs(endpoints[:, 1]).mean()),
        "smoothness_mean": float(smoothness.mean()),
    }


def summarize_metric(values: Sequence[float]) -> Dict[str, float]:
    """Return mean/std/ci95 stats for a metric."""
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return {"mean": float(array.mean()), "std": std, "ci95": float(ci95)}


def aggregate_case_records(
    records: Sequence[Mapping[str, object]],
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
) -> List[Dict[str, object]]:
    """Aggregate case-level records into grouped statistics."""
    grouped: MutableMapping[tuple, List[Mapping[str, object]]] = {}
    for record in records:
        key = tuple(record[field] for field in group_fields)
        grouped.setdefault(key, []).append(record)

    rows: List[Dict[str, object]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        row: Dict[str, object] = {
            field: key[index] for index, field in enumerate(group_fields)
        }
        row["num_cases"] = len(bucket)
        for metric in metric_fields:
            stats = summarize_metric([float(item[metric]) for item in bucket])
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_ci95"] = stats["ci95"]
        rows.append(row)
    return rows


def save_rows_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Save a list of dictionaries as CSV."""
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_rows_json(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Save rows as JSON."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, indent=2)


def save_cases_json(path: Path, cases: Sequence[EvalCase]) -> None:
    """Save evaluation cases."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(case) for case in cases], handle, indent=2)


def format_float(value: object) -> str:
    """Format floats for markdown tables."""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def save_markdown_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Save a markdown table."""
    if not rows:
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(format_float(row[key]) for key in headers) + " |\n")


def make_output_dir(day_name: str, suffix: str) -> Path:
    """Create a timestamped result directory."""
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_path("results", day_name, f"{run_tag}_{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
