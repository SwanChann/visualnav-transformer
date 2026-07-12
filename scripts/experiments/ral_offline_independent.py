#!/usr/bin/env python3
"""Independent physical-unit offline evaluation for the frozen NoMaD checkpoint."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import sys
from typing import Any, Sequence

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for candidate in (SCRIPTS_ROOT, SCRIPTS_ROOT / "shared", SCRIPTS_ROOT / "analysis"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from shared.nomad_eval_common import EvalCase, choose_obs_time, count_frames
from shared.nomad_inference import NoMaDInferenceModule, NoMaDInferenceSpec
from vint_train.data.data_utils import to_local_coords


DEFAULT_PROTOCOL = REPO_ROOT / "后续研究内容" / "benchmark" / "ral_mujoco_protocol_v0.1.json"
DEFAULT_DATASET = REPO_ROOT / "nomad_dataset" / "go_stanford"
DEFAULT_SPLIT = REPO_ROOT / "results" / "research" / "data_audit" / "proposed_splits_v2" / "go_stanford" / "test" / "traj_names.txt"
DEFAULT_RESULTS = REPO_ROOT / "results" / "research" / "ral_offline_independent"
DEFAULT_DATA_CONFIG = REPO_ROOT / "train" / "vint_train" / "data" / "data_config.yaml"
ACTION_HORIZON = 8
CONTEXT_SIZE = 3
SUITES = {"short": 8, "medium": 16, "long": 24}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_metric_waypoint_spacing(dataset_root: Path, data_config_path: Path) -> float:
    """Load the physical action scale from the same dataset config used by training."""
    config = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
    dataset_id = dataset_root.resolve().name
    try:
        spacing = float(config[dataset_id]["metric_waypoint_spacing"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing metric_waypoint_spacing for dataset {dataset_id!r}") from exc
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"Invalid metric_waypoint_spacing={spacing!r} for {dataset_id!r}")
    return spacing


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def compute_independent_metrics(chosen_m: np.ndarray, candidates_m: np.ndarray, ground_truth_m: np.ndarray) -> dict[str, float]:
    chosen = np.asarray(chosen_m, dtype=np.float64)
    candidates = np.asarray(candidates_m, dtype=np.float64)
    ground_truth = np.asarray(ground_truth_m, dtype=np.float64)
    if chosen.shape != ground_truth.shape or candidates.ndim != 3 or candidates.shape[1:] != ground_truth.shape:
        raise ValueError(f"Incompatible shapes chosen={chosen.shape}, candidates={candidates.shape}, gt={ground_truth.shape}")
    chosen_errors = np.linalg.norm(chosen - ground_truth, axis=-1)
    candidate_ades = np.linalg.norm(candidates - ground_truth[None], axis=-1).mean(axis=1)
    pairwise = []
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            pairwise.append(float(np.linalg.norm(candidates[left] - candidates[right], axis=-1).mean()))
    pred_delta = chosen[-1] - chosen[-2]
    gt_delta = ground_truth[-1] - ground_truth[-2]
    pred_heading = math.atan2(float(pred_delta[1]), float(pred_delta[0]))
    gt_heading = math.atan2(float(gt_delta[1]), float(gt_delta[0]))
    return {
        "action_ade_m": float(chosen_errors.mean()),
        "action_fde_m": float(chosen_errors[-1]),
        "progress_error_m": float(abs(chosen[-1, 0] - ground_truth[-1, 0])),
        "heading_error_rad": abs(wrap_angle(pred_heading - gt_heading)),
        "minade_k_m": float(candidate_ades.min()),
        "candidate_diversity_m": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def load_ground_truth(case: EvalCase) -> np.ndarray:
    with (Path(case.traj_path) / "traj_data.pkl").open("rb") as handle:
        data = pickle.load(handle)
    positions = np.asarray(data["position"], dtype=np.float64).reshape(len(data["position"]), -1)[:, :2]
    yaw = np.asarray(data["yaw"], dtype=np.float64).reshape(len(data["yaw"]), -1)[:, 0]
    indices = np.arange(case.obs_time, case.obs_time + ACTION_HORIZON + 1)
    if indices[-1] >= len(positions):
        raise ValueError(f"Case lacks {ACTION_HORIZON} future actions: {case.traj_name}@{case.obs_time}")
    local = to_local_coords(positions[indices], positions[indices[0]], float(yaw[indices[0]]))
    return np.asarray(local[1:], dtype=np.float32)


def freeze_cases(dataset_root: Path, split_path: Path, cases_per_suite: int, seed: int) -> list[EvalCase]:
    names = [line.strip() for line in split_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(names)
    cases = []
    for suite_name, goal_offset in SUITES.items():
        eligible = []
        for name in names:
            trajectory = dataset_root / name
            if not trajectory.is_dir():
                continue
            frames = count_frames(trajectory)
            if frames > CONTEXT_SIZE + max(goal_offset, ACTION_HORIZON) + 3:
                eligible.append((name, trajectory, frames))
        for index, (name, trajectory, frames) in enumerate(eligible[:cases_per_suite]):
            obs_time = choose_obs_time(frames, goal_offset, CONTEXT_SIZE, index)
            if obs_time + ACTION_HORIZON >= frames:
                continue
            cases.append(EvalCase(dataset_root.name, suite_name, name, str(trajectory), obs_time, obs_time + goal_offset))
    if len(cases) != len(SUITES) * cases_per_suite:
        raise RuntimeError(f"Expected {len(SUITES) * cases_per_suite} frozen cases, got {len(cases)}")
    return cases


def _condition(module: NoMaDInferenceModule, case: EvalCase):
    import offline_inference
    obs = offline_inference.prepare_observation(case.traj_path, case.obs_time).to(module.device)
    goal = offline_inference.prepare_goal(case.traj_path, case.goal_time).to(module.device)
    cond = module.encode_condition(obs, goal, goal_mask_value=0)
    uncond = module.encode_condition(obs, goal, goal_mask_value=1) if module.cfg_weight != 0.0 else None
    return cond, uncond


def _summarize(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("action_ade_m", "action_fde_m", "progress_error_m", "heading_error_rad", "minade_k_m", "candidate_diversity_m", "sampler_latency_ms")
    rows = []
    for config_id in sorted({record["config_id"] for record in records}):
        bucket = [record for record in records if record["config_id"] == config_id and record["status"] == "valid"]
        row = {"config_id": config_id, "case_n": len(bucket)}
        for metric in metrics:
            values = np.asarray([record[metric] for record in bucket], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_p50"] = float(np.percentile(values, 50))
            row[f"{metric}_p95"] = float(np.percentile(values, 95))
            row[f"{metric}_p99"] = float(np.percentile(values, 99))
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> Path:
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.results_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    data_config_path = args.data_config.resolve()
    metric_waypoint_spacing_m = load_metric_waypoint_spacing(args.dataset_root, data_config_path)
    cases = freeze_cases(args.dataset_root.resolve(), args.split.resolve(), args.cases_per_suite, args.case_seed)
    cases_payload = {
        "created_at_utc": utc_now(), "case_seed": args.case_seed, "dataset_root": str(args.dataset_root.resolve()),
        "split_path": str(args.split.resolve()), "split_sha256": sha256_file(args.split.resolve()),
        "metric_waypoint_spacing_m": metric_waypoint_spacing_m,
        "data_config_path": str(data_config_path),
        "data_config_sha256": sha256_file(data_config_path),
        "cases": [asdict(case) for case in cases],
    }
    cases_path = run_dir / "frozen_cases.json"
    cases_path.write_text(json.dumps(cases_payload, indent=2) + "\n", encoding="utf-8")
    cases_sha = sha256_file(cases_path)
    records = []
    device = args.device or protocol["runtime"]["device"]
    for config in protocol["configurations"]:
        module = NoMaDInferenceModule(NoMaDInferenceSpec(
            policy_config=protocol["immutable_inputs"]["policy_config"]["path"], policy_checkpoint=protocol["immutable_inputs"]["policy_checkpoint"]["path"],
            scheduler_kind=config["scheduler"], ddim_steps=config["diffusion_steps"], cfg_weight=config["cfg_weight"], device=device, num_samples=config["candidate_k"],
            tts_enabled=config["tts_enabled"], tts_budget=config.get("tts_budget", 8), tts_topk=config.get("tts_topk", 1), tts_verifier=config.get("tts_verifier", "heuristic"), image_resize_mode=protocol["navigation_contract"]["image_resize_mode"],
        ))
        first_cond, first_uncond = _condition(module, cases[0])
        for warmup in range(args.warmup_calls):
            torch.manual_seed(args.case_seed + warmup)
            module.sample_actions(first_cond, first_uncond)
        module.sampler_timings_ms.clear()
        for case_index, case in enumerate(cases):
            base = {"config_id": config["id"], "case_index": case_index, "dataset": case.dataset_name, "suite": case.suite_name, "trajectory": case.traj_name, "obs_time": case.obs_time, "goal_time": case.goal_time, "candidate_k": config["candidate_k"], "status": "invalid"}
            try:
                cond, uncond = _condition(module, case)
                paired_seed = args.sampling_seed + case_index
                torch.manual_seed(paired_seed); np.random.seed(paired_seed)
                selected = module.sample_actions(cond, uncond)
                candidates = np.asarray(module.last_all_candidates, dtype=np.float32) * metric_waypoint_spacing_m
                chosen = np.asarray(selected, dtype=np.float32).mean(axis=0) * metric_waypoint_spacing_m
                ground_truth = load_ground_truth(case)
                base.update(compute_independent_metrics(chosen, candidates, ground_truth))
                base.update({"sampling_seed": paired_seed, "sampler_latency_ms": float(module.sampler_timings_ms[-1]), "status": "valid", "error": None})
            except Exception as exc:
                base.update({"sampling_seed": args.sampling_seed + case_index, "sampler_latency_ms": None, "error": f"{type(exc).__name__}: {exc}"})
            records.append(base)
    raw_path = run_dir / "case_records.json"
    raw_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "case_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    summary = _summarize(records)
    summary_payload = {
        "schema_version": "0.1.0", "run_id": run_id, "created_at_utc": utc_now(), "status": "completed" if all(record["status"] == "valid" for record in records) else "completed_with_invalid",
        "protocol_sha256": sha256_file(protocol_path), "checkpoint_sha256": protocol["immutable_inputs"]["policy_checkpoint"]["sha256"],
        "split_sha256": cases_payload["split_sha256"], "frozen_cases_sha256": cases_sha, "case_count": len(cases), "configuration_count": len(protocol["configurations"]), "record_count": len(records),
        "invalid_count": sum(record["status"] != "valid" for record in records), "device": device, "precision": "float32", "candidate_k": 8, "warmup_calls": args.warmup_calls,
        "latency_scope": "sampler_per_call", "cuda_synchronization": "inside NoMaDInferenceModule.sample_actions before and after each call", "training_performed": False,
        "metric_waypoint_spacing_m": metric_waypoint_spacing_m,
        "data_config_path": str(data_config_path),
        "data_config_sha256": sha256_file(data_config_path),
        "physical_unit_note": f"NoMaD decoded waypoint units are multiplied by {args.dataset_root.resolve().name} metric_waypoint_spacing={metric_waypoint_spacing_m:g} m loaded from the frozen training data config; GT is local-coordinate pose displacement in meters.",
        "summary": summary,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "records": len(records), "invalid": summary_payload["invalid_count"]}, indent=2))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--cases-per-suite", type=int, default=5)
    parser.add_argument("--case-seed", type=int, default=20260711)
    parser.add_argument("--sampling-seed", type=int, default=11000)
    parser.add_argument("--warmup-calls", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
