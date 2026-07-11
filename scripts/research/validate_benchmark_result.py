#!/usr/bin/env python3
"""Validate benchmark result v0.1 structure and semantic invariants."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
TRACKS = {"iid", "mixed", "lodo", "corruption", "closed_loop"}
HEADS = {"deterministic", "ddpm", "ddim", "flow", "rectified_flow"}
CORRUPTIONS = {
    "brightness", "contrast", "gaussian_noise", "motion_blur",
    "frame_drop", "temporal_jitter", "goal_image_mismatch"
}
LATENCY_SCOPES = {
    "model_forward_per_call",
    "sampler_per_call",
    "full_loop_per_call",
    "case_level_mean",
}
OFFLINE_METRICS = {
    "action_ade_m",
    "action_fde_m",
    "progress_error_m",
    "latency_mean_ms",
    "latency_p95_ms",
}
CLOSED_LOOP_METRICS = {
    "success_rate",
    "spl",
    "collision_rate",
    "stuck_rate",
    "fall_rate",
    "loop_latency_p95_ms",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    def need(obj: dict[str, Any], fields: set[str], location: str) -> None:
        for field in sorted(fields - set(obj)):
            errors.append(f"{location}: missing {field}")

    need(
        payload,
        {"schema_version", "run_id", "status", "track", "created_at", "git", "data", "model", "training", "inference", "evaluation", "artifacts"},
        "$",
    )
    if errors:
        return {"passed": False, "errors": errors, "warnings": warnings}
    if payload["schema_version"] != "0.1.0":
        errors.append("schema_version must be 0.1.0")
    if payload["track"] not in TRACKS:
        errors.append(f"unknown track {payload['track']!r}")
    if payload["status"] not in {"completed", "failed", "aborted"}:
        errors.append(f"unknown status {payload['status']!r}")

    git = payload["git"]
    data = payload["data"]
    model = payload["model"]
    training = payload["training"]
    inference = payload["inference"]
    evaluation = payload["evaluation"]
    artifacts = payload["artifacts"]
    for name, obj in (("git", git), ("data", data), ("model", model), ("training", training), ("inference", inference), ("evaluation", evaluation), ("artifacts", artifacts)):
        if not isinstance(obj, dict):
            errors.append(f"{name} must be an object")
    if errors:
        return {"passed": False, "errors": errors, "warnings": warnings}

    need(git, {"commit", "dirty", "diff_sha256"}, "git")
    need(data, {"registry_sha256", "manifest_sha256", "split_id", "split_sha256", "train_datasets", "eval_dataset", "heldout_dataset"}, "data")
    need(model, {"model_id", "family", "encoder", "head", "parameter_count", "trainable_parameter_count", "input_resolution", "context_size", "observation_frames_including_current", "action_history_length", "encoder_weight_sharing", "encoder_frozen", "action_horizon"}, "model")
    need(training, {"seed", "sampler", "optimizer_steps", "examples_seen_total", "examples_seen_by_dataset"}, "training")
    need(inference, {"device", "batch_size", "precision", "nfe", "candidate_count", "latency_scope", "warmup_calls"}, "inference")
    need(evaluation, {"unit", "sample_count", "metrics", "stabilizer_mode"}, "evaluation")
    need(artifacts, {"raw_results", "config", "checkpoint"}, "artifacts")
    if errors:
        return {"passed": False, "errors": errors, "warnings": warnings}

    if not COMMIT.fullmatch(str(git["commit"])):
        errors.append("git.commit must be a 40-hex commit")
    if not isinstance(git["dirty"], bool):
        errors.append("git.dirty must be boolean")
    if git["dirty"] and not SHA256.fullmatch(str(git["diff_sha256"])):
        errors.append("dirty runs require git.diff_sha256")
    for field in ("registry_sha256", "manifest_sha256", "split_sha256"):
        if not SHA256.fullmatch(str(data[field])):
            errors.append(f"data.{field} must be SHA-256 hex")

    train_datasets = data["train_datasets"]
    if not isinstance(train_datasets, list) or not train_datasets or not all(isinstance(item, str) and item for item in train_datasets):
        errors.append("data.train_datasets must be a non-empty string list")
        train_datasets = []
    track = payload["track"]
    if track == "mixed" and len(set(train_datasets)) < 2:
        errors.append("mixed track requires at least two train datasets")
    if track == "lodo":
        heldout = data["heldout_dataset"]
        if not heldout:
            errors.append("lodo requires data.heldout_dataset")
        if heldout in train_datasets:
            errors.append("lodo heldout dataset appears in training data")
        if heldout != data["eval_dataset"]:
            errors.append("lodo eval_dataset must equal heldout_dataset")
        if len(set(train_datasets)) < 2:
            warnings.append("lodo has fewer than two train datasets; this is not a 3+ dataset benchmark")
    elif data["heldout_dataset"] is not None:
        warnings.append("heldout_dataset is set outside lodo track")
    if track == "corruption":
        corruption = data.get("corruption")
        if not isinstance(corruption, dict):
            errors.append("corruption track requires data.corruption object")
        else:
            need(corruption, {"type", "severity", "seed", "clean_reference_run_id"}, "data.corruption")
            if corruption.get("type") not in CORRUPTIONS:
                errors.append(f"unsupported corruption type {corruption.get('type')!r}")
            if corruption.get("severity") not in {1, 2, 3}:
                errors.append("corruption severity must be 1, 2, or 3")
            if not isinstance(corruption.get("seed"), int):
                errors.append("corruption seed must be integer")
            if not corruption.get("clean_reference_run_id"):
                errors.append("corruption clean_reference_run_id is required")

    for field in ("parameter_count", "trainable_parameter_count", "context_size", "observation_frames_including_current", "action_horizon"):
        if not isinstance(model[field], int) or model[field] < 1:
            errors.append(f"model.{field} must be a positive integer")
    if isinstance(model["parameter_count"], int) and isinstance(model["trainable_parameter_count"], int) and model["trainable_parameter_count"] > model["parameter_count"]:
        errors.append("trainable_parameter_count exceeds parameter_count")
    if model["head"] not in HEADS:
        errors.append(f"unsupported model.head {model['head']!r}")
    if not isinstance(model["action_history_length"], int) or model["action_history_length"] < 0:
        errors.append("model.action_history_length must be a nonnegative integer")
    for field in ("encoder_weight_sharing", "encoder_frozen"):
        if not isinstance(model[field], bool):
            errors.append(f"model.{field} must be boolean")
    if (
        isinstance(model["context_size"], int)
        and isinstance(model["observation_frames_including_current"], int)
        and model["observation_frames_including_current"] != model["context_size"] + 1
    ):
        errors.append("observation_frames_including_current must equal context_size + 1")
    resolution = model["input_resolution"]
    if not isinstance(resolution, list) or len(resolution) != 2 or not all(isinstance(value, int) and value > 0 for value in resolution):
        errors.append("model.input_resolution must be [height, width] positive integers")

    examples = training["examples_seen_by_dataset"]
    if not isinstance(examples, dict) or not all(isinstance(value, int) and value >= 0 for value in examples.values()):
        errors.append("training.examples_seen_by_dataset must map datasets to nonnegative integers")
    elif sum(examples.values()) != training["examples_seen_total"]:
        errors.append("examples_seen_total does not equal per-dataset sum")
    if isinstance(examples, dict) and set(examples) != set(train_datasets):
        errors.append("examples_seen_by_dataset keys must exactly equal train_datasets")

    if inference["latency_scope"] not in LATENCY_SCOPES:
        errors.append("invalid inference.latency_scope")
    for field in ("batch_size", "nfe", "candidate_count"):
        if not isinstance(inference[field], int) or inference[field] < 1:
            errors.append(f"inference.{field} must be positive integer")
    if not isinstance(inference["warmup_calls"], int) or inference["warmup_calls"] < 0:
        errors.append("inference.warmup_calls must be nonnegative integer")
    if model["head"] == "deterministic" and inference["nfe"] != 1:
        errors.append("deterministic head requires nfe=1")
    if model["head"] in {"flow", "rectified_flow"} and inference["nfe"] not in {1, 2, 4, 8}:
        errors.append("flow heads require nfe in {1,2,4,8}")

    metrics = evaluation["metrics"]
    if not isinstance(metrics, dict):
        errors.append("evaluation.metrics must be an object")
        metrics = {}
    required_metrics = CLOSED_LOOP_METRICS if track == "closed_loop" else OFFLINE_METRICS
    missing_metrics = sorted(required_metrics - set(metrics))
    if missing_metrics:
        errors.append(f"missing required metrics: {missing_metrics}")
    for name, value in metrics.items():
        if not _is_number(value):
            errors.append(f"metric {name} must be finite numeric")
        elif value < 0:
            errors.append(f"metric {name} must be nonnegative")
    for rate in ("success_rate", "spl", "collision_rate", "stuck_rate", "fall_rate"):
        if rate in metrics and _is_number(metrics[rate]) and not 0 <= metrics[rate] <= 1:
            errors.append(f"metric {rate} must be in [0,1]")

    if track == "closed_loop":
        if evaluation["unit"] != "episode":
            errors.append("closed_loop evaluation.unit must be episode")
        if not evaluation["stabilizer_mode"]:
            errors.append("closed_loop requires stabilizer_mode")
        if inference["latency_scope"] != "full_loop_per_call":
            errors.append("closed_loop latency_scope must be full_loop_per_call")
    else:
        if evaluation["unit"] != "trajectory":
            errors.append("offline evaluation.unit must be trajectory")
        if inference["latency_scope"] == "case_level_mean":
            warnings.append("case_level_mean is not per-call tail latency")

    if payload["status"] == "completed" and not artifacts["raw_results"]:
        errors.append("completed run requires artifacts.raw_results")
    if payload["status"] in {"failed", "aborted"} and not payload.get("failure_reason"):
        errors.append("failed/aborted run requires failure_reason")

    return {"passed": not errors, "errors": errors, "warnings": warnings}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate benchmark result v0.1.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = json.loads(args.result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    if not isinstance(payload, dict):
        print("ERROR: top-level result must be an object")
        return 2
    report = validate(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"passed={report['passed']} errors={len(report['errors'])} warnings={len(report['warnings'])}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
