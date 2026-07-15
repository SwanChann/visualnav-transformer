#!/usr/bin/env python3
"""Static validator and conservative memory-accounting aid; never trains a model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


ALLOWED_NFE = {1, 2, 4, 8}


def validate(payload: dict) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version", "plan_id", "execution", "hardware", "model", "data",
        "training", "phases", "evaluation",
    )
    for key in required:
        if key not in payload:
            errors.append(f"missing:{key}")
    if errors:
        return errors
    model = payload["model"]
    execution = payload["execution"]
    if execution.get("environment") != "rtx4090_server":
        errors.append("execution.environment must be rtx4090_server")
    if execution.get("data_residency") != "rtx4090_server_only":
        errors.append("execution.data_residency must be rtx4090_server_only")
    if execution.get("server_inventory_status") != "unverified_requires_snapshot":
        errors.append("execution.server_inventory_status must require a live snapshot")
    if int(payload["hardware"].get("gpu_count", 0)) != 1:
        errors.append("hardware.gpu_count must equal 1")
    if float(payload["hardware"].get("vram_gib", 0)) != 24:
        errors.append("hardware.vram_gib must equal 24")
    if model.get("action_units") != "meters":
        errors.append("model.action_units must be meters")
    if not set(map(int, model.get("allowed_nfe", []))).issubset(ALLOWED_NFE):
        errors.append("model.allowed_nfe contains unsupported values")
    if int(model.get("max_trainable_parameters", 0)) > 20_000_000:
        errors.append("model.max_trainable_parameters exceeds 20M")
    if int(model.get("observation_frames", 0)) != 6:
        errors.append("model.observation_frames must equal the frozen contract value 6")
    if model.get("physical_tokens") != ["metric_waypoint_spacing_m", "observation_dt_s"]:
        errors.append("model.physical_tokens must exclude dataset or embodiment identity")
    if model.get("rectified_flow_head") is not True or "residual_flow_head" in model:
        errors.append("model must use the frozen rectified_flow_head name")
    phase_ids = [phase.get("id") for phase in payload["phases"]]
    if phase_ids != ["static", "smoke", "baseline", "ablation"]:
        errors.append("phases must be ordered static, smoke, baseline, ablation")
    for phase in payload["phases"]:
        steps = int(phase.get("optimizer_steps", -1))
        if phase.get("id") == "static" and steps != 0:
            errors.append("static phase must use zero optimizer steps")
        if phase.get("id") != "static" and steps <= 0:
            errors.append(f"{phase.get('id')} optimizer_steps must be positive")
    if int(payload["data"].get("required_before_cross_dataset_claim", 0)) < 3:
        errors.append("cross-dataset claim requires at least three materialized datasets")
    if payload["data"].get("split_unit") != "trajectory":
        errors.append("data split must be trajectory-level")
    if int(payload["data"].get("leakage_tolerance", -1)) != 0:
        errors.append("data leakage tolerance must be zero")
    return errors


def analytic_budget(payload: dict) -> dict:
    """Return labeled estimates, not measured CUDA memory or training time."""
    model = payload["model"]
    max_params = int(model["max_trainable_parameters"])
    bytes_weights_fp16 = max_params * 2
    bytes_master_and_adam = max_params * (4 + 8)
    d_model = int(model["d_model"])
    tokens = int(model["observation_frames"]) + int(model["action_history_steps"]) + 3
    largest_batch = max(int(p["batch_size"]) for p in payload["phases"])
    transformer_activation_floor = largest_batch * tokens * d_model * int(model["fusion_layers"]) * 2
    conservative_multiplier = 12
    estimated_bytes = bytes_weights_fp16 + bytes_master_and_adam + transformer_activation_floor * conservative_multiplier
    return {
        "kind": "analytic_partial-accounting_not_measured",
        "assumed_parameter_cap": max_params,
        "largest_configured_batch": largest_batch,
        "estimated_partial_accounted_gib": estimated_bytes / (1024 ** 3),
        "warning": "Encoder feature maps, allocator fragmentation, dataloader buffers and library workspaces are not modeled; run a 4090 smoke profile before increasing batch size.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    payload = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    errors = validate(payload)
    print(json.dumps({"passed": not errors, "errors": errors, "budget": analytic_budget(payload)}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
