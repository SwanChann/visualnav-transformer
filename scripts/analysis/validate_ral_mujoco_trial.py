#!/usr/bin/env python3
"""Validate controlled MuJoCo trial records and batch manifests."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")

TRIAL_REQUIRED = {
    "$": {"schema_version", "record_kind", "run_id", "trial_id", "trial_key", "attempt", "created_at_utc", "completed_at_utc", "status", "git", "inputs", "environment", "episode", "policy", "bridge", "outcome", "trajectory", "timing", "artifacts"},
    "git": {"commit", "dirty", "diff_sha256"},
    "inputs": {"protocol_path", "protocol_sha256", "checkpoint_path", "checkpoint_sha256", "policy_config_path", "policy_config_sha256", "topomap_path", "topomap_tree_sha256", "scene_definition_path", "scene_definition_sha256", "mjcf_path", "mjcf_sha256", "locomotion_policy_path", "locomotion_policy_sha256"},
    "environment": {"device", "gpu", "precision", "python_version", "torch_version", "simulator_version", "renderer", "onnxruntime_version", "onnx_provider"},
    "episode": {"map", "spawn_xy_m", "goal_xy_m", "diffusion_seed", "goal_seed", "stabilizer_mode", "success_radius_m", "max_steps", "simulated_timeout_s"},
    "policy": {"encoder", "scheduler", "diffusion_steps", "cfg_weight", "tts_enabled", "tts_budget", "tts_topk", "tts_verifier", "candidate_k", "selection", "image_resize_mode", "image_size_px", "context_frames", "action_horizon", "waypoint_index"},
    "bridge": {"type", "yaw_sign", "linear_scale", "yaw_scale", "pd_max_v", "pd_max_w", "bridge_max_v_mps", "bridge_max_w_radps"},
    "outcome": {"success", "final_distance_m", "path_length_m", "shortest_path_m", "spl", "collision", "fall", "stuck", "timeout", "intervention", "recovery_count", "exception", "exit_code"},
    "timing": {"warmup_calls", "cuda_synchronized", "sampler_per_call_ms", "full_loop_per_call_ms"},
    "artifacts": {"trial_json", "runtime_dir", "timing_csv"},
}

BATCH_REQUIRED = {"schema_version", "record_kind", "run_id", "created_at_utc", "protocol_sha256", "planned_trial_count", "completed", "failed", "crashed", "infrastructure_invalid", "missing_trial_keys", "trial_records"}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _need(obj: Any, fields: set[str], location: str, errors: list[str]) -> bool:
    if not isinstance(obj, dict):
        errors.append(f"{location} must be an object")
        return False
    for field in sorted(fields - set(obj)):
        errors.append(f"{location}: missing {field}")
    return True


def validate_trial(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not _need(payload, TRIAL_REQUIRED["$"], "$", errors):
        return {"passed": False, "errors": errors, "warnings": warnings}
    for location, fields in TRIAL_REQUIRED.items():
        if location == "$":
            continue
        _need(payload.get(location), fields, location, errors)
    if errors:
        return {"passed": False, "errors": errors, "warnings": warnings}

    if payload["schema_version"] != "0.1.0" or payload["record_kind"] != "trial":
        errors.append("record must be schema_version=0.1.0 and record_kind=trial")
    if payload["status"] not in {"completed", "failed", "crashed", "infrastructure_invalid"}:
        errors.append("invalid status")
    if not isinstance(payload["attempt"], int) or payload["attempt"] < 1:
        errors.append("attempt must be a positive integer")

    git = payload["git"]
    if not COMMIT.fullmatch(str(git["commit"])):
        errors.append("git.commit must be 40 lowercase hex")
    if not isinstance(git["dirty"], bool):
        errors.append("git.dirty must be boolean")
    if git["dirty"] and not SHA256.fullmatch(str(git["diff_sha256"] or "")):
        errors.append("dirty trial requires git.diff_sha256")
    if not git["dirty"] and git["diff_sha256"] is not None:
        warnings.append("clean trial has a diff hash")

    for field, value in payload["inputs"].items():
        if field.endswith("sha256") and not SHA256.fullmatch(str(value)):
            errors.append(f"inputs.{field} must be SHA-256 hex")

    episode = payload["episode"]
    if episode["map"] not in {"easy", "medium", "hard"}:
        errors.append("invalid episode.map")
    if episode["stabilizer_mode"] not in {"policy_only_off", "system_stabilizer_on"}:
        errors.append("invalid stabilizer_mode")
    for field in ("spawn_xy_m", "goal_xy_m"):
        value = episode[field]
        if not isinstance(value, list) or len(value) != 2 or not all(_finite(item) for item in value):
            errors.append(f"episode.{field} must be finite [x,y]")

    policy = payload["policy"]
    if policy["candidate_k"] != 8:
        errors.append("frozen protocol requires candidate_k=8")
    if policy["tts_enabled"]:
        if (policy["tts_budget"], policy["tts_topk"], policy["tts_verifier"]) != (8, 1, "heuristic"):
            errors.append("frozen TTS requires budget=8, topk=1, verifier=heuristic")
    elif any(policy[field] is not None for field in ("tts_budget", "tts_topk", "tts_verifier")):
        errors.append("TTS-off trial must use null TTS fields")

    outcome = payload["outcome"]
    for field in ("success", "collision", "fall", "stuck", "timeout", "intervention"):
        if not isinstance(outcome[field], bool):
            errors.append(f"outcome.{field} must be boolean")
    if outcome["success"] and (outcome["fall"] or outcome["timeout"]):
        errors.append("success cannot coexist with terminal fall or timeout")
    if outcome["success"] and outcome["final_distance_m"] is not None and outcome["final_distance_m"] >= episode["success_radius_m"]:
        errors.append("success final distance violates strict radius")
    for field in ("path_length_m", "shortest_path_m", "spl"):
        if not _finite(outcome[field]) or outcome[field] < 0:
            errors.append(f"outcome.{field} must be finite and nonnegative")
    if _finite(outcome["spl"]) and not 0 <= outcome["spl"] <= 1:
        errors.append("outcome.spl must be in [0,1]")
    if payload["status"] in {"crashed", "infrastructure_invalid"} and not outcome["exception"]:
        errors.append("crashed/infrastructure_invalid trial requires outcome.exception")

    trajectory = payload["trajectory"]
    if not isinstance(trajectory, list):
        errors.append("trajectory must be an array")
    else:
        for index, point in enumerate(trajectory):
            if not isinstance(point, dict) or set(point) != {"tick", "x_m", "y_m"}:
                errors.append(f"trajectory[{index}] has invalid fields")
                break
            if not isinstance(point["tick"], int) or not _finite(point["x_m"]) or not _finite(point["y_m"]):
                errors.append(f"trajectory[{index}] has invalid values")
                break

    timing = payload["timing"]
    for field in ("sampler_per_call_ms", "full_loop_per_call_ms"):
        values = timing[field]
        if not isinstance(values, list) or not all(_finite(value) and value >= 0 for value in values):
            errors.append(f"timing.{field} must be a nonnegative finite array")
    if payload["status"] in {"completed", "failed"}:
        if not trajectory:
            errors.append("executed trial requires non-empty trajectory")
        if not timing["full_loop_per_call_ms"]:
            errors.append("executed trial requires full-loop timing")

    return {"passed": not errors, "errors": errors, "warnings": warnings}


def validate_batch(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not _need(payload, BATCH_REQUIRED, "$", errors):
        return {"passed": False, "errors": errors, "warnings": warnings}
    if payload.get("schema_version") != "0.1.0" or payload.get("record_kind") != "batch_manifest":
        errors.append("record must be schema_version=0.1.0 and record_kind=batch_manifest")
    if not SHA256.fullmatch(str(payload.get("protocol_sha256", ""))):
        errors.append("protocol_sha256 must be SHA-256 hex")
    counts = [payload.get(name) for name in ("completed", "failed", "crashed", "infrastructure_invalid")]
    if not all(isinstance(value, int) and value >= 0 for value in counts):
        errors.append("batch status counts must be nonnegative integers")
    elif sum(counts) + len(payload.get("missing_trial_keys", [])) != payload.get("planned_trial_count"):
        errors.append("planned count does not reconcile with statuses plus missing keys")
    records = payload.get("trial_records")
    if not isinstance(records, list):
        errors.append("trial_records must be an array")
    elif all(isinstance(value, int) for value in counts) and len(records) != sum(counts):
        errors.append("trial_records count does not reconcile with status counts")
    return {"passed": not errors, "errors": errors, "warnings": warnings}


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("record_kind") == "batch_manifest":
        return validate_batch(payload)
    return validate_trial(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        print("ERROR: top-level record must be an object")
        return 2
    report = validate(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"passed={report['passed']} errors={len(report['errors'])} warnings={len(report['warnings'])}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
