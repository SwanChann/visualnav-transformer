#!/usr/bin/env python3
"""Validate the frozen B0 execution overlay without importing a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SHA256 = frozenset("0123456789abcdef")


def canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256


def validate_b0_config(payload: Mapping[str, Any], repo_root: Path | None = None) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version", "config_id", "status", "execution_environment",
        "execution_performed", "actual_results", "parent_contracts", "prerequisites", "data",
        "model", "runs", "training", "checkpoint", "reporting", "execution_guard",
        "claims_forbidden",
    }
    errors.extend(f"missing:{name}" for name in sorted(required - set(payload)))
    if errors:
        return errors
    if payload["schema_version"] != "0.1.0":
        errors.append("schema_version must be 0.1.0")
    if payload["status"] != "frozen_pending_execution_authorization":
        errors.append("status must remain frozen_pending_execution_authorization")
    if payload["execution_environment"] != "rtx4090_server":
        errors.append("B0 execution environment must be rtx4090_server")
    if payload["execution_performed"] is not False or payload["actual_results"] is not None:
        errors.append("static B0 config must not contain execution results")

    parent = payload["parent_contracts"]
    data = payload["data"]
    for section, names in (
        (parent, ("smoke_config_sha256", "model_contract_sha256")),
        (data, ("manifest_sha256", "split_sha256", "dataset_tree_sha256")),
    ):
        for name in names:
            if not is_sha256(section.get(name)):
                errors.append(f"{name} must be a resolved SHA-256")
    if repo_root is not None:
        for path_key, hash_key in (
            ("smoke_config", "smoke_config_sha256"),
            ("model_contract", "model_contract_sha256"),
        ):
            path = repo_root / parent.get(path_key, "")
            if not path.is_file():
                errors.append(f"missing parent contract: {path_key}")
            elif canonical_text_sha256(path) != parent.get(hash_key):
                errors.append(f"parent contract hash mismatch: {path_key}")
        manifest = repo_root / data.get("manifest", "")
        if not manifest.is_file():
            errors.append("manifest path does not exist")
        elif canonical_text_sha256(manifest) != data.get("manifest_sha256"):
            errors.append("manifest hash mismatch")

    prerequisites = payload["prerequisites"]
    expected_prerequisites = {
        "data_audit", "canonical_batch_readiness", "image_forward_readiness",
        "train_step_readiness",
    }
    if set(prerequisites) != expected_prerequisites:
        errors.append("B0 prerequisite report set changed")
    else:
        for name in sorted(expected_prerequisites):
            receipt = prerequisites[name]
            if not is_sha256(receipt.get("sha256")):
                errors.append(f"prerequisite hash is unresolved: {name}")
            if repo_root is not None:
                report_path = repo_root / receipt.get("path", "")
                if not report_path.is_file():
                    errors.append(f"missing prerequisite report: {name}")
                elif canonical_text_sha256(report_path) != receipt.get("sha256"):
                    errors.append(f"prerequisite report hash mismatch: {name}")

    if data.get("dataset_id") != "go_stanford" or data.get("split") != "train":
        errors.append("B0 must remain the Go Stanford train pilot")
    if float(data.get("data_fraction", -1)) != 0.05 or int(data.get("subset_seed", -1)) != 0:
        errors.append("B0 data_fraction/subset_seed must remain 0.05/0")
    if int(data.get("num_workers", -1)) != 0:
        errors.append("B0 num_workers must remain zero for exact cursor replay")
    if data.get("image_size") != [96, 96]:
        errors.append("B0 image size must remain 96x96")
    if [data.get("observation_frames"), data.get("action_history_steps"), data.get("action_horizon")] != [6, 4, 8]:
        errors.append("B0 canonical frame/history/horizon contract changed")

    model = payload["model"]
    if model.get("encoder") != "efficientnet_b0_shared":
        errors.append("B0 encoder must be shared EfficientNet-B0")
    if model.get("encoder_weights") is not None or model.get("random_initialization") is not True:
        errors.append("B0 must use local random initialization")
    if model.get("pretrained_download_forbidden") is not True:
        errors.append("B0 must forbid pretrained downloads")
    if int(model.get("max_trainable_parameters", -1)) != 20_000_000:
        errors.append("B0 parameter budget changed")

    runs = payload["runs"]
    expected_runs = [
        ("h0-200", "h0_deterministic", 200),
        ("h1-200", "h1_rectified_flow", 200),
    ]
    observed_runs = [
        (run.get("id"), run.get("head"), int(run.get("optimizer_steps", -1)))
        for run in runs
    ] if isinstance(runs, list) else []
    if observed_runs != expected_runs:
        errors.append("B0 runs must be H0 200 then H1 200")

    training = payload["training"]
    if int(training.get("seed", -1)) != 0:
        errors.append("B0 seed must remain zero")
    if training.get("reset_seed_before_each_run") is not True or training.get("independent_model_per_run") is not True:
        errors.append("H0/H1 must use independent same-seed initialization")
    if training.get("run_order") != ["h0-200", "h1-200"]:
        errors.append("B0 run order changed")
    micro = int(training.get("micro_batch_size", -1))
    accumulation = int(training.get("gradient_accumulation_steps", -1))
    effective = int(training.get("effective_batch_size", -1))
    if (micro, accumulation, effective) != (4, 2, 8) or micro * accumulation != effective:
        errors.append("B0 batch/accumulation contract must remain 4x2=8")
    if int(training.get("total_optimizer_steps_if_authorized", -1)) != 400:
        errors.append("B0 total optimizer step budget must be 400")
    if int(training.get("total_backward_calls_if_authorized", -1)) != 800:
        errors.append("B0 total backward call budget must be 800")
    optimizer = training.get("optimizer", {})
    if optimizer != {
        "name": "adamw", "learning_rate": 0.0001, "weight_decay": 0.01,
        "betas": [0.9, 0.999], "eps": 1e-08,
    }:
        errors.append("B0 AdamW hyperparameters changed")
    if training.get("scheduler") is not None:
        errors.append("B0 scheduler must remain disabled")
    precision = training.get("precision", {})
    if precision != {
        "autocast": True, "autocast_dtype": "float16", "grad_scaler": True,
        "init_scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5,
        "growth_interval": 2000,
    }:
        errors.append("B0 FP16/GradScaler contract changed")
    gradients = training.get("gradients", {})
    if gradients != {
        "zero_grad_set_to_none": True, "max_l2_norm": 10.0,
        "non_finite_policy": "fail_closed_no_step",
    }:
        errors.append("B0 gradient policy changed")
    ema = training.get("ema", {})
    if ema != {
        "enabled": True, "decay": 0.999, "update_after_optimizer_step": 0,
        "update_every_optimizer_steps": 1, "include_trainable_parameters_only": True,
    }:
        errors.append("B0 EMA contract changed")

    checkpoint = payload["checkpoint"]
    if int(checkpoint.get("save_every_optimizer_steps", -1)) != 50:
        errors.append("B0 checkpoint interval must remain 50")
    if int(checkpoint.get("keep_last", -1)) != 3:
        errors.append("B0 checkpoint keep_last must remain 3")
    if int(checkpoint.get("exact_resume_probe_optimizer_step", -1)) != 100:
        errors.append("B0 exact-resume probe must remain at step 100")
    if int(checkpoint.get("exact_resume_probe_additional_optimizer_steps", -1)) != 0:
        errors.append("B0 resume probe cannot add optimizer steps")
    required_state = set(checkpoint.get("required_state", []))
    if not {"model", "optimizer", "grad_scaler", "ema", "python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng_all_visible_devices", "deterministic_batch_cursor", "exposure_counters", "optimizer_step"} <= required_state:
        errors.append("B0 checkpoint state is incomplete")

    guard = payload["execution_guard"]
    if guard.get("required_cli_mode") != "execute":
        errors.append("B0 execution must require --execute")
    if guard.get("required_authorization_token") != "RUN-B0-H0-200-H1-200":
        errors.append("B0 authorization token changed")
    if guard.get("required_branch") != "agent/ubuntu-sim-handoff":
        errors.append("B0 required branch changed")
    if guard.get("require_tracked_worktree_clean") is not True:
        errors.append("B0 must require a clean tracked worktree")
    if guard.get("require_head_equals_origin_branch") is not True:
        errors.append("B0 must require HEAD to equal the local origin branch ref")
    if guard.get("dry_run_is_non_training") is not True:
        errors.append("B0 dry-run must remain non-training")
    if guard.get("authorization_granted_in_this_config") is not False:
        errors.append("static config cannot grant B0 execution")
    return errors


def summary(payload: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    training = payload["training"]
    return {
        "config_id": payload["config_id"],
        "status": payload["status"],
        "config_sha256": canonical_text_sha256(config_path),
        "runs": [
            {"id": run["id"], "head": run["head"], "optimizer_steps": run["optimizer_steps"]}
            for run in payload["runs"]
        ],
        "total_optimizer_steps_if_authorized": training["total_optimizer_steps_if_authorized"],
        "total_backward_calls_if_authorized": training["total_backward_calls_if_authorized"],
        "effective_batch_size": training["effective_batch_size"],
        "execution_performed": payload["execution_performed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        errors = validate_b0_config(payload, args.repo_root)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2
    result = summary(payload, args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else f"B0 config valid: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
