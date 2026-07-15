"""Checkpoint provenance and resume compatibility metadata contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
IMMUTABLE_RESUME_FIELDS = (
    "head",
    "seed",
    "git_commit",
    "manifest_sha256",
    "split_sha256",
    "model_contract_sha256",
    "config_sha256",
)


def validate_checkpoint_metadata(
    payload: Mapping[str, Any], expected_resume: Mapping[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "run_id",
        "phase",
        "head",
        "seed",
        "optimizer_step",
        "git_commit",
        "manifest_sha256",
        "split_sha256",
        "model_contract_sha256",
        "config_sha256",
        "rng_state_present",
        "sampler_state_present",
    }
    for key in sorted(required - set(payload)):
        errors.append(f"missing:{key}")
    if errors:
        return errors
    if payload["schema_version"] != "0.1.0":
        errors.append("schema_version must be 0.1.0")
    if payload["head"] not in {"h0_deterministic", "h1_rectified_flow", "nomad"}:
        errors.append("head is unsupported")
    if payload["phase"] not in {"smoke", "baseline", "ablation"}:
        errors.append("phase is unsupported")
    if int(payload["seed"]) < 0 or int(payload["optimizer_step"]) < 0:
        errors.append("seed and optimizer_step must be non-negative")
    if not GIT_SHA.fullmatch(str(payload["git_commit"])):
        errors.append("git_commit must be a full lowercase Git SHA")
    for key in ("manifest_sha256", "split_sha256", "model_contract_sha256", "config_sha256"):
        if not SHA256.fullmatch(str(payload[key])):
            errors.append(f"{key} must be a resolved lowercase SHA-256")
    if payload["rng_state_present"] is not True:
        errors.append("rng_state_present must be true")
    if payload["sampler_state_present"] is not True:
        errors.append("sampler_state_present must be true")
    if expected_resume is not None:
        for key in IMMUTABLE_RESUME_FIELDS:
            if key in expected_resume and payload.get(key) != expected_resume.get(key):
                errors.append(f"resume_mismatch:{key}")
    return errors
