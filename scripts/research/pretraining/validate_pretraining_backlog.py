#!/usr/bin/env python3
"""Validate the no-training backlog and 200-step smoke configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ALLOWED_STATUS = {"ready", "in_progress", "blocked_external", "done", "stopped"}
ALLOWED_ENVIRONMENTS = {"windows", "rtx4090_server", "ubuntu", "target_device_tbd"}
EXPECTED_STAGE_ENVIRONMENT = {
    "TRAIN-INFRA": "windows",
    "DATA-PILOT": "rtx4090_server",
    "B0": "rtx4090_server",
    "BASELINE": "rtx4090_server",
    "H1": "rtx4090_server",
    "OFFLINE-GATE": "rtx4090_server",
    "SIM-GATE": "ubuntu",
    "DEVICE-ROBOT-GATE": "target_device_tbd",
}


def validate_backlog(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != "0.1.0":
        errors.append("schema_version must be 0.1.0")
    if payload.get("training_performed") is not False:
        errors.append("training_performed must be false in the pre-execution backlog")
    if payload.get("data_residency") != "rtx4090_server_only":
        errors.append("data_residency must be rtx4090_server_only")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return errors + ["tasks must be a non-empty list"]
    ids = [task.get("id") for task in tasks]
    if any(not task_id for task_id in ids) or len(ids) != len(set(ids)):
        errors.append("task ids must be unique and non-empty")
    known_ids = set(ids)
    graph: dict[str, list[str]] = {}
    for task in tasks:
        task_id = task.get("id", "<missing>")
        status = task.get("status")
        if status not in ALLOWED_STATUS:
            errors.append(f"{task_id}:invalid status")
        environment = task.get("execution_environment")
        if environment not in ALLOWED_ENVIRONMENTS:
            errors.append(f"{task_id}:invalid execution_environment")
        expected_environment = EXPECTED_STAGE_ENVIRONMENT.get(task.get("stage"))
        if expected_environment is not None and environment != expected_environment:
            errors.append(
                f"{task_id}:stage {task.get('stage')} must run on {expected_environment}"
            )
        dependencies = task.get("depends_on", [])
        graph[task_id] = dependencies
        for dependency in dependencies:
            if dependency not in known_ids:
                errors.append(f"{task_id}:unknown dependency:{dependency}")
        if not task.get("acceptance"):
            errors.append(f"{task_id}:missing acceptance")
        if not task.get("falsification"):
            errors.append(f"{task_id}:missing falsification")
        if status == "done" and not task.get("evidence"):
            errors.append(f"{task_id}:done task needs evidence")
        if status == "done" and task.get("requires_training") is True:
            errors.append(f"{task_id}:training task cannot be done before execution")
        if task.get("external") is True and status not in {"blocked_external", "stopped"}:
            errors.append(f"{task_id}:external task must remain blocked_external or stopped")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            errors.append(f"dependency_cycle:{task_id}")
            return
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in graph.get(task_id, []):
            if dependency in graph:
                visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in graph:
        visit(task_id)
    return errors


def validate_smoke_config(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != "0.1.0":
        errors.append("smoke schema_version must be 0.1.0")
    if payload.get("status") != "planned_no_training_executed":
        errors.append("smoke status must remain planned_no_training_executed")
    if payload.get("execution_environment") != "rtx4090_server":
        errors.append("smoke execution_environment must be rtx4090_server")
    if payload.get("data_residency") != "rtx4090_server_only":
        errors.append("smoke data_residency must be rtx4090_server_only")
    if int(payload.get("optimizer_steps", -1)) != 200:
        errors.append("smoke optimizer_steps must equal 200")
    if payload.get("actual_results") not in (None, {}):
        errors.append("smoke actual_results must be empty before execution")
    if payload.get("heads") != ["h0_deterministic", "h1_rectified_flow"]:
        errors.append("smoke must cover H0 then H1")
    required = set(payload.get("required_provenance", []))
    expected = {
        "git_commit",
        "manifest_sha256",
        "split_sha256",
        "model_contract_sha256",
        "config_sha256",
        "rng_state",
        "sampler_state",
    }
    if required != expected:
        errors.append("smoke required_provenance is incomplete")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backlog", type=Path)
    parser.add_argument("--smoke-config", type=Path)
    args = parser.parse_args()
    backlog = yaml.safe_load(args.backlog.read_text(encoding="utf-8"))
    errors = validate_backlog(backlog)
    if args.smoke_config:
        smoke = yaml.safe_load(args.smoke_config.read_text(encoding="utf-8"))
        errors.extend(validate_smoke_config(smoke))
    print(json.dumps({"passed": not errors, "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
