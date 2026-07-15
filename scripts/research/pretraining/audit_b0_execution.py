#!/usr/bin/env python3
"""Audit a completed B0 JSON report and local checkpoint receipts read-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def audit(report: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    if report.get("schema_version") != "0.1.0":
        raise ValueError("unexpected B0 report schema")
    config_path = repo_root / "scripts/research/pretraining/b0_execution_config_v0.1.yaml"
    runner_path = repo_root / "scripts/research/pretraining/run_tinynav_b0.py"
    config_exact = canonical_text_sha256(config_path) == report["config_sha256"]
    runner_exact = file_sha256(runner_path) == report["runner_sha256"]
    boundary = report["execution_boundary"]
    run_audits: list[dict[str, Any]] = []
    all_checkpoint_receipts_exact = True
    total_checkpoint_bytes = 0
    for run in report["runs"]:
        records = run["step_records"]
        losses = [float(record["action_loss_mean"]) for record in records]
        gradients = [float(record["gradient_l2_norm_preclip"]) for record in records]
        timings = [float(record["elapsed_ms_including_data_excluding_checkpoint"]) for record in records]
        finite = all(math.isfinite(value) for value in losses + gradients + timings)
        scaler_no_skip = all(
            float(record["grad_scaler_scale_after"])
            >= float(record["grad_scaler_scale_before"])
            for record in records
        )
        receipts = []
        for checkpoint in run["retained_checkpoints"]:
            path = repo_root / checkpoint["path"]
            exact = (
                path.is_file()
                and path.stat().st_size == int(checkpoint["size_bytes"])
                and file_sha256(path) == checkpoint["sha256"]
            )
            all_checkpoint_receipts_exact &= exact
            total_checkpoint_bytes += path.stat().st_size if path.is_file() else 0
            receipts.append({**checkpoint, "receipt_exact": exact})
        steady_timings = timings[10:]
        identity = [item for record in records for item in record["batch_identity"]]
        run_audits.append({
            "run_id": run["run_id"],
            "head": run["head"],
            "optimizer_steps": run["optimizer_steps"],
            "backward_calls": run["backward_calls"],
            "examples_seen": run["examples_seen"],
            "step_records": len(records),
            "step_sequence_exact": [record["optimizer_step"] for record in records] == list(range(1, 201)),
            "all_recorded_values_finite": finite,
            "grad_scaler_no_skip": scaler_no_skip,
            "loss_range_observed_not_convergence": [min(losses), max(losses)],
            "gradient_norm_range_preclip": [min(gradients), max(gradients)],
            "steady_step_timing_ms_median": statistics.median(steady_timings),
            "steady_step_timing_ms_p95": percentile(steady_timings, 0.95),
            "effective_examples_per_second_from_median": 8000.0 / statistics.median(steady_timings),
            "peak_cuda_memory_bytes_training_steps": run["peak_cuda_memory_bytes_training_steps"],
            "target_max_abs_m": run["target_max_abs_m"],
            "sample_identity_count": len(identity),
            "sample_identity_unique_count": len(set(identity)),
            "exact_resume_gates": run["resume_probe"]["gates"],
            "retained_checkpoints": receipts,
        })
    gates = {
        "config_hash_exact": config_exact,
        "runner_hash_exact": runner_exact,
        "execution_budget_exact": (
            boundary["optimizer_steps"] == 400 and boundary["backward_calls"] == 800
        ),
        "no_evaluation": boundary["evaluation_performed"] is False,
        "no_pretrained_download": boundary["pretrained_weights_downloaded"] is False,
        "two_runs_present": [run["run_id"] for run in run_audits] == ["h0-200", "h1-200"],
        "each_run_budget_exact": all(
            run["optimizer_steps"] == 200
            and run["backward_calls"] == 400
            and run["examples_seen"] == 1600
            and run["step_records"] == 200
            for run in run_audits
        ),
        "all_values_finite": all(run["all_recorded_values_finite"] for run in run_audits),
        "no_grad_scaler_skip": all(run["grad_scaler_no_skip"] for run in run_audits),
        "all_exact_resume_gates_pass": all(
            all(run["exact_resume_gates"].values()) for run in run_audits
        ),
        "all_samples_unique_within_each_run": all(
            run["sample_identity_count"] == run["sample_identity_unique_count"] == 1600
            for run in run_audits
        ),
        "meter_targets_finite_and_under_10m": all(
            math.isfinite(run["target_max_abs_m"]) and 0 < run["target_max_abs_m"] < 10
            for run in run_audits
        ),
        "checkpoint_retention_exact": all(
            [Path(item["path"]).stem for item in run["retained_checkpoints"]]
            == ["step-100", "step-150", "step-200"]
            for run in run_audits
        ),
        "all_checkpoint_receipts_exact": all_checkpoint_receipts_exact,
        "checkpoint_space_under_authorized_1gib": total_checkpoint_bytes <= 1024 ** 3,
    }
    if not all(gates.values()):
        raise ValueError(f"B0 audit failed: {gates}")
    return {
        "schema_version": "0.1.0",
        "audit_id": "tinynavbrain-go-stanford-b0-v0.1-audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report_sha256": file_sha256(
            repo_root / "results/research/pretraining/b0_go_stanford_v0.1/b0_execution.json"
        ),
        "config_sha256": report["config_sha256"],
        "runner_sha256": report["runner_sha256"],
        "git": report["git"],
        "execution_boundary": boundary,
        "runs": run_audits,
        "total_checkpoint_bytes": total_checkpoint_bytes,
        "gates": gates,
        "passed": True,
        "claim_boundary": (
            "This audit establishes B0 plumbing, finite recorded values, exact resume, local "
            "checkpoint receipts, and scoped smoke timing/memory only. Loss ranges are not "
            "convergence or model-quality evidence; the <10 m check is not independent physical calibration."
        ),
    }


def write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# TinyNavBrain Go Stanford B0 Audit",
        "",
        f"- Passed: {result['passed']}",
        f"- Git: `{result['git']['commit']}`",
        f"- Optimizer steps / backward calls: {result['execution_boundary']['optimizer_steps']} / {result['execution_boundary']['backward_calls']}",
        f"- Local checkpoint bytes: {result['total_checkpoint_bytes']}",
        "- Evaluation performed: False",
        "- Pretrained weights downloaded: False",
        "",
        "## Runs",
        "",
    ]
    for run in result["runs"]:
        lines.extend([
            f"### {run['run_id']}",
            "",
            f"- Steps/backward/examples: {run['optimizer_steps']} / {run['backward_calls']} / {run['examples_seen']}",
            f"- Finite / GradScaler no-skip / exact-resume: {run['all_recorded_values_finite']} / {run['grad_scaler_no_skip']} / {all(run['exact_resume_gates'].values())}",
            f"- Scoped steady step median/p95: {run['steady_step_timing_ms_median']:.3f} / {run['steady_step_timing_ms_p95']:.3f} ms",
            f"- Scoped effective examples/s: {run['effective_examples_per_second_from_median']:.3f}",
            f"- Training-step peak CUDA memory: {run['peak_cuda_memory_bytes_training_steps'] / (1024 ** 2):.2f} MiB",
            "",
        ])
    lines.extend(["## Gates", ""])
    lines.extend(f"- {name}: {value}" for name, value in result["gates"].items())
    lines.extend(["", "## Evidence boundary", "", result["claim_boundary"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        result = audit(report, args.repo_root)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print("B0 audit passed; no model execution performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
