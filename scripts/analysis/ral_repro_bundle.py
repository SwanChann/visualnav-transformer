#!/usr/bin/env python3
"""Build a checksum-bound reproducibility manifest for controlled RA-L evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SECRET_PATTERNS = [
    re.compile(r"AKLT[A-Za-z0-9]{12,}"),
    re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]+['\"]"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def scan_secrets(paths: list[Path]) -> list[str]:
    findings = []
    for path in paths:
        if path.suffix.lower() not in {".py", ".json", ".md", ".csv", ".txt", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            findings.append(relative(path))
    return findings


def build_manifest(run_dir: Path, analysis_dir: Path, protocol: Path) -> dict[str, Any]:
    fixed = [
        protocol,
        REPO_ROOT / "后续研究内容/benchmark/ral_mujoco_protocol_v0.1.json",
        REPO_ROOT / "后续研究内容/benchmark/ral_mujoco_calibration_v0.3.json",
        REPO_ROOT / "后续研究内容/benchmark/ral_mujoco_trial_schema_v0.1.json",
        REPO_ROOT / "scripts/experiments/ral_mujoco_controlled.py",
        REPO_ROOT / "scripts/analysis/validate_ral_mujoco_trial.py",
        REPO_ROOT / "scripts/analysis/ral_controlled_results.py",
        REPO_ROOT / "scripts/analysis/ral_controlled_figures.py",
        REPO_ROOT / "scripts/analysis/test_legacy_pd_source.py",
        REPO_ROOT / "deployment/README.md",
        REPO_ROOT / "deployment/src/pd_controller.py",
        REPO_ROOT / "results/research/ubuntu_environment/pip_freeze.txt",
        run_dir / "REPRODUCE.md",
    ]
    raw_trials = sorted((run_dir / "trials").glob("*.json"))
    run_files = [run_dir / "batch_manifest.json", run_dir / "selected_plan.json"]
    analysis_files = sorted(path for path in analysis_dir.rglob("*") if path.is_file())
    figure_files = sorted(path for path in (REPO_ROOT / "投稿冲刺/workspace/controlled_results").glob("*") if path.is_file())
    paper_files = [
        REPO_ROOT / "投稿冲刺/workspace/paper_package/01_title_abstract_contributions.md",
        REPO_ROOT / "投稿冲刺/workspace/paper_package/02_method_results_skeleton.md",
        REPO_ROOT / "投稿冲刺/workspace/paper_package/03_figures_tables_captions.md",
        REPO_ROOT / "投稿冲刺/workspace/paper_package/04_evidence_inclusion_rules.md",
        REPO_ROOT / "投稿冲刺/workspace/paper_package/05_reviewer_checklist.md",
        REPO_ROOT / "投稿冲刺/workspace/paper_package/claim_evidence_matrix.csv",
        REPO_ROOT / "投稿冲刺/workspace/paper_package/claim_evidence_matrix.md",
    ]
    repeat_files = sorted(path for path in (REPO_ROOT / "results/research/ral_mujoco_controlled/u09-v03-repeat-audit-20260711").rglob("*") if path.is_file())
    offline_files = sorted(path for path in (REPO_ROOT / "results/research/ral_offline_independent/u10-independent-20260711-v5").glob("*") if path.is_file())
    invalidation_files = [
        REPO_ROOT / "results/research/ral_offline_independent/u10-independent-20260711-v2/INVALIDATED.md",
        REPO_ROOT / "results/research/ral_mujoco_controlled/u09-v02-controlled-20260711/INVALIDATED_AS_PRIMARY.md",
    ]
    backend_files = sorted(path for path in (REPO_ROOT / "scripts/research/policy_backend").glob("*") if path.is_file())
    training_plan_files = sorted(path for path in (REPO_ROOT / "scripts/research/training_plans").glob("*") if path.is_file())
    backend_results = sorted(path for path in (REPO_ROOT / "results/research/policy_backend_exploration").glob("*") if path.is_file())
    research_files = [REPO_ROOT / "后续研究内容/05_policy后端与小模型研究.md"]
    files = fixed + run_files + raw_trials + analysis_files + figure_files + paper_files + repeat_files + offline_files + invalidation_files + backend_files + training_plan_files + backend_results + research_files
    missing = [relative(path) if path.is_absolute() and str(path).startswith(str(REPO_ROOT)) else str(path) for path in files if not path.is_file()]
    present = [path for path in files if path.is_file()]
    entries = [{"path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in present]
    return {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_id": run_dir.name,
        "protocol_sha256": sha256_file(protocol),
        "raw_trial_count": len(raw_trials),
        "file_count": len(entries),
        "missing_files": missing,
        "secret_scan_findings": scan_secrets(present),
        "files": entries,
        "passed": not missing and not scan_secrets(present),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args.run_dir.resolve(), args.analysis_dir.resolve(), args.protocol.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("run_id", "raw_trial_count", "file_count", "missing_files", "secret_scan_findings", "passed")}, indent=2))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
