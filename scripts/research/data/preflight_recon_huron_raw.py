#!/usr/bin/env python3
"""Build a read-only RECON or HuRoN raw-data conversion preflight.

The tool inventories candidate raw files and renders the exact legacy processor
command that a later, separately authorized conversion could use.  It never
imports or invokes either processor and never writes below the raw or processed
dataset roots.  The only optional writes are JSON/Markdown readiness reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATASET_SPECS: dict[str, dict[str, Any]] = {
    "recon": {
        "raw_suffixes": (".h5", ".hdf5"),
        "required_child": "recon_release",
        "processor": "train/process_recon.py",
        "processor_config": None,
        "conversion_args": (),
    },
    "huron_sacson": {
        "raw_suffixes": (".bag",),
        "required_child": None,
        "processor": "train/process_bags.py",
        "processor_config": "train/vint_train/process_data/process_bags_config.yaml",
        "conversion_args": ("--dataset-name", "sacson"),
    },
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def git_context(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": run("rev-parse", "HEAD"),
            "tracked_worktree_dirty": bool(
                run("status", "--porcelain=v1", "--untracked-files=no")
            ),
            "worktree_dirty_including_preserved_untracked": bool(
                run("status", "--porcelain=v1")
            ),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "branch": None,
            "commit": None,
            "tracked_worktree_dirty": None,
            "worktree_dirty_including_preserved_untracked": None,
        }


def load_registry_entry(path: Path, dataset_id: str) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        item for item in payload.get("datasets", [])
        if item.get("dataset_id") == dataset_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"dataset_id must match exactly one registry entry: {dataset_id}"
        )
    return matches[0], sha256_file(path)


def planned_source_id(dataset_id: str, path: Path) -> str:
    if dataset_id == "recon":
        return path.stem
    return f"{path.parent.name}_{path.stem}"


def discover_candidates(
    dataset_id: str,
    input_root: Path,
    checksum_mode: str,
) -> tuple[Path, list[dict[str, Any]]]:
    spec = DATASET_SPECS[dataset_id]
    scan_root = (
        input_root / spec["required_child"]
        if spec["required_child"]
        else input_root
    )
    if not scan_root.is_dir():
        return scan_root, []
    suffixes = set(spec["raw_suffixes"])
    paths = sorted(
        path for path in scan_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    candidates: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        candidates.append(
            {
                "relative_path": path.relative_to(input_root).as_posix(),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "planned_source_id": planned_source_id(dataset_id, path),
                "sha256": sha256_file(path) if checksum_mode == "sha256" else None,
            }
        )
    return scan_root, candidates


def validate_receipt(
    receipt_path: Path | None,
    license_snapshot_path: Path | None,
    repo_root: Path,
    dataset_id: str,
    registry_sha256: str,
    registry_entry: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(receipt_path) if receipt_path else None,
        "present": bool(receipt_path and receipt_path.is_file()),
        "schema_and_registry_binding_valid": False,
        "license_snapshot_path": (
            str(license_snapshot_path) if license_snapshot_path else None
        ),
        "license_snapshot_present": bool(
            license_snapshot_path and license_snapshot_path.is_file()
        ),
        "license_snapshot_hash_matches": False,
        "artifact_checksum_reverified": False,
        "errors": [],
    }
    if not result["present"]:
        result["errors"].append("raw artifact receipt is missing")
        return result
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["errors"].append(f"cannot read receipt: {exc}")
        return result

    artifact = receipt.get("artifact", {})
    license_record = receipt.get("license", {})
    source_url = receipt.get("source_url", "")
    parsed_source = urlparse(source_url)
    official_urls = [
        registry_entry.get("official_download"),
        registry_entry.get("official_page"),
    ]
    official_hosts = {
        urlparse(url).netloc.lower() for url in official_urls if url
    }
    checks = {
        "not_a_template": receipt.get("_template_only") is not True,
        "schema_version": receipt.get("schema_version") == "0.1.0",
        "dataset_id": receipt.get("dataset_id") == dataset_id,
        "registry_sha256": receipt.get("registry_sha256") == registry_sha256,
        "registry_entry_sha256": (
            receipt.get("registry_entry_sha256")
            == canonical_sha256(registry_entry)
        ),
        "artifact_path": bool(str(artifact.get("path", "")).strip()),
        "artifact_sha256": is_sha256(artifact.get("sha256")),
        "artifact_bytes": (
            isinstance(artifact.get("bytes"), int)
            and artifact["bytes"] >= 0
        ),
        "source_url_https_official_host": (
            parsed_source.scheme == "https"
            and bool(parsed_source.netloc)
            and parsed_source.netloc.lower() in official_hosts
        ),
        "license_name": (
            license_record.get("name")
            == registry_entry.get("license_spdx_or_name")
        ),
        "license_status": (
            license_record.get("status") == registry_entry.get("license_status")
        ),
        "license_snapshot_sha256": (
            is_sha256(license_record.get("snapshot_sha256"))
        ),
        "operator": bool(str(receipt.get("operator", "")).strip()),
        "conversion_not_started": receipt.get("conversion_status") == "not_started",
    }
    for name, passed in checks.items():
        if not passed:
            result["errors"].append(f"receipt field check failed: {name}")
    result["schema_and_registry_binding_valid"] = all(checks.values())

    artifact_path_value = str(artifact.get("path", "")).strip()
    if artifact_path_value:
        artifact_path = Path(artifact_path_value)
        if not artifact_path.is_absolute():
            artifact_path = repo_root / artifact_path
        result["artifact_resolved_path"] = str(artifact_path)
        if artifact_path.is_file():
            actual_bytes = artifact_path.stat().st_size
            actual_sha256 = sha256_file(artifact_path)
            result["artifact_actual_bytes"] = actual_bytes
            result["artifact_actual_sha256"] = actual_sha256
            result["artifact_checksum_reverified"] = (
                actual_bytes == artifact.get("bytes")
                and actual_sha256 == artifact.get("sha256")
            )
            if not result["artifact_checksum_reverified"]:
                result["errors"].append("raw artifact checksum or byte count mismatch")
        else:
            result["errors"].append("receipt raw artifact is missing locally")

    if result["license_snapshot_present"]:
        if license_snapshot_path.stat().st_size == 0:
            result["errors"].append("license snapshot is empty")
            return result
        actual = sha256_file(license_snapshot_path)
        result["license_snapshot_sha256"] = actual
        result["license_snapshot_hash_matches"] = (
            actual == license_record.get("snapshot_sha256")
        )
        if not result["license_snapshot_hash_matches"]:
            result["errors"].append("license snapshot checksum mismatch")
    else:
        result["errors"].append("license snapshot is missing")
    return result


def output_root_safe(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def build_report(
    *,
    repo_root: Path,
    registry_path: Path,
    dataset_id: str,
    input_root: Path,
    checksum_mode: str,
    receipt_path: Path | None = None,
    license_snapshot_path: Path | None = None,
) -> dict[str, Any]:
    if dataset_id not in DATASET_SPECS:
        raise ValueError(f"unsupported dataset_id: {dataset_id}")
    if checksum_mode not in {"metadata", "sha256"}:
        raise ValueError(f"unsupported checksum_mode: {checksum_mode}")

    entry, registry_sha256 = load_registry_entry(registry_path, dataset_id)
    spec = DATASET_SPECS[dataset_id]
    output_root = repo_root / entry["expected_local_root"]
    scan_root, candidates = discover_candidates(dataset_id, input_root, checksum_mode)
    source_ids = [candidate["planned_source_id"] for candidate in candidates]
    duplicates = sorted(
        item for item, count in Counter(source_ids).items() if count > 1
    )
    processor_path = repo_root / spec["processor"]
    config_path = (
        repo_root / spec["processor_config"] if spec["processor_config"] else None
    )
    receipt = validate_receipt(
        receipt_path,
        license_snapshot_path,
        repo_root,
        dataset_id,
        registry_sha256,
        entry,
    )
    conversion_command = [
        "python",
        processor_path.name,
        *spec["conversion_args"],
        "--input-dir",
        str(input_root),
        "--output-dir",
        str(output_root),
    ]
    gates = {
        "input_root_present": input_root.is_dir(),
        "expected_raw_layout_present": scan_root.is_dir(),
        "raw_candidates_present": bool(candidates),
        "planned_source_ids_unique": not duplicates,
        "processor_present": processor_path.is_file(),
        "processor_config_present_or_not_required": (
            config_path is None or config_path.is_file()
        ),
        "raw_artifact_receipt_schema_binding_valid": (
            receipt["schema_and_registry_binding_valid"]
        ),
        "license_snapshot_hash_matches_receipt": (
            receipt["license_snapshot_hash_matches"]
        ),
        "receipt_artifact_checksum_reverified": (
            receipt["artifact_checksum_reverified"]
        ),
        "raw_file_sha256_captured": (
            bool(candidates)
            and checksum_mode == "sha256"
            and all(candidate["sha256"] for candidate in candidates)
        ),
        "processed_output_root_safe": output_root_safe(output_root),
        "conversion_execution_count_is_zero": True,
        "model_training_evaluation_count_is_zero": True,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    script_path = Path(__file__).resolve()
    try:
        script_display_path = str(script_path.relative_to(repo_root))
    except ValueError:
        script_display_path = str(script_path)
    return {
        "schema_version": "0.1.0",
        "gate_id": f"{dataset_id}-raw-dry-run-preflight",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "git": git_context(repo_root),
        "preflight_script": {
            "path": script_display_path,
            "sha256": sha256_file(script_path),
        },
        "registry": {
            "path": str(registry_path),
            "sha256": registry_sha256,
            "expected_local_root": entry["expected_local_root"],
            "raw_format": entry["raw_format"],
            "metric_waypoint_spacing_m_registry_record": entry[
                "metric_waypoint_spacing_m"
            ],
        },
        "input": {
            "root": str(input_root),
            "scan_root": str(scan_root),
            "checksum_mode": checksum_mode,
            "candidate_count": len(candidates),
            "total_bytes": sum(candidate["bytes"] for candidate in candidates),
            "candidates": candidates,
            "duplicate_planned_source_ids": duplicates,
        },
        "receipt": receipt,
        "processor": {
            "path": spec["processor"],
            "sha256": sha256_file(processor_path) if processor_path.is_file() else None,
            "config_path": spec["processor_config"],
            "config_sha256": (
                sha256_file(config_path) if config_path and config_path.is_file() else None
            ),
            "working_directory": str(processor_path.parent),
            "planned_command": conversion_command,
            "executed": False,
        },
        "output": {
            "processed_root": str(output_root),
            "created_or_modified": False,
        },
        "execution_counts": {
            "downloads": 0,
            "conversions": 0,
            "model_forward": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "training": 0,
            "evaluation": 0,
            "simulation": 0,
        },
        "gates": gates,
        "blocking_gates": blockers,
        "passed": not blockers,
        "claim_boundary": (
            "Dry-run inventory and conversion-command rendering only. Raw candidate "
            "presence and hashes do not validate payload semantics, physical scale, "
            "timestamps, session grouping, conversion correctness, or model behavior."
        ),
    }


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        f"# {report['dataset_id']} raw DATA-PILOT dry-run",
        "",
        f"- Passed: {report['passed']}",
        f"- Candidates: {report['input']['candidate_count']}",
        f"- Candidate bytes: {report['input']['total_bytes']}",
        f"- Checksum mode: `{report['input']['checksum_mode']}`",
        "- Conversion executed: False",
        "- Model/training/evaluation/simulation executed: False",
        "",
        "## Planned conversion command (not executed)",
        "",
        f"Working directory: `{report['processor']['working_directory']}`",
        "",
        "```text",
        " ".join(report["processor"]["planned_command"]),
        "```",
        "",
        "## Blocking gates",
        "",
    ]
    lines.extend(f"- {item}" for item in report["blocking_gates"])
    lines.extend(["", "## Evidence boundary", "", report["claim_boundary"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--dataset-id", choices=sorted(DATASET_SPECS), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--checksum-mode", choices=("metadata", "sha256"), default="metadata"
    )
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--license-snapshot", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report(
            repo_root=args.repo_root.resolve(),
            registry_path=args.registry.resolve(),
            dataset_id=args.dataset_id,
            input_root=args.input_root.resolve(),
            checksum_mode=args.checksum_mode,
            receipt_path=args.receipt.resolve() if args.receipt else None,
            license_snapshot_path=(
                args.license_snapshot.resolve() if args.license_snapshot else None
            ),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.out_md:
        write_markdown(args.out_md, report)
    print(
        f"{args.dataset_id} dry-run complete; "
        f"candidates={report['input']['candidate_count']}, passed={report['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
