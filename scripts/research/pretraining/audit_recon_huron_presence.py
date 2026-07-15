#!/usr/bin/env python3
"""Read-only local presence audit for RECON and HuRoN/SACSoN pilots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


RAW_SUFFIXES = {".h5", ".hdf5", ".bag", ".db3", ".mcap"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_context(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_worktree_dirty": bool(
            run("status", "--porcelain=v1", "--untracked-files=no")
        ),
        "worktree_dirty_including_preserved_untracked": bool(
            run("status", "--porcelain=v1")
        ),
    }


def path_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    file_count = 0
    directory_count = 0
    total_bytes = 0
    suffix_counts: dict[str, int] = {}
    newest_mtime_ns = stat.st_mtime_ns
    for root, directories, files in os.walk(path):
        directory_count += len(directories)
        for name in files:
            candidate = Path(root) / name
            file_stat = candidate.stat()
            file_count += 1
            total_bytes += file_stat.st_size
            newest_mtime_ns = max(newest_mtime_ns, file_stat.st_mtime_ns)
            suffix = candidate.suffix.lower() or "<none>"
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    return {
        "path": str(path),
        "exists": True,
        "is_directory": path.is_dir(),
        "file_count": file_count,
        "directory_count_below_root": directory_count,
        "total_file_bytes": total_bytes,
        "root_mtime_ns": stat.st_mtime_ns,
        "newest_mtime_ns": newest_mtime_ns,
        "suffix_counts": dict(sorted(suffix_counts.items())),
    }


def processed_trajectory_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(
        1
        for child in path.iterdir()
        if child.is_dir()
        and (child / "traj_data.pkl").is_file()
        and any(child.glob("*.jpg"))
    )


def manifest_dataset_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return sorted({row.get("dataset_id", "") for row in csv.DictReader(handle)})


def processor_help(command: list[str], cwd: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {
        "command": command,
        "cwd": str(cwd),
        "returncode": result.returncode,
        "usage_line": (result.stdout or result.stderr).splitlines()[0]
        if (result.stdout or result.stderr).splitlines() else "",
    }


def audit(repo_root: Path, diffusion_root: Path) -> dict[str, Any]:
    audit_script_path = Path(__file__).resolve()
    registry_path = repo_root / "后续研究内容/data/dataset_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    selected = {
        item["dataset_id"]: item
        for item in registry["datasets"]
        if item["dataset_id"] in {"recon", "huron_sacson"}
    }
    if set(selected) != {"recon", "huron_sacson"}:
        raise ValueError("registry does not contain both RECON and HuRoN/SACSoN")

    expected_roots = {
        name: repo_root / item["expected_local_root"] for name, item in selected.items()
    }
    expected_snapshots = {
        name: path_snapshot(path) for name, path in expected_roots.items()
    }
    recon_datavis = repo_root / "nomad_dataset/recon_datavis"
    datavis_snapshot = path_snapshot(recon_datavis)

    manifest_paths = sorted((repo_root / "results/research/data_audit").glob("*.csv"))
    manifests = [
        {
            "path": str(path.relative_to(repo_root)),
            "sha256": file_sha256(path),
            "dataset_ids": manifest_dataset_ids(path),
        }
        for path in manifest_paths
    ]
    governance_files = sorted((repo_root / "后续研究内容/data").rglob("*"))
    receipt_candidates = [
        str(path.relative_to(repo_root))
        for path in governance_files
        if path.is_file()
        and any(token in path.name.lower() for token in ("receipt", "license", "snapshot"))
        and any(token in path.name.lower() for token in ("recon", "huron", "sacson"))
    ]

    raw_candidates: list[dict[str, Any]] = []
    excluded = {
        (repo_root / "nomad_dataset/go_stanford").resolve(),
        (repo_root / "nomad_dataset/go_stanford_dataset").resolve(),
    }
    for workspace in (repo_root, diffusion_root):
        for root, directories, files in os.walk(workspace):
            root_path = Path(root)
            directories[:] = [
                name for name in directories
                if (root_path / name).resolve() not in excluded
            ]
            for name in files:
                path = root_path / name
                if path.suffix.lower() in RAW_SUFFIXES:
                    stat = path.stat()
                    raw_candidates.append({
                        "path": str(path),
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    })

    processors = {
        "recon": repo_root / "train/process_recon.py",
        "huron_sacson": repo_root / "train/process_bags.py",
        "huron_config": repo_root / "train/vint_train/process_data/process_bags_config.yaml",
        "bag_utils": repo_root / "train/vint_train/process_data/process_data_utils.py",
    }
    processor_receipts = {
        name: {
            "path": str(path.relative_to(repo_root)),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
            "sha256": file_sha256(path) if path.is_file() else None,
        }
        for name, path in processors.items()
    }
    dependency_specs = {
        name: importlib.util.find_spec(name) is not None
        for name in ("h5py", "PIL", "cv2", "numpy", "yaml", "rosbag", "rosbags")
    }
    helps = {
        "recon": processor_help(
            ["python", "process_recon.py", "--help"], repo_root / "train"
        ),
        "huron_sacson": processor_help(
            ["python", "process_bags.py", "--help"], repo_root / "train"
        ),
    }

    datasets: dict[str, Any] = {}
    for name, item in selected.items():
        snapshot = expected_snapshots[name]
        dataset_tokens = ("recon",) if name == "recon" else ("huron", "sacson")
        dataset_receipt_candidates = [
            candidate
            for candidate in receipt_candidates
            if any(token in Path(candidate).name.lower() for token in dataset_tokens)
        ]
        manifest_rows_present = any(name in entry["dataset_ids"] for entry in manifests)
        if name == "huron_sacson":
            manifest_rows_present = any(
                bool({"huron_sacson", "sacson"} & set(entry["dataset_ids"]))
                for entry in manifests
            )
        datasets[name] = {
            "registry_contract": {
                "display_name": item["display_name"],
                "raw_format": item["raw_format"],
                "expected_local_root": item["expected_local_root"],
                "metric_waypoint_spacing_m": item["metric_waypoint_spacing_m"],
                "nominal_dt_s": item["nominal_dt_s"],
                "registry_license_status": item["license_status"],
                "registry_license_name": item["license_spdx_or_name"],
                "registry_local_status": item["local_status"],
            },
            "local_root": snapshot,
            "processed_trajectory_count": processed_trajectory_count(expected_roots[name]),
            "manifest_rows_present": manifest_rows_present,
            "raw_artifact_receipt_present": any(
                "receipt" in Path(candidate).name.lower()
                for candidate in dataset_receipt_candidates
            ),
            "live_official_license_snapshot_present": any(
                any(
                    token in Path(candidate).name.lower()
                    for token in ("license", "snapshot")
                )
                for candidate in dataset_receipt_candidates
            ),
            "materialized": snapshot["exists"] and processed_trajectory_count(expected_roots[name]) > 0,
        }

    gates = {
        "recon_expected_root_present": datasets["recon"]["local_root"]["exists"],
        "recon_processed_trajectories_present": datasets["recon"]["processed_trajectory_count"] > 0,
        "recon_manifest_rows_present": datasets["recon"]["manifest_rows_present"],
        "recon_raw_receipt_present": datasets["recon"]["raw_artifact_receipt_present"],
        "huron_expected_root_present": datasets["huron_sacson"]["local_root"]["exists"],
        "huron_processed_trajectories_present": datasets["huron_sacson"]["processed_trajectory_count"] > 0,
        "huron_manifest_rows_present": datasets["huron_sacson"]["manifest_rows_present"],
        "huron_raw_receipt_present": datasets["huron_sacson"]["raw_artifact_receipt_present"],
        "processor_entrypoints_present": all(
            receipt["exists"] for receipt in processor_receipts.values()
        ),
        "processor_help_read_only_passed": all(
            result["returncode"] == 0 for result in helps.values()
        ),
        "no_external_download_performed": True,
        "no_data_conversion_performed": True,
        "no_model_or_training_performed": True,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": "0.1.0",
        "audit_id": "recon-huron-local-presence-20260716",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "workspace_roots": [str(repo_root), str(diffusion_root)],
            "read_only_data_scan": True,
            "external_download_performed": False,
            "data_conversion_performed": False,
            "model_execution_performed": False,
            "training_or_evaluation_performed": False,
        },
        "git": git_context(repo_root),
        "audit_script": {
            "path": str(audit_script_path.relative_to(repo_root)),
            "sha256": file_sha256(audit_script_path),
        },
        "registry": {
            "path": str(registry_path.relative_to(repo_root)),
            "sha256": file_sha256(registry_path),
            "license_evidence_boundary": (
                "Registry statements are repository records, not a live official-page snapshot or payload receipt."
            ),
        },
        "datasets": datasets,
        "recon_datavis_candidate": {
            **datavis_snapshot,
            "classification": "visualization_source_code_not_dataset_payload",
        },
        "raw_payload_extension_candidates": raw_candidates,
        "manifests": manifests,
        "receipt_or_license_snapshot_candidates": receipt_candidates,
        "processors": processor_receipts,
        "processor_help": helps,
        "dependency_specs": dependency_specs,
        "static_processor_risks": [
            "processors create output directly and do not emit raw artifact receipts or checksums",
            "processors do not pin their own code/config version in each processed trajectory",
            "RECON session identity depends only on HDF5 filename and is not independently registered",
            "HuRoN output name preserves one parent directory plus bag filename but not collection policy/version metadata",
            "HuRoN human-containing imagery still requires an official terms/privacy snapshot attached to any manifest",
        ],
        "gates": gates,
        "blocking_gates": blockers,
        "materialized_dataset_count": sum(
            1 for dataset in datasets.values() if dataset["materialized"]
        ),
        "passed": not blockers,
        "claim_boundary": (
            "This is a local presence audit only. Processor code and registry records do not count as data. "
            "No RECON/HuRoN content, license payload, physical scale, timestamp, or session split was validated."
        ),
    }


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# RECON / HuRoN Local Presence Audit",
        "",
        f"- Passed: {report['passed']}",
        f"- Materialized datasets: {report['materialized_dataset_count']} / 2",
        "- External download performed: False",
        "- Data conversion performed: False",
        "- Model/training/evaluation performed: False",
        "",
        "## Verified local facts",
        "",
    ]
    for name, dataset in report["datasets"].items():
        lines.extend([
            f"### {name}",
            "",
            f"- Expected root: `{dataset['registry_contract']['expected_local_root']}`",
            f"- Root exists: {dataset['local_root']['exists']}",
            f"- Processed trajectories: {dataset['processed_trajectory_count']}",
            f"- Manifest rows: {dataset['manifest_rows_present']}",
            f"- Raw receipt / live license snapshot: {dataset['raw_artifact_receipt_present']} / {dataset['live_official_license_snapshot_present']}",
            "",
        ])
    candidate = report["recon_datavis_candidate"]
    lines.extend([
        "### recon_datavis",
        "",
        f"- Size/files: {candidate.get('total_file_bytes', 0)} bytes / {candidate.get('file_count', 0)} files",
        f"- Classification: {candidate['classification']}",
        "",
        "## Processor readiness (not data readiness)",
        "",
        f"- Entrypoints present: {report['gates']['processor_entrypoints_present']}",
        f"- Read-only `--help` passed: {report['gates']['processor_help_read_only_passed']}",
        f"- Dependency specs: `{json.dumps(report['dependency_specs'], sort_keys=True)}`",
        "",
        "## Blocking gates",
        "",
    ])
    lines.extend(f"- {item}" for item in report["blocking_gates"])
    lines.extend(["", "## Static processor risks", ""])
    lines.extend(f"- {item}" for item in report["static_processor_risks"])
    lines.extend(["", "## Evidence boundary", "", report["claim_boundary"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--diffusion-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit(args.repo_root.resolve(), args.diffusion_root.resolve())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, report)
    print(
        "RECON/HuRoN presence audit complete; "
        f"materialized={report['materialized_dataset_count']}/2, passed={report['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
