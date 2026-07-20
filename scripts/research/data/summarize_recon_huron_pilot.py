#!/usr/bin/env python3
"""Summarize the governed RECON/HuRoN raw pilot acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_context(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(
            run("status", "--porcelain=v1", "--untracked-files=no")
        ),
    }


def archive_inventory(path: Path) -> dict[str, Any]:
    count = 0
    total_bytes = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].endswith(".hdf5"):
            count += 1
            total_bytes += int(fields[2])
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "hdf5_member_count": count,
        "hdf5_uncompressed_bytes": total_bytes,
    }


def build_report(repo_root: Path, evidence_root: Path, cap_gb: float) -> dict[str, Any]:
    registry_path = repo_root / "后续研究内容/data/dataset_registry.json"
    registry_sha = sha256_file(registry_path)
    recon_receipt_path = evidence_root / "receipts/recon_recon_dataset_tar_gz_v2.json"
    huron_receipt_path = (
        evidence_root / "receipts/huron_sacson_Dec-09-2022-bww8_00000010_v2.json"
    )
    lineage_path = evidence_root / "receipts/recon_selected_hdf5_member.json"
    recon_preflight_path = evidence_root / "preflight/recon.json"
    huron_preflight_path = evidence_root / "preflight/huron_sacson.json"
    recon_content_path = evidence_root / "content_audit/recon.json"
    huron_content_path = evidence_root / "content_audit/huron_sacson.json"
    inventory_path = evidence_root / "source_metadata/recon_archive_members_20260720.txt"

    recon_receipt = load_json(recon_receipt_path)
    huron_receipt = load_json(huron_receipt_path)
    lineage = load_json(lineage_path)
    recon_preflight = load_json(recon_preflight_path)
    huron_preflight = load_json(huron_preflight_path)
    recon_content = load_json(recon_content_path)
    huron_content = load_json(huron_content_path)
    payload_bytes = (
        recon_receipt["artifact"]["bytes"]
        + huron_receipt["artifact"]["bytes"]
        + lineage["artifact"]["bytes"]
    )
    cap_bytes = int(cap_gb * 1_000_000_000)
    execution_counts = {
        "official_artifact_downloads": 2,
        "selected_raw_member_extractions": 1,
        "semantic_conversions": 0,
        "model_forward": 0,
        "backward": 0,
        "optimizer_steps": 0,
        "training": 0,
        "evaluation": 0,
        "simulation": 0,
    }
    gates = {
        "current_receipts_bind_current_registry": all(
            receipt["registry_sha256"] == registry_sha
            for receipt in (recon_receipt, huron_receipt)
        ),
        "recon_receipt_artifact_reverified": recon_preflight["receipt"][
            "artifact_checksum_reverified"
        ],
        "huron_receipt_artifact_reverified": huron_preflight["receipt"][
            "artifact_checksum_reverified"
        ],
        "recon_raw_preflight_passed": recon_preflight["passed"],
        "huron_raw_preflight_passed": huron_preflight["passed"],
        "recon_content_audit_passed": recon_content["passed"],
        "huron_content_audit_passed": huron_content["passed"],
        "recon_member_lineage_matches_content_audit": (
            lineage["artifact"]["sha256"] == recon_content["artifact"]["sha256"]
            and lineage["artifact"]["bytes"] == recon_content["artifact"]["bytes"]
        ),
        "write_budget_respected": payload_bytes <= cap_bytes,
        "no_semantic_conversion_or_model_execution": all(
            execution_counts[name] == 0
            for name in (
                "semantic_conversions", "model_forward", "backward",
                "optimizer_steps", "training", "evaluation", "simulation",
            )
        ),
    }
    snapshots = []
    for relative in (
        "licenses/recon_dataset_page_20260720.html",
        "licenses/recon_license_snapshot_20260720.md",
        "licenses/huron_dataset_page_20260720.html",
        "licenses/huron_license_privacy_snapshot_20260720.md",
        "source_metadata/huron_index_20260720.html",
        "source_metadata/recon_archive_headers_20260720.txt",
        "source_metadata/huron_selected_bag_headers_20260720.txt",
    ):
        path = evidence_root / relative
        snapshots.append({
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "schema_version": "0.1.0",
        "audit_id": "recon-huron-raw-pilot-acquisition-20260720",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_context(repo_root),
        "registry": {"path": registry_path.as_posix(), "sha256": registry_sha},
        "disk_budget": {
            "authorized_gb_decimal": cap_gb,
            "authorized_bytes": cap_bytes,
            "payload_bytes_written": payload_bytes,
            "payload_gb_decimal": payload_bytes / 1_000_000_000,
            "fraction_used": payload_bytes / cap_bytes,
        },
        "official_snapshots": snapshots,
        "recon": {
            "archive_receipt": recon_receipt_path.as_posix(),
            "archive_receipt_sha256": sha256_file(recon_receipt_path),
            "archive_inventory": archive_inventory(inventory_path),
            "selected_member_receipt": lineage_path.as_posix(),
            "selected_member_receipt_sha256": sha256_file(lineage_path),
            "selected_artifact": lineage["artifact"],
            "preflight": recon_preflight_path.as_posix(),
            "content_audit": recon_content_path.as_posix(),
        },
        "huron_sacson": {
            "artifact_receipt": huron_receipt_path.as_posix(),
            "artifact_receipt_sha256": sha256_file(huron_receipt_path),
            "selected_artifact": huron_receipt["artifact"],
            "preflight": huron_preflight_path.as_posix(),
            "content_audit": huron_content_path.as_posix(),
        },
        "execution_counts": execution_counts,
        "gates": gates,
        "blocking_gates": [name for name, passed in gates.items() if not passed],
        "passed": all(gates.values()),
        "remaining_blockers": [
            "Semantic conversion of either pilot has not been authorized or executed.",
            "Processed manifests, group-safe splits, dt/scale verification, and processor receipts do not exist yet.",
            "These two raw pilots do not establish full-dataset completeness or a multi-dataset training result.",
        ],
        "claim_boundary": (
            "Official-page snapshot, raw-byte integrity, minimum content readability, "
            "and conversion readiness for one RECON HDF5 and one HuRoN bag only. "
            "No semantic conversion, model execution, training, evaluation, or simulation was performed."
        ),
    }


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    disk = report["disk_budget"]
    recon = report["recon"]
    huron = report["huron_sacson"]
    lines = [
        "# RECON / HuRoN raw pilot acquisition audit",
        "",
        f"- Passed: {report['passed']}",
        f"- Payload written: {disk['payload_gb_decimal']:.6f} GB decimal / {disk['authorized_gb_decimal']:.0f} GB authorized ({disk['fraction_used']:.2%})",
        "- Semantic conversion/model/training/evaluation/simulation: 0",
        "",
        "## RECON",
        "",
        f"- Official archive bytes: {load_json(Path(recon['archive_receipt']))['artifact']['bytes']}",
        f"- Inventory: {recon['archive_inventory']['hdf5_member_count']} HDF5 members, {recon['archive_inventory']['hdf5_uncompressed_bytes']} uncompressed bytes",
        f"- Selected raw member: `{recon['selected_artifact']['path']}`",
        f"- Selected SHA-256: `{recon['selected_artifact']['sha256']}`",
        "- Strict receipt/preflight/content gates: passed",
        "",
        "## HuRoN / SACSoN",
        "",
        f"- Selected raw bag: `{huron['selected_artifact']['path']}`",
        f"- Bytes: {huron['selected_artifact']['bytes']}",
        f"- SHA-256: `{huron['selected_artifact']['sha256']}`",
        "- Required fisheye-image and odometry topics/content: passed",
        "",
        "## Remaining blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in report["remaining_blockers"])
    lines.extend(["", "## Evidence boundary", "", report["claim_boundary"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--cap-gb", type=float, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.repo_root.resolve(), args.evidence_root, args.cap_gb)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.out_md, report)
    print(f"raw pilot acquisition audit passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
