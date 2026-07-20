#!/usr/bin/env python3
"""Summarize RECON/HuRoN isolated conversion, manifest, and promotion gates."""

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


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_context(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
        "tracked_dirty": bool(run("status", "--porcelain=v1", "--untracked-files=no")),
    }


def candidate_peak_bound(inventory_path: Path, max_prefix_bytes: int) -> dict[str, Any]:
    sizes = []
    for line in inventory_path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].endswith(".hdf5"):
            sizes.append(int(fields[2]))
    prefix = [size for size in sizes if size <= max_prefix_bytes]
    exploratory = {}
    for threshold in (350_000, 700_000, 850_000, 950_000):
        selected = sorted(size for size in sizes if size >= threshold)[:100]
        exploratory[str(threshold)] = {
            "count": len(selected),
            "bytes": sum(selected),
        }
    return {
        "complete_prefix_count": len(prefix),
        "complete_prefix_bytes": sum(prefix),
        "exploratory_scans": exploratory,
        "temporary_candidate_bytes_upper_bound": (
            sum(prefix) + sum(item["bytes"] for item in exploratory.values())
        ),
    }


def build_report(repo_root: Path, evidence_root: Path, cap_bytes: int) -> dict[str, Any]:
    selection = load(evidence_root / "source_metadata/recon_candidate_selection.json")
    recon_conversion = load(evidence_root / "receipts/recon_conversion.json")
    huron_conversion = load(evidence_root / "receipts/huron_sacson_conversion.json")
    recon_audit = load(evidence_root / "processed_audit/recon.json")
    huron_audit = load(evidence_root / "processed_audit/huron_sacson.json")
    manifest_audit = load(evidence_root / "manifest/audit.json")
    recon_revalidation = load(evidence_root / "preflight/recon.json")
    huron_revalidation = load(evidence_root / "preflight/huron_sacson.json")
    registry_path = repo_root / "后续研究内容/data/dataset_registry.json"
    registry_sha = sha256_file(registry_path)
    current_receipts = []
    for relative in (
        "receipts/recon_recon_dataset_tar_gz_v3.json",
        "receipts/huron_sacson_Dec-09-2022-bww8_00000010_v3.json",
    ):
        path = evidence_root / relative
        payload = load(path)
        current_receipts.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "registry_sha256": payload["registry_sha256"],
                "artifact": payload["artifact"],
            }
        )
    inventory_path = (
        repo_root
        / "results/research/data_acquisition/recon_huron_pilot_20260720/source_metadata/recon_archive_members_20260720.txt"
    )
    peak = candidate_peak_bound(
        inventory_path, selection["inventory"]["max_member_bytes"]
    )
    permanent_data_bytes = (
        selection["selected"]["bytes"]
        + recon_conversion["output"]["bytes"]
        + huron_conversion["output"]["bytes"]
    )
    evidence_bytes = sum(
        path.stat().st_size
        for path in evidence_root.rglob("*")
        if path.is_file()
    )
    peak_stage_bytes_upper_bound = (
        peak["temporary_candidate_bytes_upper_bound"]
        + permanent_data_bytes
        + evidence_bytes
    )
    raw_revalidation_reports = (recon_revalidation, huron_revalidation)
    gates = {
        "recon_complete_prefix_selection_passed": selection["passed"],
        "recon_conversion_integrity_passed": recon_audit["passed"],
        "huron_conversion_integrity_passed": huron_audit["passed"],
        "group_safe_manifest_audit_passed": manifest_audit["passed"],
        "manifest_is_train_only_pilot": manifest_audit["split_counts"] == {"train": 2},
        "current_raw_receipts_bind_current_registry": all(
            item["registry_sha256"] == registry_sha for item in current_receipts
        ),
        "postconversion_raw_revalidation_has_only_expected_output_root_blocker": all(
            report["blocking_gates"] == ["processed_output_root_safe"]
            and report["receipt"]["artifact_checksum_reverified"]
            for report in raw_revalidation_reports
        ),
        "stage_disk_cap_respected": peak_stage_bytes_upper_bound <= cap_bytes,
        "no_model_training_evaluation_or_simulation": all(
            conversion["execution_counts"].get(name) == 0
            for conversion in (recon_conversion, huron_conversion)
            for name in (
                "model_forward", "backward", "optimizer_steps",
                "training", "evaluation", "simulation",
            )
        ),
    }
    promotion_blockers = sorted(
        set(recon_audit["promotion_blockers"] + huron_audit["promotion_blockers"])
    )
    return {
        "schema_version": "0.1.0",
        "audit_id": "recon-huron-conversion-pilot-20260720",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_context(repo_root),
        "registry": {"path": registry_path.as_posix(), "sha256": registry_sha},
        "disk_budget": {
            "authorized_bytes": cap_bytes,
            "permanent_data_bytes": permanent_data_bytes,
            "evidence_bytes_before_summary": evidence_bytes,
            **peak,
            "peak_stage_bytes_upper_bound": peak_stage_bytes_upper_bound,
            "fraction_used_upper_bound": peak_stage_bytes_upper_bound / cap_bytes,
        },
        "selection": {
            "report": (evidence_root / "source_metadata/recon_candidate_selection.json").as_posix(),
            "candidate_count": selection["candidate_count"],
            "eligible_count": selection["eligible_count"],
            "selected": selection["selected"],
        },
        "recon": {
            "conversion_receipt": (evidence_root / "receipts/recon_conversion.json").as_posix(),
            "processed_audit": (evidence_root / "processed_audit/recon.json").as_posix(),
            "result": recon_conversion["result"],
            "promotion_ready": recon_audit["promotion_ready"],
        },
        "huron_sacson": {
            "conversion_receipt": (evidence_root / "receipts/huron_sacson_conversion.json").as_posix(),
            "processed_audit": (evidence_root / "processed_audit/huron_sacson.json").as_posix(),
            "result": huron_conversion["result"],
            "promotion_ready": huron_audit["promotion_ready"],
        },
        "manifest": {
            "path": (evidence_root / "manifest/grouped_train_only.csv").as_posix(),
            "audit": (evidence_root / "manifest/audit.json").as_posix(),
            "row_count": manifest_audit["row_count"],
            "split_counts": manifest_audit["split_counts"],
            "note": "Pilot-only train assignment; no evaluation holdout is claimed or used.",
        },
        "current_raw_receipts": current_receipts,
        "execution_counts": {
            "semantic_conversions": 2,
            "model_forward": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "training": 0,
            "evaluation": 0,
            "simulation": 0,
        },
        "gates": gates,
        "blocking_gates": [name for name, passed in gates.items() if not passed],
        "conversion_pilot_passed": all(gates.values()),
        "promotion_blockers": promotion_blockers,
        "promotion_ready": all(gates.values()) and not promotion_blockers,
        "claim_boundary": (
            "Two isolated processed pilots, integrity checks, and a train-only leakage-safe "
            "manifest. RECON dt remains inferred because per-frame timestamps are absent; "
            "collection policy/version metadata is absent. No model execution, training, "
            "evaluation, or simulation was performed."
        ),
    }


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    disk = report["disk_budget"]
    lines = [
        "# RECON / HuRoN conversion pilot audit",
        "",
        f"- Conversion pilot passed: {report['conversion_pilot_passed']}",
        f"- Promotion ready: {report['promotion_ready']}",
        f"- Peak stage storage upper bound: {disk['peak_stage_bytes_upper_bound'] / 1e9:.6f} GB / {disk['authorized_bytes'] / 1e9:.0f} GB",
        "- Model/backward/optimizer/training/evaluation/simulation: 0",
        "",
        "## RECON",
        "",
        f"- Candidate prefix audited: {report['selection']['candidate_count']}",
        f"- Selected: `{report['selection']['selected']['relative_path']}`",
        f"- Frames/windows: {report['recon']['result']['frame_count']} / {report['recon']['result']['canonical_window_count']}",
        f"- dt: {report['recon']['result']['nominal_dt_s']} s ({report['recon']['result']['nominal_dt_status']})",
        f"- Observed median step: {report['recon']['result']['metric_waypoint_spacing_m_observed']} m",
        "",
        "## HuRoN / SACSoN",
        "",
        f"- Synchronized/retained frames: {report['huron_sacson']['result']['synchronized_frame_count_before_filter']} / {report['huron_sacson']['result']['frame_count']}",
        f"- Canonical windows: {report['huron_sacson']['result']['canonical_window_count']}",
        f"- dt: {report['huron_sacson']['result']['nominal_dt_s']} s ({report['huron_sacson']['result']['nominal_dt_status']})",
        f"- Observed median step: {report['huron_sacson']['result']['metric_waypoint_spacing_m_observed']} m",
        "",
        "## Promotion blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in report["promotion_blockers"])
    lines.extend(["", "## Evidence boundary", "", report["claim_boundary"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--cap-bytes", type=int, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.repo_root.resolve(), args.evidence_root, args.cap_bytes)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.out_md, report)
    print(
        f"conversion pilot passed={report['conversion_pilot_passed']}, "
        f"promotion_ready={report['promotion_ready']}"
    )
    return 0 if report["conversion_pilot_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
