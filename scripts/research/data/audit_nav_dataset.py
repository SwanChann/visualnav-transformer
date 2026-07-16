#!/usr/bin/env python3
"""Audit navigation manifests for schema, provenance, and split leakage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_nav_manifest import FIELDS, MANIFEST_VERSION, ManifestError


VALID_SPLITS = {"train", "val", "test", "heldout"}
TRUSTED_SESSION_METHODS = {
    "recon": {"recon_hdf5_stem_identity", "explicit_raw_session_map"},
    "huron_sacson": {
        "sacson_legacy_bag_remove_segment_index",
        "explicit_raw_session_map",
    },
}


def canonical_split(value: str) -> str:
    normalized = value.strip().lower()
    return "val" if normalized == "validation" else normalized


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["dataset_id"]: item for item in payload.get("datasets", [])}


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"Manifest missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ManifestError("Manifest has no rows")
    return rows


def _positive_number(value: str) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _consolidate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for item in items:
        grouped[(item["code"], item["message"])].extend(item["rows"])
    consolidated = []
    for (code, message), row_numbers in sorted(grouped.items()):
        unique_rows = sorted(set(row_numbers))
        consolidated.append(
            {
                "code": code,
                "message": message,
                "occurrence_count": len(unique_rows),
                "rows": unique_rows[:50],
            }
        )
    return consolidated


def audit_rows(
    rows: list[dict[str, str]],
    registry: dict[str, dict[str, Any]],
    check_paths: bool,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def add(target: list[dict[str, Any]], code: str, message: str, rows_: list[int]) -> None:
        target.append({"code": code, "message": message, "rows": rows_[:50]})

    identity_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    leakage_splits: dict[str, set[str]] = defaultdict(set)
    leakage_rows: dict[str, list[int]] = defaultdict(list)
    session_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    session_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    split_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, start=2):
        dataset_id = row["dataset_id"].strip()
        trajectory_id = row["trajectory_id"].strip()
        split = canonical_split(row["split"])
        source_session = row["source_session"].strip()
        source_session_method = row["source_session_method"].strip()
        leakage_group = row["leakage_group"].strip()
        identity_rows[(dataset_id, trajectory_id)].append(index)
        leakage_splits[leakage_group].add(split)
        leakage_rows[leakage_group].append(index)
        session_key = (dataset_id, source_session)
        session_splits[session_key].add(split)
        session_rows[session_key].append(index)
        split_counts[split] += 1
        dataset_counts[dataset_id] += 1

        if row["manifest_version"] != MANIFEST_VERSION:
            add(errors, "manifest_version", f"Unsupported version {row['manifest_version']!r}", [index])
        if not dataset_id or dataset_id not in registry:
            add(errors, "unknown_dataset", f"Unknown dataset_id {dataset_id!r}", [index])
            continue
        if not trajectory_id:
            add(errors, "empty_trajectory_id", "trajectory_id is empty", [index])
        if not source_session:
            add(errors, "empty_source_session", "source_session is empty", [index])
        if not source_session_method:
            add(
                errors,
                "empty_source_session_method",
                "source_session_method is empty",
                [index],
            )
        if (
            dataset_id in TRUSTED_SESSION_METHODS
            and source_session_method not in TRUSTED_SESSION_METHODS[dataset_id]
        ):
            add(
                errors,
                "unsafe_session_method",
                f"Dataset {dataset_id} requires an auditable raw HDF5/bag session method",
                [index],
            )
        if not leakage_group:
            add(errors, "empty_leakage_group", "leakage_group is empty", [index])
        if split not in VALID_SPLITS:
            add(errors, "invalid_or_unassigned_split", f"Invalid split {split!r}", [index])
        if not _positive_number(row["num_frames"]):
            add(errors, "no_frames", "num_frames must be positive", [index])
        if str(row["has_traj_data"]).lower() != "true":
            add(errors, "missing_traj_data", "traj_data.pkl is missing", [index])
        if not _positive_number(row["nominal_dt_s"]):
            add(errors, "missing_dt", "nominal_dt_s must be known and positive", [index])
        if not _positive_number(row["metric_waypoint_spacing_m"]):
            add(errors, "invalid_metric_scale", "metric_waypoint_spacing_m must be positive", [index])

        registry_entry = registry[dataset_id]
        expected_license = registry_entry.get("license_spdx_or_name", "UNKNOWN")
        if row["license_id"] != expected_license:
            add(errors, "license_mismatch", "Manifest license differs from registry", [index])
        if "UNKNOWN" in expected_license or "NEEDS_CONFIRMATION" in expected_license:
            add(errors, "license_blocked", f"Dataset {dataset_id} has unresolved license", [index])
        if "NC" in expected_license:
            add(warnings, "noncommercial_license", f"Dataset {dataset_id} is non-commercial", [index])
        if row["processor_version"] in {"", "unknown", "preexisting_unknown"}:
            add(warnings, "unknown_processor_version", "Processor version is not pinned", [index])
        if check_paths and not Path(row["source_path"]).is_dir():
            add(errors, "missing_source_path", f"Missing path {row['source_path']}", [index])

    for identity, indices in identity_rows.items():
        if len(indices) > 1:
            add(errors, "duplicate_trajectory", f"Duplicate identity {identity}", indices)
    for group, splits in leakage_splits.items():
        valid = splits & VALID_SPLITS
        if len(valid) > 1:
            add(errors, "leakage_group_cross_split", f"{group} spans {sorted(valid)}", leakage_rows[group])
    for session, splits in session_splits.items():
        valid = splits & VALID_SPLITS
        if len(valid) > 1:
            add(errors, "source_session_cross_split", f"{session} spans {sorted(valid)}", session_rows[session])

    errors = _consolidate(errors)
    warnings = _consolidate(warnings)
    return {
        "manifest_version": MANIFEST_VERSION,
        "row_count": len(rows),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "error_occurrence_count": sum(item["occurrence_count"] for item in errors),
        "warning_occurrence_count": sum(item["occurrence_count"] for item in warnings),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Navigation Dataset Manifest Audit",
        "",
        f"- Rows: {report['row_count']}",
        f"- Datasets: {report['dataset_counts']}",
        f"- Splits: {report['split_counts']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        f"- Passed: {report['passed']}",
        "",
        "## Errors",
        "",
    ]
    if report["errors"]:
        for item in report["errors"]:
            lines.append(
                f"- `{item['code']}`: {item['message']} "
                f"({item['occurrence_count']} occurrence(s); sample rows {item['rows']})"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        for item in report["warnings"]:
            lines.append(
                f"- `{item['code']}`: {item['message']} "
                f"({item['occurrence_count']} occurrence(s); sample rows {item['rows']})"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "A passing manifest establishes data-contract consistency only; it does not establish "
            "model quality, benchmark validity, or permission beyond the recorded license.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a navigation trajectory manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--check-paths", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = load_manifest(args.manifest)
        report = audit_rows(rows, load_registry(args.registry), args.check_paths)
    except (ManifestError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.out_md, report)
    print(
        f"Audited {report['row_count']} rows: errors={report['error_count']}, "
        f"warnings={report['warning_count']}, passed={report['passed']}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
