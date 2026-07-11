#!/usr/bin/env python3
"""Build a stable, trajectory-level navigation dataset manifest.

This tool reads already processed ViNT-style trajectory folders.  It does not
unpickle trajectory data and therefore does not execute untrusted pickle code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


MANIFEST_VERSION = "0.1.0"
FIELDS = (
    "manifest_version",
    "dataset_id",
    "trajectory_id",
    "source_session",
    "source_session_method",
    "source_path",
    "split",
    "leakage_group",
    "robot_id",
    "camera_id",
    "environment",
    "num_frames",
    "has_traj_data",
    "nominal_dt_s",
    "metric_waypoint_spacing_m",
    "position_unit",
    "yaw_unit",
    "license_id",
    "license_status",
    "processor_version",
    "trajectory_meta_sha256",
)


class ManifestError(ValueError):
    pass


def load_registry(path: Path, dataset_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read registry {path}: {exc}") from exc
    matches = [item for item in payload.get("datasets", []) if item.get("dataset_id") == dataset_id]
    if len(matches) != 1:
        raise ManifestError(f"Expected one registry entry for {dataset_id!r}, found {len(matches)}")
    return matches[0]


def load_splits(split_root: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    conflicts: list[str] = []
    if not split_root.is_dir():
        raise ManifestError(f"Split root does not exist: {split_root}")
    for split_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        names_path = split_dir / "traj_names.txt"
        if not names_path.is_file():
            continue
        split = split_dir.name.lower()
        for name in names_path.read_text(encoding="utf-8").splitlines():
            name = name.strip()
            if not name:
                continue
            previous = assignments.get(name)
            if previous is not None and previous != split:
                conflicts.append(f"{name}: {previous} vs {split}")
            assignments[name] = split
    if conflicts:
        raise ManifestError("Trajectories assigned to multiple splits: " + "; ".join(conflicts[:10]))
    if not assignments:
        raise ManifestError(f"No traj_names.txt assignments found under {split_root}")
    return assignments


def infer_session(dataset_id: str, trajectory_id: str) -> tuple[str, str]:
    if dataset_id == "go_stanford":
        prefix, separator, suffix = trajectory_id.rpartition("_")
        if separator and suffix.isdigit() and prefix:
            return prefix, "heuristic_remove_trailing_segment"
    return trajectory_id, "trajectory_id_fallback"


def meta_sha256(path: Path) -> str:
    if not path.is_file():
        return "NA"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_frames(trajectory_dir: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png"}
    return sum(
        1
        for child in trajectory_dir.iterdir()
        if child.is_file() and child.suffix.lower() in extensions
    )


def build_rows(
    dataset_id: str,
    dataset_root: Path,
    split_root: Path,
    registry_entry: dict[str, Any],
    processor_version: str,
) -> list[dict[str, Any]]:
    if not dataset_root.is_dir():
        raise ManifestError(f"Dataset root does not exist: {dataset_root}")
    splits = load_splits(split_root)
    rows: list[dict[str, Any]] = []
    for trajectory_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        trajectory_id = trajectory_dir.name
        source_session, method = infer_session(dataset_id, trajectory_id)
        meta_path = trajectory_dir / "traj_data.pkl"
        rows.append(
            {
                "manifest_version": MANIFEST_VERSION,
                "dataset_id": dataset_id,
                "trajectory_id": trajectory_id,
                "source_session": source_session,
                "source_session_method": method,
                "source_path": trajectory_dir.as_posix(),
                "split": splits.get(trajectory_id, "unassigned"),
                "leakage_group": f"{dataset_id}:{source_session}",
                "robot_id": registry_entry.get("platform", "unknown"),
                "camera_id": "default",
                "environment": ";".join(registry_entry.get("environment", [])),
                "num_frames": count_frames(trajectory_dir),
                "has_traj_data": meta_path.is_file(),
                "nominal_dt_s": registry_entry.get("nominal_dt_s") or "NA",
                "metric_waypoint_spacing_m": registry_entry.get("metric_waypoint_spacing_m", "NA"),
                "position_unit": "meter",
                "yaw_unit": "radian",
                "license_id": registry_entry.get("license_spdx_or_name", "UNKNOWN"),
                "license_status": registry_entry.get("license_status", "unknown"),
                "processor_version": processor_version,
                "trajectory_meta_sha256": meta_sha256(meta_path),
            }
        )
    if not rows:
        raise ManifestError(f"No trajectory directories found under {dataset_root}")
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]], merge_existing: bool) -> None:
    combined = rows
    if merge_existing and path.is_file():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
        dataset_ids = {row["dataset_id"] for row in rows}
        combined = [row for row in existing if row.get("dataset_id") not in dataset_ids] + rows
    combined.sort(key=lambda row: (row["dataset_id"], row["trajectory_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(combined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a ViNT trajectory manifest without training.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--processor-version", default="preexisting_unknown")
    parser.add_argument("--merge-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        entry = load_registry(args.registry, args.dataset_id)
        rows = build_rows(
            args.dataset_id,
            args.dataset_root,
            args.split_root,
            entry,
            args.processor_version,
        )
        write_manifest(args.out, rows, args.merge_existing)
    except ManifestError as exc:
        print(f"ERROR: {exc}")
        return 2
    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    print(f"Wrote {len(rows)} {args.dataset_id} trajectories to {args.out}")
    print("Splits: " + ", ".join(f"{key}={value}" for key, value in sorted(split_counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
