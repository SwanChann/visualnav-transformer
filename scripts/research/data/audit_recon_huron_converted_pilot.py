#!/usr/bin/env python3
"""Independently audit isolated RECON/HuRoN processed pilot outputs."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Mapping

from convert_recon_huron_pilot import sha256_file, tree_inventory


def load_registry_entry(path: Path, dataset_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        item for item in payload.get("datasets", [])
        if item.get("dataset_id") == dataset_id
    ]
    if len(matches) != 1:
        raise ValueError(f"dataset_id must match one registry entry: {dataset_id}")
    return matches[0]


def audit_trajectory(path: Path, canonical_window_frames: int) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    metadata_path = path / "source_metadata.json"
    trajectory_path = path / "traj_data.pkl"
    if not metadata_path.is_file() or not trajectory_path.is_file():
        raise ValueError(f"trajectory metadata missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with trajectory_path.open("rb") as handle:
        trajectory = pickle.load(handle)
    positions = np.asarray(trajectory.get("position"), dtype=np.float64)
    yaws = np.asarray(trajectory.get("yaw"), dtype=np.float64)
    images = sorted(
        path.glob("*.jpg"), key=lambda item: int(item.stem)
    )
    expected_indices = list(range(len(images)))
    actual_indices = [int(item.stem) for item in images]
    errors = []
    if actual_indices != expected_indices:
        errors.append("image indices are not contiguous from zero")
    if positions.ndim != 2 or positions.shape[1] != 2:
        errors.append(f"position shape is not [N,2]: {positions.shape}")
    if yaws.ndim != 1:
        errors.append(f"yaw shape is not [N]: {yaws.shape}")
    if not (len(images) == len(positions) == len(yaws)):
        errors.append("image/position/yaw lengths disagree")
    if not np.isfinite(positions).all() or not np.isfinite(yaws).all():
        errors.append("position/yaw contains non-finite values")
    image_shapes = set()
    for image_path in images:
        with Image.open(image_path) as image:
            image.load()
            image_shapes.add((image.mode, *image.size))
    if len(image_shapes) != 1:
        errors.append(f"image shapes/modes disagree: {sorted(image_shapes)}")
    canonical_count = max(0, len(images) - canonical_window_frames + 1)
    if canonical_count <= 0:
        errors.append("trajectory cannot form a canonical window")
    if metadata.get("canonical_window_count") != canonical_count:
        errors.append("source metadata canonical window count mismatch")
    nominal_dt = metadata.get("nominal_dt", {})
    dt = nominal_dt.get("median_s")
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        errors.append("nominal dt is missing, non-finite, or non-positive")
    policy = metadata.get("policy_metadata", {})
    if not policy.get("status"):
        errors.append("policy metadata status is not recorded")
    timestamp = metadata.get("timestamp", {})
    if not timestamp.get("status") and not timestamp.get("authority"):
        errors.append("timestamp provenance/status is not recorded")
    return {
        "trajectory_id": path.name,
        "frame_count": len(images),
        "canonical_window_count": canonical_count,
        "image_shapes": [list(item) for item in sorted(image_shapes)],
        "nominal_dt_s": dt,
        "nominal_dt_status": nominal_dt.get("status"),
        "timestamp": timestamp,
        "policy_metadata": policy,
        "spatial": metadata.get("spatial", {}),
        "source_session": metadata.get("source_session", path.name),
        "errors": errors,
        "passed": not errors,
    }


def build_report(
    dataset_id: str,
    dataset_root: Path,
    conversion_receipt_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    receipt = json.loads(conversion_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("dataset_id") != dataset_id:
        raise ValueError("conversion receipt dataset mismatch")
    entry = load_registry_entry(registry_path, dataset_id)
    canonical_window_frames = int(
        receipt.get("parameters", {}).get("canonical_window_frames", 0)
    )
    if canonical_window_frames < 2:
        raise ValueError("invalid canonical window in conversion receipt")
    trajectories = [
        audit_trajectory(path, canonical_window_frames)
        for path in sorted(item for item in dataset_root.iterdir() if item.is_dir())
    ]
    if not trajectories:
        raise ValueError("processed pilot contains no trajectories")
    _, tree_sha256, output_bytes = tree_inventory(dataset_root)
    observed_spacing = receipt["result"]["metric_waypoint_spacing_m_observed"]
    expected_spacing = entry["metric_waypoint_spacing_m"]
    relative_spacing_error = abs(observed_spacing - expected_spacing) / expected_spacing
    source_sessions = {item["source_session"] for item in trajectories}
    if dataset_id == "recon":
        session_valid = all(
            item["trajectory_id"] == item["source_session"] for item in trajectories
        )
        timestamp_validated = False
    else:
        session_valid = len(source_sessions) == 1 and all(
            item["trajectory_id"].startswith(item["source_session"] + "_")
            for item in trajectories
        )
        timestamp_validated = all(
            item["timestamp"].get("authority") == "rosbag_record_time"
            and len(item["timestamp"].get("record_times_s", [])) == item["frame_count"]
            and all(
                later > earlier
                for earlier, later in zip(
                    item["timestamp"].get("record_times_s", []),
                    item["timestamp"].get("record_times_s", [])[1:],
                )
            )
            for item in trajectories
        )
    policy_available = all(
        item["policy_metadata"].get("collection_policy_version") is not None
        for item in trajectories
    )
    gates = {
        "conversion_receipt_output_root_matches": (
            Path(receipt["output"]["root"]).resolve() == dataset_root.resolve()
        ),
        "conversion_receipt_tree_sha256_matches": (
            receipt["output"]["tree_sha256"] == tree_sha256
            and receipt["output"]["bytes"] == output_bytes
        ),
        "all_trajectories_content_valid": all(item["passed"] for item in trajectories),
        "canonical_windows_present": sum(
            item["canonical_window_count"] for item in trajectories
        ) > 0,
        "source_session_identity_valid": session_valid,
        "metric_spacing_within_50pct_of_registry_record": relative_spacing_error <= 0.5,
        "execution_boundary_preserved": all(
            receipt["execution_counts"].get(name) == 0
            for name in (
                "model_forward", "backward", "optimizer_steps",
                "training", "evaluation", "simulation",
            )
        ),
    }
    promotion_blockers = []
    if not timestamp_validated:
        promotion_blockers.append(
            "per-frame timestamps are absent from the selected RECON HDF5; dt is kinematically inferred"
        )
    if not policy_available:
        promotion_blockers.append(
            "collection policy/version metadata is not encoded in the selected raw artifact"
        )
    return {
        "schema_version": "0.1.0",
        "audit_id": f"{dataset_id}-converted-pilot",
        "dataset_id": dataset_id,
        "dataset_root": dataset_root.as_posix(),
        "conversion_receipt": {
            "path": conversion_receipt_path.as_posix(),
            "sha256": sha256_file(conversion_receipt_path),
        },
        "registry": {
            "path": registry_path.as_posix(),
            "metric_waypoint_spacing_m": expected_spacing,
        },
        "output": {
            "bytes": output_bytes,
            "tree_sha256": tree_sha256,
            "trajectory_count": len(trajectories),
            "frame_count": sum(item["frame_count"] for item in trajectories),
            "canonical_window_count": sum(
                item["canonical_window_count"] for item in trajectories
            ),
        },
        "timing": {
            "per_frame_timestamp_validated": timestamp_validated,
            "nominal_dt_s": receipt["result"]["nominal_dt_s"],
            "status": receipt["result"]["nominal_dt_status"],
        },
        "scale": {
            "registry_metric_waypoint_spacing_m": expected_spacing,
            "observed_median_step_m": observed_spacing,
            "relative_error": relative_spacing_error,
        },
        "policy_metadata_available": policy_available,
        "source_sessions": sorted(source_sessions),
        "trajectories": trajectories,
        "gates": gates,
        "blocking_gates": [name for name, passed in gates.items() if not passed],
        "passed": all(gates.values()),
        "promotion_blockers": promotion_blockers,
        "promotion_ready": all(gates.values()) and not promotion_blockers,
        "execution_counts": receipt["execution_counts"],
        "claim_boundary": (
            "Independent processed-pilot integrity and provenance audit only. A passing "
            "conversion gate does not erase explicitly listed timestamp/policy blockers "
            "or establish model, training, evaluation, or simulation results."
        ),
    }


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        f"# {report['dataset_id']} converted pilot audit",
        "",
        f"- Conversion integrity passed: {report['passed']}",
        f"- Promotion ready: {report['promotion_ready']}",
        f"- Trajectories: {report['output']['trajectory_count']}",
        f"- Frames: {report['output']['frame_count']}",
        f"- Canonical windows: {report['output']['canonical_window_count']}",
        f"- Nominal dt: {report['timing']['nominal_dt_s']} s ({report['timing']['status']})",
        f"- Observed median step: {report['scale']['observed_median_step_m']} m",
        "- Model/training/evaluation/simulation executed: False",
        "",
        "## Promotion blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in report["promotion_blockers"])
    if not report["promotion_blockers"]:
        lines.append("- None.")
    lines.extend(["", "## Evidence boundary", "", report["claim_boundary"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", choices=("recon", "huron_sacson"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--conversion-receipt", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(
            args.dataset_id,
            args.dataset_root.resolve(),
            args.conversion_receipt.resolve(),
            args.registry.resolve(),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, pickle.UnpicklingError) as exc:
        print(f"ERROR: {exc}")
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.out_md, report)
    print(
        f"{args.dataset_id} converted audit passed={report['passed']}, "
        f"promotion_ready={report['promotion_ready']}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
