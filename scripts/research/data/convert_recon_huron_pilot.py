#!/usr/bin/env python3
"""Convert one governed RECON HDF5 or HuRoN bag into an isolated ViNT pilot.

The converter is intentionally single-artifact, refuses overwrite, records raw
lineage and timing/scale limitations, and never imports a model or training code.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    import numpy as np

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def tree_inventory(root: Path) -> tuple[list[dict[str, Any]], str, int]:
    files = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    encoded = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return files, hashlib.sha256(encoded).hexdigest(), total_bytes


def yaw_from_quaternion(orientation: Any) -> float:
    t3 = 2.0 * (
        orientation.w * orientation.z + orientation.x * orientation.y
    )
    t4 = 1.0 - 2.0 * (
        orientation.y * orientation.y + orientation.z * orientation.z
    )
    return math.atan2(t3, t4)


def forward_segment_indices(
    positions: Any,
    yaws: Any,
    *,
    epsilon_m: float = 1e-5,
    min_frames: int = 2,
) -> list[list[int]]:
    """Return maximal forward-motion runs while retaining both edge endpoints."""
    import numpy as np

    positions = np.asarray(positions, dtype=np.float64)
    yaws = np.asarray(yaws, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("positions must have shape [N, >=2]")
    if len(positions) != len(yaws):
        raise ValueError("position/yaw length mismatch")
    segments: list[list[int]] = []
    current: list[int] = []
    for index in range(len(positions) - 1):
        delta = positions[index + 1, :2] - positions[index, :2]
        projection = float(
            delta[0] * math.cos(float(yaws[index]))
            + delta[1] * math.sin(float(yaws[index]))
        )
        if projection >= epsilon_m:
            if not current:
                current = [index]
            current.append(index + 1)
        elif current:
            if len(current) >= min_frames:
                segments.append(current)
            current = []
    if current and len(current) >= min_frames:
        segments.append(current)
    return segments


def spatial_stats(positions: Any) -> dict[str, Any]:
    import numpy as np

    positions = np.asarray(positions, dtype=np.float64)[:, :2]
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if not len(steps):
        return {
            "step_count": 0,
            "median_step_m": None,
            "mean_step_m": None,
            "min_step_m": None,
            "max_step_m": None,
            "path_length_m": 0.0,
        }
    return {
        "step_count": int(len(steps)),
        "median_step_m": float(np.median(steps)),
        "mean_step_m": float(np.mean(steps)),
        "min_step_m": float(np.min(steps)),
        "max_step_m": float(np.max(steps)),
        "path_length_m": float(np.sum(steps)),
    }


def write_trajectory(
    trajectory_root: Path,
    images: Iterable[bytes],
    positions: Any,
    yaws: Any,
    source_metadata: dict[str, Any],
) -> int:
    import numpy as np
    from PIL import Image

    image_bytes = list(images)
    positions = np.asarray(positions, dtype=np.float64)[:, :2]
    yaws = np.asarray(yaws, dtype=np.float64)
    if not (len(image_bytes) == len(positions) == len(yaws)):
        raise ValueError("image/position/yaw length mismatch")
    trajectory_root.mkdir(parents=True, exist_ok=False)
    for index, encoded in enumerate(image_bytes):
        with Image.open(io.BytesIO(encoded)) as image:
            image.load()
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(trajectory_root / f"{index}.jpg", format="JPEG")
    with (trajectory_root / "traj_data.pkl").open("wb") as handle:
        pickle.dump(
            {"position": positions, "yaw": yaws},
            handle,
            protocol=4,
        )
    (trajectory_root / "source_metadata.json").write_text(
        json.dumps(source_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(image_bytes)


def convert_recon(
    artifact: Path,
    output_root: Path,
    min_frames: int,
    canonical_window_frames: int,
) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(artifact, "r") as handle:
        required = ("jackal/position", "jackal/yaw", "images/rgb_left")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"missing RECON nodes: {missing}")
        positions = np.asarray(handle["jackal/position"][:, :2], dtype=np.float64)
        yaws = np.asarray(handle["jackal/yaw"][:], dtype=np.float64)
        images = [bytes(item) for item in handle["images/rgb_left"][:]]
        attributes = {key: json_value(value) for key, value in handle.attrs.items()}
        linear_velocity = (
            np.asarray(handle["jackal/linear_velocity"][:], dtype=np.float64)
            if "jackal/linear_velocity" in handle
            else None
        )
    if not (len(images) == len(positions) == len(yaws)):
        raise ValueError("RECON stream lengths disagree")
    if len(images) < min_frames:
        raise ValueError(f"RECON pilot has {len(images)} frames; requires {min_frames}")
    if not np.isfinite(positions).all() or not np.isfinite(yaws).all():
        raise ValueError("RECON pose contains non-finite values")

    dt_estimates = []
    if linear_velocity is not None and len(linear_velocity) == len(positions):
        speeds = (linear_velocity[:-1] + linear_velocity[1:]) / 2.0
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        valid = np.isfinite(speeds) & (np.abs(speeds) > 1e-3)
        dt_estimates = (distances[valid] / np.abs(speeds[valid])).tolist()
    metadata = {
        "schema_version": "0.1.0",
        "dataset_id": "recon",
        "source_artifact": artifact.as_posix(),
        "source_artifact_sha256": sha256_file(artifact),
        "source_attributes": attributes,
        "frame_count": len(images),
        "canonical_window_frames": canonical_window_frames,
        "canonical_window_count": max(0, len(images) - canonical_window_frames + 1),
        "timestamp": {
            "per_frame_timestamp_present": False,
            "status": "not_present_in_selected_hdf5",
        },
        "nominal_dt": {
            "status": "kinematic_estimate_not_timestamp_validated",
            "median_s": float(np.median(dt_estimates)) if dt_estimates else None,
            "sample_count": len(dt_estimates),
        },
        "spatial": spatial_stats(positions),
        "policy_metadata": {
            "rollout_number": attributes.get("rollout_number"),
            "source_bag_names": attributes.get("inorder_rosbag_fnames"),
            "collection_policy_version": None,
            "status": "not_present_in_selected_hdf5",
        },
    }
    trajectory_id = artifact.stem
    count = write_trajectory(
        output_root / trajectory_id, images, positions, yaws, metadata
    )
    return {
        "trajectory_count": 1,
        "trajectory_ids": [trajectory_id],
        "frame_count": count,
        "canonical_window_count": metadata["canonical_window_count"],
        "nominal_dt_s": metadata["nominal_dt"]["median_s"],
        "nominal_dt_status": metadata["nominal_dt"]["status"],
        "metric_waypoint_spacing_m_observed": metadata["spatial"]["median_step_m"],
        "policy_metadata_status": metadata["policy_metadata"]["status"],
    }


def convert_huron(
    artifact: Path,
    output_root: Path,
    sample_rate_hz: float,
    min_frames: int,
    canonical_window_frames: int,
) -> dict[str, Any]:
    import numpy as np
    import rosbag
    from PIL import Image

    image_topic = "/fisheye_image/compressed"
    odom_topic = "/odometry"
    records = []
    image_latest = None
    odom_latest = None
    with rosbag.Bag(str(artifact), "r") as bag:
        topic_info = bag.get_type_and_topic_info().topics
        for topic in (image_topic, odom_topic):
            if topic not in topic_info or topic_info[topic].message_count <= 0:
                raise ValueError(f"missing HuRoN topic: {topic}")
        current_time = float(bag.get_start_time())
        for topic, message, stamp in bag.read_messages(topics=[image_topic, odom_topic]):
            record_time = float(stamp.to_sec())
            if topic == image_topic:
                image_latest = (bytes(message.data), record_time, float(message.header.stamp.to_sec()))
            else:
                position = message.pose.pose.position
                odom_latest = (
                    [float(position.x), float(position.y)],
                    yaw_from_quaternion(message.pose.pose.orientation),
                    record_time,
                    float(message.header.stamp.to_sec()),
                    str(message.header.frame_id),
                    str(message.child_frame_id),
                )
            if record_time - current_time >= 1.0 / sample_rate_hz:
                if image_latest is not None and odom_latest is not None:
                    with Image.open(io.BytesIO(image_latest[0])) as image:
                        image.load()
                    records.append((image_latest, odom_latest, record_time))
                    current_time = record_time
    if len(records) < min_frames:
        raise ValueError(f"HuRoN synchronized pilot has {len(records)} frames; requires {min_frames}")
    images = [item[0][0] for item in records]
    positions = np.asarray([item[1][0] for item in records], dtype=np.float64)
    yaws = np.asarray([item[1][1] for item in records], dtype=np.float64)
    segments = forward_segment_indices(
        positions, yaws, min_frames=min_frames
    )
    if not segments:
        raise ValueError("HuRoN pilot has no sufficiently long forward segment")

    trajectory_ids = []
    total_frames = 0
    total_windows = 0
    dt_values = []
    observed_steps = []
    source_session = f"{artifact.parent.name}_{artifact.stem}"
    for segment_index, indices in enumerate(segments):
        segment_positions = positions[indices]
        segment_yaws = yaws[indices]
        segment_images = [images[index] for index in indices]
        anchor_times = [records[index][2] for index in indices]
        record_deltas = np.diff(anchor_times)
        dt_values.extend(record_deltas.tolist())
        stats = spatial_stats(segment_positions)
        if stats["median_step_m"] is not None:
            observed_steps.append(stats["median_step_m"])
        trajectory_id = f"{source_session}_{segment_index}"
        trajectory_ids.append(trajectory_id)
        metadata = {
            "schema_version": "0.1.0",
            "dataset_id": "huron_sacson",
            "source_artifact": artifact.as_posix(),
            "source_artifact_sha256": sha256_file(artifact),
            "source_session": source_session,
            "segment_index": segment_index,
            "source_indices": indices,
            "frame_count": len(indices),
            "canonical_window_frames": canonical_window_frames,
            "canonical_window_count": max(0, len(indices) - canonical_window_frames + 1),
            "timestamp": {
                "authority": "rosbag_record_time",
                "record_times_s": anchor_times,
                "image_record_times_s": [records[index][0][1] for index in indices],
                "odom_record_times_s": [records[index][1][2] for index in indices],
                "image_header_times_s": [records[index][0][2] for index in indices],
                "odom_header_times_s": [records[index][1][3] for index in indices],
                "header_record_clock_mismatch_present": True,
            },
            "nominal_dt": {
                "status": "measured_from_rosbag_record_time_after_sampling",
                "requested_sample_rate_hz": sample_rate_hz,
                "target_s": 1.0 / sample_rate_hz,
                "median_s": float(np.median(record_deltas)),
                "min_s": float(np.min(record_deltas)),
                "max_s": float(np.max(record_deltas)),
            },
            "spatial": stats,
            "frames": {
                "odom_frame_id": records[indices[0]][1][4],
                "odom_child_frame_id": records[indices[0]][1][5],
            },
            "policy_metadata": {
                "collection_policy_version": None,
                "interaction_loss_enabled": None,
                "status": "not_encoded_in_selected_bag",
            },
        }
        total_frames += write_trajectory(
            output_root / trajectory_id,
            segment_images,
            segment_positions,
            segment_yaws,
            metadata,
        )
        total_windows += metadata["canonical_window_count"]
    return {
        "trajectory_count": len(trajectory_ids),
        "trajectory_ids": trajectory_ids,
        "synchronized_frame_count_before_filter": len(records),
        "frame_count": total_frames,
        "canonical_window_count": total_windows,
        "nominal_dt_s": float(np.median(dt_values)),
        "nominal_dt_status": "measured_from_rosbag_record_time_after_sampling",
        "metric_waypoint_spacing_m_observed": float(np.median(observed_steps)),
        "policy_metadata_status": "not_encoded_in_selected_bag",
    }


def convert(args: argparse.Namespace) -> dict[str, Any]:
    artifact = args.artifact.resolve()
    output_root = args.output_root.resolve()
    raw_receipt = args.raw_receipt.resolve()
    if not artifact.is_file() or not raw_receipt.is_file():
        raise FileNotFoundError("artifact and raw receipt must exist")
    raw_receipt_payload = json.loads(raw_receipt.read_text(encoding="utf-8"))
    receipt_artifact = raw_receipt_payload.get("artifact", {})
    artifact_sha256 = sha256_file(artifact)
    if raw_receipt_payload.get("dataset_id") != args.dataset_id:
        raise ValueError("raw receipt dataset_id does not match conversion dataset")
    if (
        receipt_artifact.get("bytes") != artifact.stat().st_size
        or receipt_artifact.get("sha256") != artifact_sha256
    ):
        raise ValueError("raw receipt does not bind the selected artifact bytes")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    if args.receipt_out.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.receipt_out}")
    temporary = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        if args.dataset_id == "recon":
            result = convert_recon(
                artifact, temporary, args.min_frames, args.canonical_window_frames
            )
        else:
            result = convert_huron(
                artifact,
                temporary,
                args.sample_rate_hz,
                args.min_frames,
                args.canonical_window_frames,
            )
        files, tree_sha256, output_bytes = tree_inventory(temporary)
        if output_bytes > args.budget_bytes:
            raise ValueError(
                f"converted output exceeds budget: {output_bytes} > {args.budget_bytes}"
            )
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    script_path = Path(__file__).resolve()
    receipt = {
        "schema_version": "0.1.0",
        "receipt_type": "isolated_processed_pilot_conversion",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": args.dataset_id,
        "raw_artifact": {
            "path": artifact.as_posix(),
            "bytes": artifact.stat().st_size,
            "sha256": artifact_sha256,
        },
        "raw_receipt": {
            "path": raw_receipt.as_posix(),
            "sha256": sha256_file(raw_receipt),
        },
        "processor": {
            "path": script_path.as_posix(),
            "sha256": sha256_file(script_path),
            "legacy_reference": (
                "train/process_recon.py"
                if args.dataset_id == "recon"
                else "train/process_bags.py"
            ),
        },
        "parameters": {
            "min_frames": args.min_frames,
            "canonical_window_frames": args.canonical_window_frames,
            "sample_rate_hz": (
                args.sample_rate_hz if args.dataset_id == "huron_sacson" else None
            ),
            "budget_bytes": args.budget_bytes,
        },
        "result": result,
        "output": {
            "root": output_root.as_posix(),
            "bytes": output_bytes,
            "tree_sha256": tree_sha256,
            "files": files,
        },
        "execution_counts": {
            "raw_artifact_conversions": 1,
            "model_forward": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "training": 0,
            "evaluation": 0,
            "simulation": 0,
        },
        "claim_boundary": (
            "One isolated processed pilot only. Timing/scale fields retain their "
            "measured or inferred status; missing policy metadata remains missing. "
            "No model execution, training, evaluation, or simulation was performed."
        ),
    }
    args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_out.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", choices=("recon", "huron_sacson"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=14)
    parser.add_argument("--canonical-window-frames", type=int, default=14)
    parser.add_argument("--sample-rate-hz", type=float, default=4.0)
    parser.add_argument("--budget-bytes", type=int, default=10_000_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_frames < 2 or args.canonical_window_frames < 2:
        print("ERROR: frame thresholds must be >= 2")
        return 2
    if not math.isfinite(args.sample_rate_hz) or args.sample_rate_hz <= 0:
        print("ERROR: sample rate must be positive and finite")
        return 2
    try:
        receipt = convert(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        f"{args.dataset_id} conversion complete: "
        f"trajectories={receipt['result']['trajectory_count']}, "
        f"frames={receipt['result']['frame_count']}, "
        f"bytes={receipt['output']['bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
