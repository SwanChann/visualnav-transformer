#!/usr/bin/env python3
"""Read-only content audit for one RECON HDF5 or HuRoN ROS1 bag pilot.

The audit opens only the supplied raw artifact, samples the minimum content
needed to validate the legacy processor contract, and writes JSON/Markdown
receipts. It never invokes a converter or model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_recon(path: Path) -> dict[str, Any]:
    import h5py
    import numpy as np
    from PIL import Image

    required = ("jackal/position", "jackal/yaw", "images/rgb_left")
    with h5py.File(path, "r") as handle:
        missing = [name for name in required if name not in handle]
        if missing:
            return {
                "required_nodes": list(required),
                "missing_nodes": missing,
                "passed": False,
                "errors": [f"missing HDF5 node: {name}" for name in missing],
            }

        position = handle["jackal/position"]
        yaw = handle["jackal/yaw"]
        images = handle["images/rgb_left"]
        counts = {
            "position": int(position.shape[0]),
            "yaw": int(yaw.shape[0]),
            "rgb_left": int(images.shape[0]),
        }
        errors: list[str] = []
        if min(counts.values()) <= 0:
            errors.append(f"one or more required streams are empty: {counts}")
        if len(set(counts.values())) != 1:
            errors.append(f"required stream lengths disagree: {counts}")
        if len(position.shape) < 2 or position.shape[1] < 2:
            errors.append(f"position must have at least x/y columns: {position.shape}")

        sample_indices = sorted({0, max(0, counts["rgb_left"] // 2), max(0, counts["rgb_left"] - 1)})
        decoded_images: list[dict[str, Any]] = []
        finite_pose_samples = True
        if counts["rgb_left"] > 0:
            for index in sample_indices:
                try:
                    raw = images[index]
                    encoded = raw if isinstance(raw, bytes) else np.asarray(raw).tobytes()
                    with Image.open(io.BytesIO(encoded)) as image:
                        image.load()
                        decoded_images.append(
                            {
                                "index": index,
                                "format": image.format,
                                "mode": image.mode,
                                "size": list(image.size),
                                "encoded_bytes": len(encoded),
                            }
                        )
                    pose = np.asarray(position[index, :2], dtype=np.float64)
                    yaw_value = float(np.asarray(yaw[index]).reshape(-1)[0])
                    finite_pose_samples = finite_pose_samples and bool(
                        np.isfinite(pose).all() and math.isfinite(yaw_value)
                    )
                except (OSError, ValueError, TypeError, IndexError) as exc:
                    errors.append(f"sample {index} failed to decode/read: {exc}")
        if len(decoded_images) != len(sample_indices):
            errors.append("not all sampled rgb_left images decoded")
        if not finite_pose_samples:
            errors.append("sampled position/yaw contains non-finite values")

        return {
            "required_nodes": list(required),
            "missing_nodes": [],
            "stream_shapes": {
                name: list(handle[node].shape)
                for name, node in {
                    "position": "jackal/position",
                    "yaw": "jackal/yaw",
                    "rgb_left": "images/rgb_left",
                }.items()
            },
            "stream_dtypes": {
                name: str(handle[node].dtype)
                for name, node in {
                    "position": "jackal/position",
                    "yaw": "jackal/yaw",
                    "rgb_left": "images/rgb_left",
                }.items()
            },
            "counts": counts,
            "sample_indices": sample_indices,
            "decoded_images": decoded_images,
            "finite_pose_samples": finite_pose_samples,
            "errors": errors,
            "passed": not errors,
        }


def audit_huron(path: Path) -> dict[str, Any]:
    import rosbag
    from PIL import Image

    required = {
        "/fisheye_image/compressed": "sensor_msgs/CompressedImage",
        "/odometry": "nav_msgs/Odometry",
    }
    errors: list[str] = []
    with rosbag.Bag(str(path), "r") as bag:
        info = bag.get_type_and_topic_info()
        topics: dict[str, Any] = {}
        for topic, topic_info in sorted(info.topics.items()):
            topics[topic] = {
                "message_type": topic_info.msg_type,
                "message_count": int(topic_info.message_count),
                "connections": int(topic_info.connections),
                "frequency_hz": (
                    float(topic_info.frequency)
                    if topic_info.frequency is not None
                    and math.isfinite(float(topic_info.frequency))
                    else None
                ),
            }
        for topic, expected_type in required.items():
            actual = topics.get(topic)
            if actual is None:
                errors.append(f"missing required topic: {topic}")
            elif actual["message_type"] != expected_type:
                errors.append(
                    f"topic type mismatch for {topic}: {actual['message_type']} != {expected_type}"
                )
            elif actual["message_count"] <= 0:
                errors.append(f"required topic is empty: {topic}")

        start = float(bag.get_start_time())
        end = float(bag.get_end_time())
        duration = end - start
        if not math.isfinite(duration) or duration <= 0:
            errors.append(f"invalid bag duration: {duration}")

        image_sample: dict[str, Any] | None = None
        odometry_sample: dict[str, Any] | None = None
        try:
            for topic, message, stamp in bag.read_messages(topics=list(required)):
                if topic == "/fisheye_image/compressed" and image_sample is None:
                    with Image.open(io.BytesIO(bytes(message.data))) as image:
                        image.load()
                        image_sample = {
                            "stamp_s": float(stamp.to_sec()),
                            "format_field": str(message.format),
                            "decoded_format": image.format,
                            "mode": image.mode,
                            "size": list(image.size),
                            "encoded_bytes": len(message.data),
                        }
                elif topic == "/odometry" and odometry_sample is None:
                    position = message.pose.pose.position
                    orientation = message.pose.pose.orientation
                    values = [
                        position.x,
                        position.y,
                        position.z,
                        orientation.x,
                        orientation.y,
                        orientation.z,
                        orientation.w,
                    ]
                    finite = all(math.isfinite(float(value)) for value in values)
                    odometry_sample = {
                        "stamp_s": float(stamp.to_sec()),
                        "frame_id": str(message.header.frame_id),
                        "child_frame_id": str(message.child_frame_id),
                        "finite_pose": finite,
                    }
                if image_sample is not None and odometry_sample is not None:
                    break
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"sample message decode failed: {exc}")
        if image_sample is None:
            errors.append("no fisheye image sample decoded")
        if odometry_sample is None:
            errors.append("no odometry sample decoded")
        elif not odometry_sample["finite_pose"]:
            errors.append("sample odometry pose contains non-finite values")

        return {
            "required_topics": required,
            "all_topics": topics,
            "start_time_s": start,
            "end_time_s": end,
            "duration_s": duration,
            "message_count": sum(item["message_count"] for item in topics.values()),
            "image_sample": image_sample,
            "odometry_sample": odometry_sample,
            "errors": errors,
            "passed": not errors,
        }


def build_report(dataset_id: str, artifact: Path) -> dict[str, Any]:
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if dataset_id == "recon":
        content = audit_recon(artifact)
    elif dataset_id == "huron_sacson":
        content = audit_huron(artifact)
    else:
        raise ValueError(f"unsupported dataset_id: {dataset_id}")
    return {
        "schema_version": "0.1.0",
        "audit_id": f"{dataset_id}-raw-pilot-content",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "artifact": {
            "path": str(artifact),
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        },
        "content": content,
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
        "passed": content["passed"],
        "claim_boundary": (
            "Raw pilot content/readability audit only. It does not validate the full "
            "dataset, metric scale, collection-session completeness, conversion output, "
            "model behavior, training, evaluation, or simulation."
        ),
    }


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    content = report["content"]
    lines = [
        f"# {report['dataset_id']} raw pilot content audit",
        "",
        f"- Passed: {report['passed']}",
        f"- Artifact: `{report['artifact']['path']}`",
        f"- Bytes: {report['artifact']['bytes']}",
        f"- SHA-256: `{report['artifact']['sha256']}`",
        "- Conversion/model/training/evaluation/simulation executed: False",
        "",
        "## Content",
        "",
        "```json",
        json.dumps(content, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Evidence boundary",
        "",
        report["claim_boundary"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", choices=("recon", "huron_sacson"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args.dataset_id, args.artifact.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.out_md, report)
    print(f"{args.dataset_id} raw content audit passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
