#!/usr/bin/env python3
"""Read-only, content-level audit for the materialized Go Stanford pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from go_stanford_adapter import (
    GO_STANFORD_DATASET_ID,
    GoStanfordDataError,
    build_go_stanford_canonical_sample,
    load_go_stanford_trajectory,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise GoStanfordDataError(f"manifest has no rows: {path}")
    return rows


def load_contract_scale(registry_path: Path, data_config_path: Path) -> dict[str, float]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matches = [
        item for item in registry.get("datasets", []) if item.get("dataset_id") == GO_STANFORD_DATASET_ID
    ]
    if len(matches) != 1:
        raise GoStanfordDataError("registry must contain exactly one go_stanford entry")
    config = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
    return {
        "registry_waypoint_spacing_m": float(matches[0]["metric_waypoint_spacing_m"]),
        "registry_nominal_dt_s": float(matches[0]["nominal_dt_s"]),
        "loader_waypoint_spacing_m": float(config[GO_STANFORD_DATASET_ID]["metric_waypoint_spacing"]),
    }


def split_sha256(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (item["dataset_id"], item["trajectory_id"])):
        digest.update(
            f"{row['dataset_id']}\0{row['trajectory_id']}\0{row['leakage_group']}\0{row['split']}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_worktree_dirty(repo_root: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _record_error(errors: Counter[str], samples: dict[str, list[str]], code: str, detail: str) -> None:
    errors[code] += 1
    if len(samples.setdefault(code, [])) < 20:
        samples[code].append(detail)


def audit_pilot(
    *,
    repo_root: Path,
    dataset_root: Path,
    manifest_path: Path,
    registry_path: Path,
    data_config_path: Path,
    decode_all_images: bool,
    observation_frames: int,
    action_history_steps: int,
    action_horizon: int,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    rows = [row for row in load_manifest(manifest_path) if row["dataset_id"] == GO_STANFORD_DATASET_ID]
    manifest_by_id = {row["trajectory_id"]: row for row in rows}
    errors: Counter[str] = Counter()
    error_samples: dict[str, list[str]] = {}
    image_shapes: Counter[str] = Counter()
    image_modes: Counter[str] = Counter()
    steps: list[np.ndarray] = []
    decoded_images = 0
    backup_matches = 0
    eligible_trajectories = 0
    eligible_samples = 0
    dataset_tree = hashlib.sha256()
    canonical_sample: dict[str, Any] | None = None

    actual_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    actual_ids = {path.name for path in actual_dirs}
    manifest_ids = set(manifest_by_id)
    for missing in sorted(manifest_ids - actual_ids):
        _record_error(errors, error_samples, "manifest_path_missing", missing)
    for missing in sorted(actual_ids - manifest_ids):
        _record_error(errors, error_samples, "trajectory_missing_from_manifest", missing)

    for trajectory_dir in actual_dirs:
        trajectory_id = trajectory_dir.name
        row = manifest_by_id.get(trajectory_id)
        try:
            trajectory = load_go_stanford_trajectory(trajectory_dir)
        except GoStanfordDataError as exc:
            _record_error(errors, error_samples, "trajectory_contract", f"{trajectory_id}: {exc}")
            continue
        if row is None:
            continue
        expected_frames = int(row["num_frames"])
        if expected_frames != trajectory.frame_count:
            _record_error(
                errors,
                error_samples,
                "manifest_frame_count",
                f"{trajectory_id}: manifest={expected_frames}, actual={trajectory.frame_count}",
            )
        metadata_path = trajectory_dir / "traj_data.pkl"
        metadata_sha = sha256_file(metadata_path)
        if metadata_sha != row["trajectory_meta_sha256"]:
            _record_error(errors, error_samples, "metadata_sha256", trajectory_id)
        backup_path = trajectory_dir / "traj_data.pkl.backup"
        if not backup_path.is_file():
            _record_error(errors, error_samples, "missing_metadata_backup", trajectory_id)
        elif sha256_file(backup_path) != metadata_sha:
            _record_error(errors, error_samples, "metadata_backup_mismatch", trajectory_id)
        else:
            backup_matches += 1

        dataset_tree.update(
            f"{trajectory_id}/traj_data.pkl\0{metadata_path.stat().st_size}\0{metadata_sha}\n".encode()
        )
        if trajectory.frame_count > 1:
            steps.append(np.linalg.norm(np.diff(trajectory.positions_xy_m, axis=0), axis=1))
        minimum_length = observation_frames + action_horizon
        if trajectory.frame_count >= minimum_length:
            eligible_trajectories += 1
            eligible_samples += trajectory.frame_count - minimum_length + 1
            if canonical_sample is None:
                try:
                    batch = build_go_stanford_canonical_sample(
                        trajectory,
                        current_index=observation_frames - 1,
                        observation_frames=observation_frames,
                        action_history_steps=action_history_steps,
                        action_horizon=action_horizon,
                        image_size=image_size,
                    )
                    canonical_sample = {
                        "trajectory_id": trajectory_id,
                        "current_index": observation_frames - 1,
                        "obs_shape": list(batch["obs_images"].shape),
                        "goal_shape": list(batch["goal_image"].shape),
                        "history_shape": list(batch["action_history"].shape),
                        "target_shape": list(batch["target_waypoints"].shape),
                        "target_unit": "meter",
                        "dt_s": float(batch["dt_s"][0, 0]),
                        "waypoint_spacing_m": float(batch["waypoint_spacing_m"][0, 0]),
                    }
                except GoStanfordDataError as exc:
                    _record_error(
                        errors, error_samples, "canonical_sample", f"{trajectory_id}: {exc}"
                    )

        for image_path in trajectory.image_paths:
            image_sha = sha256_file(image_path)
            dataset_tree.update(
                f"{trajectory_id}/{image_path.name}\0{image_path.stat().st_size}\0{image_sha}\n".encode()
            )
            if not decode_all_images:
                continue
            try:
                with Image.open(image_path) as image:
                    image.load()
                    image_shapes[f"{image.width}x{image.height}"] += 1
                    image_modes[image.mode] += 1
                decoded_images += 1
            except Exception as exc:
                _record_error(
                    errors,
                    error_samples,
                    "image_decode",
                    f"{trajectory_id}/{image_path.name}: {type(exc).__name__}: {exc}",
                )

    all_steps = np.concatenate(steps) if steps else np.asarray([], dtype=np.float64)
    positive_steps = all_steps[all_steps > 0]
    scale = load_contract_scale(registry_path, data_config_path)
    manifest_scales = sorted({float(row["metric_waypoint_spacing_m"]) for row in rows})
    contract_values = manifest_scales + [
        scale["registry_waypoint_spacing_m"],
        scale["loader_waypoint_spacing_m"],
    ]
    scale_consistent = len({round(value, 12) for value in contract_values}) == 1
    if not scale_consistent:
        _record_error(errors, error_samples, "metric_scale_contract", repr(contract_values))

    return {
        "schema_version": "0.1.0",
        "audit_id": "go-stanford-live-data-pilot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_boundary": {
            "training_performed": False,
            "model_forward_performed": False,
            "data_downloaded": False,
            "dataset_mutated": False,
            "full_image_decode": decode_all_images,
        },
        "provenance": {
            "git_commit": git_head(repo_root),
            "tracked_worktree_dirty": git_worktree_dirty(repo_root),
            "adapter_sha256": sha256_file(Path(__file__).with_name("go_stanford_adapter.py")),
            "audit_script_sha256": sha256_file(Path(__file__)),
            "dataset_root": str(dataset_root.resolve()),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_text_file(manifest_path),
            "split_sha256": split_sha256(rows),
            "canonical_dataset_tree_sha256": dataset_tree.hexdigest(),
            "tree_scope": "sorted trajectory_id; traj_data.pkl and numbered jpg content; backups excluded",
            "raw_artifact_receipt_present": False,
            "processor_version_pinned": False,
        },
        "counts": {
            "manifest_rows": len(rows),
            "trajectory_directories": len(actual_dirs),
            "metadata_backup_matches": backup_matches,
            "images_expected": sum(int(row["num_frames"]) for row in rows),
            "images_decoded": decoded_images,
            "eligible_trajectories": eligible_trajectories,
            "eligible_canonical_samples": eligible_samples,
        },
        "image_contract": {
            "shape_counts": dict(sorted(image_shapes.items())),
            "mode_counts": dict(sorted(image_modes.items())),
        },
        "pose_contract": {
            "position_unit_from_contract": "meter",
            "yaw_unit_from_contract": "radian",
            "independent_metric_calibration_performed": False,
            "step_count": int(all_steps.size),
            "zero_step_count": int(np.count_nonzero(all_steps == 0)),
            "positive_step_quantiles_m": (
                {
                    "p01": float(np.quantile(positive_steps, 0.01)),
                    "p50": float(np.quantile(positive_steps, 0.50)),
                    "p99": float(np.quantile(positive_steps, 0.99)),
                    "max": float(np.max(positive_steps)),
                }
                if positive_steps.size
                else {}
            ),
        },
        "scale_contract": {
            **scale,
            "manifest_waypoint_spacing_m_values": manifest_scales,
            "consistent": scale_consistent,
            "note": "Contract agreement is not independent physical calibration.",
        },
        "timestamp_contract": {
            "nominal_dt_s": scale["registry_nominal_dt_s"],
            "per_frame_timestamps_present": False,
            "monotonicity_verified": False,
        },
        "canonical_sample": canonical_sample,
        "errors": {
            "count_by_code": dict(sorted(errors.items())),
            "samples": error_samples,
        },
        "passed": not errors and canonical_sample is not None,
        "claim_boundary": (
            "This audit establishes processed-data readability and contract consistency only; "
            "it is not training, model-quality evidence, independent physical calibration, or a "
            "raw-artifact provenance receipt."
        ),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    counts = report["counts"]
    provenance = report["provenance"]
    scale = report["scale_contract"]
    timestamps = report["timestamp_contract"]
    errors = report["errors"]["count_by_code"]
    lines = [
        "# Go Stanford Live DATA-PILOT Audit",
        "",
        f"- Passed: **{report['passed']}**",
        f"- Git commit: `{provenance['git_commit']}`",
        f"- Dataset root: `{provenance['dataset_root']}`",
        f"- Manifest rows / trajectory directories: {counts['manifest_rows']} / {counts['trajectory_directories']}",
        f"- Expected / decoded images: {counts['images_expected']} / {counts['images_decoded']}",
        f"- Canonical-sample-eligible trajectories: {counts['eligible_trajectories']}",
        f"- Canonical samples available under the frozen 6+8 frame geometry: {counts['eligible_canonical_samples']}",
        f"- Manifest SHA-256: `{provenance['manifest_sha256']}`",
        f"- Split SHA-256: `{provenance['split_sha256']}`",
        f"- Canonical dataset tree SHA-256: `{provenance['canonical_dataset_tree_sha256']}`",
        "",
        "## Contract checks",
        "",
        f"- Scale agreement: {scale['consistent']} (manifest/registry/loader = 0.12 m).",
        f"- Metadata backups matching primary pickle: {counts['metadata_backup_matches']}.",
        f"- Per-frame timestamps present: {timestamps['per_frame_timestamps_present']}.",
        f"- Timestamp monotonicity verified: {timestamps['monotonicity_verified']}.",
        f"- Raw artifact receipt present: {provenance['raw_artifact_receipt_present']}.",
        f"- Processor version pinned: {provenance['processor_version_pinned']}.",
        "",
        "## Errors",
        "",
    ]
    if errors:
        lines.extend(f"- `{code}`: {count}" for code, count in errors.items())
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            report["claim_boundary"],
            "",
            "The full DATA-PILOT backlog item remains incomplete while RECON and HuRoN are not authorized/materialized.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--decode-all-images", action="store_true")
    parser.add_argument("--observation-frames", type=int, default=6)
    parser.add_argument("--action-history-steps", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=96)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit_pilot(
            repo_root=args.repo_root,
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            registry_path=args.registry,
            data_config_path=args.data_config,
            decode_all_images=args.decode_all_images,
            observation_frames=args.observation_frames,
            action_history_steps=args.action_history_steps,
            action_horizon=args.action_horizon,
            image_size=(args.image_height, args.image_width),
        )
    except (GoStanfordDataError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, report)
    print(
        f"Audited {report['counts']['trajectory_directories']} trajectories / "
        f"{report['counts']['images_decoded']} decoded images: passed={report['passed']}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
