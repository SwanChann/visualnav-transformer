#!/usr/bin/env python3
"""Run the frozen, no-training RA-L controlled MuJoCo matrix."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for candidate in (SCRIPTS_ROOT, SCRIPTS_ROOT / "simulation", SCRIPTS_ROOT / "analysis"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

DEFAULT_PROTOCOL = REPO_ROOT / "后续研究内容" / "benchmark" / "ral_mujoco_protocol_v0.1.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "research" / "ral_mujoco_controlled"
VALID_FINAL_STATUSES = {"completed", "failed", "crashed", "infrastructure_invalid"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_file(path: Path) -> str:
    """Hash UTF-8 text with LF line endings on every operating system."""
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Protocol hashes are content identities, so Git's platform-specific
    # checkout line endings must not change them. Binary assets remain raw-byte
    # hashed by ``sha256_file`` in ``verify_immutable_files``.
    protocol_sha256 = sha256_text_file(path)
    if "base_protocol_path" in payload:
        base_path = (REPO_ROOT / payload["base_protocol_path"]).resolve()
        expected_base_hash = payload.get("base_protocol_sha256")
        actual_base_hash = sha256_text_file(base_path)
        if actual_base_hash != expected_base_hash:
            raise ValueError(f"Base protocol hash mismatch: {actual_base_hash} != {expected_base_hash}")
        base = json.loads(base_path.read_text(encoding="utf-8"))
        protocol = copy.deepcopy(base)
        overrides = payload.get("overrides", {})
        allowed = {"timeout_max_steps", "simulated_navigation_seconds", "scene_ids", "planned_trial_count_before_gate", "policy_action_scale_m"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"Unsupported protocol overlay fields: {sorted(unknown)}")
        scene_ids = overrides.get("scene_ids")
        if scene_ids is not None:
            requested = list(scene_ids)
            protocol["scenes"] = [scene for scene in protocol["scenes"] if scene["id"] in requested]
            if [scene["id"] for scene in protocol["scenes"]] != requested:
                by_id = {scene["id"]: scene for scene in protocol["scenes"]}
                protocol["scenes"] = [by_id[scene_id] for scene_id in requested if scene_id in by_id]
            if len(protocol["scenes"]) != len(requested):
                raise ValueError("Protocol overlay references an unknown scene ID.")
            protocol["planned_matrix"]["scene_count_before_gate"] = len(protocol["scenes"])
        if "timeout_max_steps" in overrides:
            protocol["navigation_contract"]["timeout"]["max_steps"] = int(overrides["timeout_max_steps"])
        if "simulated_navigation_seconds" in overrides:
            protocol["navigation_contract"]["timeout"]["simulated_navigation_seconds"] = float(overrides["simulated_navigation_seconds"])
        if "planned_trial_count_before_gate" in overrides:
            protocol["planned_matrix"]["planned_trial_count_before_gate"] = int(overrides["planned_trial_count_before_gate"])
        if "policy_action_scale_m" in overrides:
            scale = float(overrides["policy_action_scale_m"])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("policy_action_scale_m must be finite and positive")
            protocol["navigation_contract"]["policy_action_scale_m"] = scale
        protocol["protocol_id"] = payload["protocol_id"]
        protocol["status"] = payload["status"]
        protocol["overlay_metadata"] = payload
        payload = protocol
    if payload.get("status") not in {"frozen_before_baseline_calibration", "frozen_before_full_matrix"}:
        raise ValueError("Controlled protocol is not frozen.")
    return payload, protocol_sha256


def _run_git(*args: str, binary: bool = False):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True,
        text=not binary,
    ).stdout


def git_metadata() -> dict[str, Any]:
    commit = _run_git("rev-parse", "HEAD").strip()
    source_pathspec = ("--", ".", ":(exclude)results/**")
    status_bytes = _run_git("status", "--porcelain=v1", "-z", "--untracked-files=all", *source_pathspec, binary=True)
    dirty = bool(status_bytes)
    if not dirty:
        return {"commit": commit, "dirty": False, "diff_sha256": None}
    digest = hashlib.sha256()
    digest.update(status_bytes)
    digest.update(_run_git("diff", "--binary", "HEAD", *source_pathspec, binary=True))
    for entry in status_bytes.split(b"\0"):
        if entry.startswith(b"?? "):
            relative = entry[3:].decode("utf-8", errors="surrogateescape")
            path = REPO_ROOT / relative
            if path.is_file():
                digest.update(entry[3:])
                digest.update(path.read_bytes())
    return {"commit": commit, "dirty": True, "diff_sha256": digest.hexdigest()}


def trial_key(plan: dict[str, Any]) -> str:
    return "__".join(
        [plan["configuration"]["id"], plan["scene"]["id"], f"s{plan['diffusion_seed']}", f"g{plan['goal_seed']}", plan["stabilizer"]["id"]]
    )


def expand_trial_plans(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    plans = []
    for configuration in protocol["configurations"]:
        for scene in protocol["scenes"]:
            for diffusion_seed in protocol["randomization"]["diffusion_seeds"]:
                for goal_seed in protocol["randomization"]["goal_seeds"]:
                    for stabilizer in protocol["stabilizer_strata"]:
                        plan = {
                            "configuration": configuration,
                            "scene": scene,
                            "diffusion_seed": int(diffusion_seed),
                            "goal_seed": int(goal_seed),
                            "stabilizer": stabilizer,
                        }
                        plan["trial_key"] = trial_key(plan)
                        plans.append(plan)
    keys = [plan["trial_key"] for plan in plans]
    if len(keys) != len(set(keys)):
        raise ValueError("Protocol expansion produced duplicate trial keys.")
    expected = int(protocol["planned_matrix"]["planned_trial_count_before_gate"])
    if len(plans) != expected:
        raise ValueError(f"Protocol expansion produced {len(plans)} trials, expected {expected}.")
    return plans


def filter_plans(plans: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = []
    for plan in plans:
        if args.config and plan["configuration"]["id"] not in args.config:
            continue
        if args.scene and plan["scene"]["id"] not in args.scene:
            continue
        if args.seed and plan["diffusion_seed"] not in args.seed:
            continue
        if args.stabilizer and plan["stabilizer"]["id"] not in args.stabilizer:
            continue
        selected.append(plan)
    return selected


def _input_record(protocol: dict[str, Any], protocol_path: Path, protocol_sha256: str, scene: dict[str, Any]) -> dict[str, Any]:
    immutable = protocol["immutable_inputs"]
    return {
        "protocol_path": str(protocol_path.relative_to(REPO_ROOT)), "protocol_sha256": protocol_sha256,
        "checkpoint_path": immutable["policy_checkpoint"]["path"], "checkpoint_sha256": immutable["policy_checkpoint"]["sha256"],
        "policy_config_path": immutable["policy_config"]["path"], "policy_config_sha256": immutable["policy_config"]["sha256"],
        "topomap_path": scene["topomap_path"], "topomap_tree_sha256": scene["topomap_tree_sha256"],
        "scene_definition_path": immutable["scene_definition"]["path"], "scene_definition_sha256": immutable["scene_definition"]["sha256"],
        "mjcf_path": immutable["lite3_mjcf"]["path"], "mjcf_sha256": immutable["lite3_mjcf"]["sha256"],
        "locomotion_policy_path": immutable["locomotion_policy"]["path"], "locomotion_policy_sha256": immutable["locomotion_policy"]["sha256"],
    }


def verify_immutable_files(record: dict[str, Any], expected_commit: str | None = None) -> None:
    if expected_commit is not None:
        current_commit = _run_git("rev-parse", "HEAD").strip()
        if current_commit != expected_commit:
            raise ValueError(f"HEAD mismatch: {current_commit} != {expected_commit}")
    pairs = (
        ("checkpoint_path", "checkpoint_sha256"), ("policy_config_path", "policy_config_sha256"),
        ("scene_definition_path", "scene_definition_sha256"), ("mjcf_path", "mjcf_sha256"),
        ("locomotion_policy_path", "locomotion_policy_sha256"),
    )
    for path_field, hash_field in pairs:
        path = REPO_ROOT / record[path_field]
        if not path.is_file():
            raise FileNotFoundError(f"Missing immutable input: {path}")
        actual = sha256_file(path)
        if actual != record[hash_field]:
            raise ValueError(f"Hash mismatch for {path_field}: {actual} != {record[hash_field]}")
    topomap = REPO_ROOT / record["topomap_path"]
    if not topomap.is_dir() or not any(topomap.glob("*.png")):
        raise FileNotFoundError(f"Missing topomap images: {topomap}")
    topomap_status = _run_git("status", "--porcelain=v1", "--", record["topomap_path"])
    if topomap_status.strip():
        raise ValueError(f"Topomap differs from the pinned commit: {record['topomap_path']}")


def _base_record(plan: dict[str, Any], protocol: dict[str, Any], protocol_path: Path, protocol_sha256: str, run_id: str, attempt: int, trial_path: Path, runtime_dir: Path) -> dict[str, Any]:
    config = plan["configuration"]
    scene = plan["scene"]
    nav = protocol["navigation_contract"]
    return {
        "schema_version": "0.1.0", "record_kind": "trial", "run_id": run_id,
        "trial_id": f"{plan['trial_key']}__a{attempt:02d}", "trial_key": plan["trial_key"], "attempt": attempt,
        "created_at_utc": utc_now(), "completed_at_utc": utc_now(), "status": "crashed",
        "git": git_metadata(), "inputs": _input_record(protocol, protocol_path, protocol_sha256, scene),
        "environment": {"device": protocol["runtime"]["device"], "gpu": "unknown", "precision": protocol["runtime"]["precision"], "python_version": sys.version.split()[0], "torch_version": "unknown", "simulator_version": protocol["runtime"]["simulator"].replace("mujoco-", "", 1), "renderer": protocol["runtime"]["renderer"], "onnxruntime_version": "unknown", "onnx_provider": protocol["runtime"]["onnx_provider"]},
        "episode": {"map": scene["id"], "spawn_xy_m": scene["spawn_xy_m"], "goal_xy_m": scene["goal_xy_m"], "diffusion_seed": plan["diffusion_seed"], "goal_seed": plan["goal_seed"], "stabilizer_mode": plan["stabilizer"]["id"], "success_radius_m": nav["success"]["radius_m"], "max_steps": nav["timeout"]["max_steps"], "simulated_timeout_s": nav["timeout"]["simulated_navigation_seconds"]},
        "policy": {"encoder": "nomad_vint/efficientnet-b0", "scheduler": config["scheduler"], "diffusion_steps": config["diffusion_steps"], "cfg_weight": config["cfg_weight"], "tts_enabled": config["tts_enabled"], "tts_budget": config.get("tts_budget"), "tts_topk": config.get("tts_topk"), "tts_verifier": config.get("tts_verifier"), "candidate_k": config["candidate_k"], "selection": config["selection"], "image_resize_mode": nav["image_resize_mode"], "image_size_px": nav["image_size_px"], "context_frames": nav["context_frames"], "action_horizon": nav["action_horizon"], "waypoint_index": nav["waypoint_index"]},
        "bridge": nav["controller"].copy(),
        "outcome": {"success": False, "final_distance_m": None, "path_length_m": 0.0, "shortest_path_m": 0.0, "spl": 0.0, "collision": False, "fall": False, "stuck": False, "timeout": False, "intervention": False, "recovery_count": 0, "exception": "trial did not execute", "exit_code": None},
        "trajectory": [], "timing": {"warmup_calls": protocol["latency"]["warmup_calls"], "cuda_synchronized": True, "sampler_per_call_ms": [], "full_loop_per_call_ms": []},
        "artifacts": {"trial_json": str(trial_path.relative_to(REPO_ROOT)), "runtime_dir": str(runtime_dir.relative_to(REPO_ROOT)), "timing_csv": None},
    }


def _path_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(points, dtype=float), axis=0), axis=1).sum())


def _read_full_loop_timing(path: Path) -> list[float]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [float(row["total_ms"]) for row in csv.DictReader(handle) if row.get("label") == "navigate" and row.get("total_ms")]


def execute_lite3(record: dict[str, Any], protocol: dict[str, Any], plan: dict[str, Any], runtime_dir: Path) -> dict[str, Any]:
    import torch
    import mujoco
    import onnxruntime
    from simulation.nomad_mujoco_lite3_state_machine import build_parser
    from simulation.lite3_system.session import NavigationSession
    from simulation.lite3_system.system import Lite3System

    seed = int(plan["diffusion_seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    config = plan["configuration"]
    cli = ["--mode", "navigate", "--map", plan["scene"]["id"], "--no-gui", "--camera", "off", "--max-steps", str(protocol["navigation_contract"]["timeout"]["max_steps"]), "--seed", str(seed), "--goal-seed", str(plan["goal_seed"]), "--topomap-dir", plan["scene"]["topomap_path"], "--scheduler", config["scheduler"], "--ddim-steps", str(config["diffusion_steps"]), "--cfg-weight", str(config["cfg_weight"]), "--policy-config", protocol["immutable_inputs"]["policy_config"]["path"], "--policy-checkpoint", protocol["immutable_inputs"]["policy_checkpoint"]["path"], "--policy-device", protocol["runtime"]["device"], "--image-resize-mode", protocol["navigation_contract"]["image_resize_mode"], "--policy-action-scale-m", str(protocol["navigation_contract"].get("policy_action_scale_m", 1.0)), "--profile-timing", "--profile-interval", "10"]
    if config["tts_enabled"]:
        cli += ["--tts", "--tts-budget", str(config["tts_budget"]), "--tts-topk", str(config["tts_topk"]), "--tts-verifier", config["tts_verifier"]]
    if not plan["stabilizer"]["mujoco_route_stabilizer"]:
        cli.append("--no-mujoco-route-stabilizer")
    args = build_parser().parse_args(cli)
    args.suppress_legacy_results = True
    system = Lite3System(args, result_prefix=record["trial_id"], session=NavigationSession(record["trial_id"], run_dir=runtime_dir))
    exit_code = system.run()
    points = [[float(point[0]), float(point[1])] for point in system.trajectory]
    goal = np.asarray(plan["scene"]["goal_xy_m"], dtype=float)
    final_distance = float(np.linalg.norm(np.asarray(points[-1]) - goal)) if points else None
    path_length = _path_length(points)
    reference = np.asarray(system.legacy.build_scene_reference_path(system.legacy.SCENE_MAPS[plan["scene"]["id"]], num_points=600), dtype=float)
    shortest = float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum())
    success = bool(system.goal_reached and not system.fall_detected)
    record["environment"].update({"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch_version": torch.__version__, "simulator_version": mujoco.__version__, "onnxruntime_version": onnxruntime.__version__})
    record["status"] = "completed" if success else "failed"
    record["outcome"] = {"success": success, "final_distance_m": final_distance, "path_length_m": path_length, "shortest_path_m": shortest, "spl": float(success * shortest / max(shortest, path_length, 1e-9)), "collision": bool(system.collision_detected), "fall": bool(system.fall_detected), "stuck": bool(system.stuck_detected), "timeout": bool(not success and system.tick >= args.max_steps), "intervention": bool(system.session.estop_requested or system.session.exit_requested), "recovery_count": int(system.recovery_count_total), "exception": None, "exit_code": int(exit_code)}
    record["trajectory"] = [{"tick": index, "x_m": point[0], "y_m": point[1]} for index, point in enumerate(points)]
    sampler = list(system.high_level.inference.sampler_timings_ms)
    record["timing"]["sampler_per_call_ms"] = sampler
    record["timing"]["full_loop_per_call_ms"] = _read_full_loop_timing(system.timing_csv_path)
    if system.timing_csv_path.is_file():
        record["artifacts"]["timing_csv"] = str(system.timing_csv_path.relative_to(REPO_ROOT))
    return record


def next_attempt(trials_dir: Path, key: str) -> int:
    attempts = []
    for path in trials_dir.glob(f"{key}__a*.json"):
        try:
            attempts.append(int(path.stem.rsplit("__a", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(attempts, default=0) + 1


def has_complete_attempt(trials_dir: Path, key: str) -> bool:
    for path in trials_dir.glob(f"{key}__a*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") in {"completed", "failed"}:
            return True
    return False


def run_trial_attempt(plan: dict[str, Any], protocol: dict[str, Any], protocol_path: Path, protocol_sha256: str, run_id: str, run_dir: Path, executor: Callable = execute_lite3) -> Path:
    trials_dir = run_dir / "trials"
    attempt = next_attempt(trials_dir, plan["trial_key"])
    trial_path = trials_dir / f"{plan['trial_key']}__a{attempt:02d}.json"
    runtime_dir = run_dir / "runtime" / f"{plan['trial_key']}__a{attempt:02d}"
    record = _base_record(plan, protocol, protocol_path, protocol_sha256, run_id, attempt, trial_path, runtime_dir)
    try:
        verify_immutable_files(record["inputs"], expected_commit=protocol["immutable_inputs"]["git_commit"])
    except (FileNotFoundError, ValueError) as exc:
        record["status"] = "infrastructure_invalid"
        record["outcome"]["exception"] = f"{type(exc).__name__}: {exc}"
        record["outcome"]["exit_code"] = 2
    else:
        try:
            record = executor(record, protocol, plan, runtime_dir)
        except Exception as exc:  # A runtime exception must remain as a trial record.
            record["status"] = "crashed"
            record["outcome"]["exception"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            record["outcome"]["exit_code"] = 1
    record["completed_at_utc"] = utc_now()
    atomic_write_json(trial_path, record)
    return trial_path


def build_manifest(run_id: str, protocol_sha256: str, plans: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    candidates_by_key: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    selected_keys = {plan["trial_key"] for plan in plans}
    for path in sorted((run_dir / "trials").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("trial_key") in selected_keys and payload.get("status") in VALID_FINAL_STATUSES:
            candidates_by_key.setdefault(payload["trial_key"], []).append((path, payload))
    primary_by_key: dict[str, tuple[Path, dict[str, Any]]] = {}
    for key, candidates in candidates_by_key.items():
        scientific = [item for item in candidates if item[1]["status"] in {"completed", "failed"}]
        primary_by_key[key] = (scientific or candidates)[0]
    counts = {status: 0 for status in VALID_FINAL_STATUSES}
    for _, payload in primary_by_key.values():
        counts[payload["status"]] += 1
    missing = sorted({plan["trial_key"] for plan in plans} - set(primary_by_key))
    records = [str(primary_by_key[key][0].relative_to(REPO_ROOT)) for key in sorted(primary_by_key)]
    return {"schema_version": "0.1.0", "record_kind": "batch_manifest", "run_id": run_id, "created_at_utc": utc_now(), "protocol_sha256": protocol_sha256, "planned_trial_count": len(plans), "completed": counts["completed"], "failed": counts["failed"], "crashed": counts["crashed"], "infrastructure_invalid": counts["infrastructure_invalid"], "missing_trial_keys": missing, "trial_records": records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", action="append")
    parser.add_argument("--scene", action="append", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--stabilizer", action="append", choices=["policy_only_off", "system_stabilizer_on"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.expanduser().resolve()
    protocol, protocol_sha256 = load_protocol(protocol_path)
    plans = filter_plans(expand_trial_plans(protocol), args)
    if not plans:
        raise SystemExit("No trials match the requested filters.")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.results_root.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "selected_plan.json", {"run_id": run_id, "protocol_sha256": protocol_sha256, "trial_keys": [plan["trial_key"] for plan in plans]})
    if not args.dry_run:
        for plan in plans:
            if has_complete_attempt(run_dir / "trials", plan["trial_key"]):
                continue
            run_trial_attempt(plan, protocol, protocol_path, protocol_sha256, run_id, run_dir)
    manifest = build_manifest(run_id, protocol_sha256, plans, run_dir)
    atomic_write_json(run_dir / "batch_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "selected_trials": len(plans), "dry_run": args.dry_run, "missing": len(manifest["missing_trial_keys"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
