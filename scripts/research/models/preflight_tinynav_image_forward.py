#!/usr/bin/env python3
"""GPU inference-only readiness check for TinyNavBrain image-to-waypoint APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
PRETRAINING = ROOT / "scripts" / "research" / "pretraining"
sys.path.insert(0, str(PRETRAINING))

from go_stanford_adapter import (  # noqa: E402
    GoStanfordCanonicalDataset,
    make_go_stanford_dataloader,
)
from tinynavbrain_image_policy import TinyNavBrainImagePolicy  # noqa: E402
from tinynavbrain_scaffold import TinyNavConfig  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_context(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def timed_forward(device: torch.device, function: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = function()
    torch.cuda.synchronize(device)
    return output, (time.perf_counter() - started) * 1000.0


def run_preflight(
    *,
    repo_root: Path,
    dataset_root: Path,
    manifest_path: Path,
    data_audit_path: Path,
    batch_readiness_path: Path,
    smoke_config_path: Path,
    model_contract_path: Path,
    device_index: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device_index} does not exist")
    device = torch.device(f"cuda:{device_index}")
    smoke = yaml.safe_load(smoke_config_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "planned_no_training_executed" or smoke.get("actual_results") is not None:
        raise ValueError("smoke config must remain unexecuted during forward preflight")
    audit = json.loads(data_audit_path.read_text(encoding="utf-8"))
    batch_readiness = json.loads(batch_readiness_path.read_text(encoding="utf-8"))
    if not audit.get("passed") or not batch_readiness["gates"]["canonical_batches_passed"]:
        raise ValueError("data and canonical batch readiness must pass first")

    config = TinyNavConfig()
    dataset = GoStanfordCanonicalDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
        observation_frames=int(smoke["observation_frames"]),
        action_history_steps=int(smoke["action_history_steps"]),
        action_horizon=int(smoke["action_horizon"]),
        image_size=(96, 96),
        data_fraction=float(smoke["data_fraction"]),
        subset_seed=int(smoke["seed"]),
    )
    loader = make_go_stanford_dataloader(
        dataset,
        batch_size=int(smoke["batch_size_candidate"]),
        shuffle=False,
        num_workers=0,
    )
    cpu_batch = next(iter(loader))
    batch = move_batch(cpu_batch, device)

    torch.manual_seed(int(smoke["seed"]))
    torch.cuda.manual_seed_all(int(smoke["seed"]))
    model = TinyNavBrainImagePolicy(config, encoder_weights=None).to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        # One unreported warmup removes first-kernel initialization from gate timings.
        model(batch, head="h0_deterministic")
        h0, h0_ms = timed_forward(
            device, lambda: model(batch, head="h0_deterministic")
        )
        x_t = torch.zeros(
            batch["obs_images"].shape[0], config.action_horizon, config.waypoint_dim,
            device=device,
        )
        t = torch.full((batch["obs_images"].shape[0], 1), 0.5, device=device)
        h1_velocity, h1_velocity_ms = timed_forward(
            device, lambda: model(batch, head="h1_velocity", x_t=x_t, t=t)
        )
        h1_sample, h1_sample_ms = timed_forward(
            device,
            lambda: model(
                batch,
                head="h1_rectified_flow",
                nfe=2,
                candidate_count=1,
                seed=int(smoke["seed"]),
            ),
        )
        h1_repeat = model(
            batch,
            head="h1_rectified_flow",
            nfe=2,
            candidate_count=1,
            seed=int(smoke["seed"]),
        )
        torch.cuda.synchronize(device)

    outputs = {
        "context": h0["context"],
        "h0_waypoints": h0["waypoints_m"],
        "progress": h0["progress_m"],
        "h1_velocity": h1_velocity["velocity"],
        "h1_candidates": h1_sample["waypoint_candidates_m"],
    }
    non_finite = {
        name: int((~torch.isfinite(value)).sum().item()) for name, value in outputs.items()
    }
    if any(non_finite.values()):
        raise ValueError(f"non-finite forward outputs: {non_finite}")
    reproducible = torch.equal(
        h1_sample["waypoint_candidates_m"], h1_repeat["waypoint_candidates_m"]
    )
    if not reproducible:
        raise ValueError("fixed-seed H1 sampling is not reproducible")
    gradients_created = any(parameter.grad is not None for parameter in model.parameters())
    if gradients_created:
        raise ValueError("parameter gradients unexpectedly exist after inference-only gate")
    properties = torch.cuda.get_device_properties(device)
    commit, dirty = git_context(repo_root)

    return {
        "schema_version": "0.1.0",
        "gate_id": "tinynavbrain-image-forward-only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_boundary": {
            "encoder_weights": "random_initialization_weights_none",
            "weights_downloaded": False,
            "model_instantiated": True,
            "model_forward_performed": True,
            "inference_mode": True,
            "backward_performed": False,
            "optimizer_created": False,
            "training_performed": False,
            "checkpoint_written": False,
        },
        "provenance": {
            "git_commit": commit,
            "tracked_worktree_dirty": dirty,
            "manifest_sha256": audit["provenance"]["manifest_sha256"],
            "split_sha256": audit["provenance"]["split_sha256"],
            "dataset_tree_sha256": audit["provenance"]["canonical_dataset_tree_sha256"],
            "model_contract_sha256": canonical_text_sha256(model_contract_path),
            "smoke_config_sha256": canonical_text_sha256(smoke_config_path),
            "adapter_sha256": file_sha256(PRETRAINING / "go_stanford_adapter.py"),
            "scaffold_sha256": file_sha256(Path(__file__).with_name("tinynavbrain_scaffold.py")),
            "image_policy_sha256": file_sha256(Path(__file__).with_name("tinynavbrain_image_policy.py")),
            "gate_script_sha256": file_sha256(Path(__file__)),
        },
        "hardware": {
            "device_index": device_index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "model": {
            "shared_image_encoder": "torchvision_efficientnet_b0",
            "shared_encoder_instance_count": 1,
            "trainable_parameters": model.trainable_parameter_count(),
            "encoder_trainable_parameters": model.encoder_parameter_count(),
            "max_trainable_parameters": config.max_trainable_parameters,
            "parameter_budget_passed": model.trainable_parameter_count()
            < config.max_trainable_parameters,
        },
        "batch": {
            "batch_size": int(batch["obs_images"].shape[0]),
            "obs_shape": list(batch["obs_images"].shape),
            "goal_shape": list(batch["goal_image"].shape),
            "history_shape": list(batch["action_history"].shape),
            "target_shape": list(batch["target_waypoints"].shape),
            "trajectory_ids": list(batch["trajectory_id"]),
        },
        "outputs": {
            "shapes": {name: list(value.shape) for name, value in outputs.items()},
            "non_finite_counts": non_finite,
            "h1_fixed_seed_reproducible": reproducible,
        },
        "forward_gate_measurements": {
            "h0_forward_ms_batch": h0_ms,
            "h1_velocity_forward_ms_batch": h1_velocity_ms,
            "h1_nfe2_sample_forward_ms_batch": h1_sample_ms,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "scope": "inference-only forward gate; not training throughput or deployment latency",
        },
        "gates": {
            "synthetic_cpu_forward_tests_passed": True,
            "real_batch_h0_passed": True,
            "real_batch_h1_velocity_passed": True,
            "real_batch_h1_sample_passed": True,
            "finite_outputs": True,
            "fixed_seed_reproducible": reproducible,
            "single_shared_encoder": True,
            "parameter_budget_passed": model.trainable_parameter_count()
            < config.max_trainable_parameters,
            "no_gradients_created": not gradients_created,
        },
        "ready_for_train_loop_implementation": True,
        "ready_for_b0_training": False,
        "remaining_before_b0": [
            "implement and test loss/train-step/checkpoint exact-resume path",
            "freeze current dirty worktree in an authorized Git commit",
            "obtain separate backward/optimizer/training authorization",
        ],
        "claim_boundary": (
            "Random-initialized inference-only shape/finite/timing checks establish interface "
            "readiness only. They are not training feasibility, model quality, or method evidence."
        ),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    model = report["model"]
    measurements = report["forward_gate_measurements"]
    lines = [
        "# TinyNavBrain IMAGE-FORWARD Gate",
        "",
        f"- GPU: {report['hardware']['name']} (cuda:{report['hardware']['device_index']})",
        f"- Encoder: {model['shared_image_encoder']} / shared instances: {model['shared_encoder_instance_count']}",
        f"- Trainable parameters: {model['trainable_parameters']} / budget {model['max_trainable_parameters']}",
        f"- H0 batch forward: {measurements['h0_forward_ms_batch']:.3f} ms",
        f"- H1 velocity batch forward: {measurements['h1_velocity_forward_ms_batch']:.3f} ms",
        f"- H1 NFE=2 sample batch forward: {measurements['h1_nfe2_sample_forward_ms_batch']:.3f} ms",
        f"- Peak CUDA memory: {measurements['peak_cuda_memory_bytes'] / (1024 ** 2):.2f} MiB",
        f"- Ready for train-loop implementation: {report['ready_for_train_loop_implementation']}",
        f"- Ready for B0 training: {report['ready_for_b0_training']}",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in report["gates"].items())
    lines.extend(["", "## Remaining before B0", ""])
    lines.extend(f"- {item}" for item in report["remaining_before_b0"])
    lines.extend(["", "## Evidence boundary", "", report["claim_boundary"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--batch-readiness", type=Path, required=True)
    parser.add_argument("--smoke-config", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_preflight(
            repo_root=args.repo_root,
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            data_audit_path=args.data_audit,
            batch_readiness_path=args.batch_readiness,
            smoke_config_path=args.smoke_config,
            model_contract_path=args.model_contract,
            device_index=args.device_index,
        )
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, report)
    print(
        f"IMAGE-FORWARD gate passed on {report['hardware']['name']}; "
        f"parameters={report['model']['trainable_parameters']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
