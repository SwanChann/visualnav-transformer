#!/usr/bin/env python3
"""Run exactly two authorized TinyNavBrain optimizer steps and verify resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "scripts" / "research" / "models"
sys.path.insert(0, str(MODELS))

from go_stanford_adapter import (  # noqa: E402
    GoStanfordCanonicalDataset,
    collate_go_stanford_canonical,
)
from train_step_runtime import (  # noqa: E402
    DeterministicBatchCursor,
    load_exact_resume_checkpoint,
    optimizer_train_step,
    save_exact_resume_checkpoint,
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


def git_context(repo_root: Path) -> tuple[str, bool, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    tracked_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=repo_root, check=True, capture_output=True, text=True,
    ).stdout.strip())
    any_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return commit, tracked_dirty, any_dirty


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def synthetic_batch(device: torch.device, config: TinyNavConfig, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    batch_size = 2
    return {
        "obs_images": torch.randn(
            batch_size, config.observation_frames, 3, 96, 96,
            generator=generator, device=device,
        ),
        "goal_image": torch.randn(batch_size, 3, 96, 96, generator=generator, device=device),
        "goal_mask": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "action_history": 0.05 * torch.randn(
            batch_size, config.action_history_length, 3,
            generator=generator, device=device,
        ),
        "action_history_mask": torch.ones(
            batch_size, config.action_history_length, dtype=torch.bool, device=device,
        ),
        "dt_s": torch.full((batch_size, 1), 0.25, device=device),
        "waypoint_spacing_m": torch.full((batch_size, 1), 0.12, device=device),
        "target_waypoints": 0.25 * torch.randn(
            batch_size, config.action_horizon, config.waypoint_dim,
            generator=generator, device=device,
        ),
        "target_mask": torch.ones(
            batch_size, config.action_horizon, dtype=torch.bool, device=device,
        ),
        "dataset_id": ["synthetic_contract"] * batch_size,
        "trajectory_id": [f"synthetic-{index}" for index in range(batch_size)],
    }


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def batch_identity(batch: Mapping[str, Any]) -> list[str]:
    return [
        f"{trajectory}:{int(index)}"
        for trajectory, index in zip(batch["trajectory_id"], batch["current_index"])
    ]


def timed_step(device: torch.device, function: Any) -> tuple[Any, float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize(device)
    return result, (time.perf_counter() - started) * 1000.0


def run_gate(
    *,
    repo_root: Path,
    dataset_root: Path,
    manifest_path: Path,
    data_audit_path: Path,
    batch_readiness_path: Path,
    image_readiness_path: Path,
    smoke_config_path: Path,
    model_contract_path: Path,
    checkpoint_path: Path,
    device_index: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or not 0 <= device_index < torch.cuda.device_count():
        raise ValueError(f"CUDA device {device_index} is unavailable")
    device = torch.device(f"cuda:{device_index}")
    smoke = yaml.safe_load(smoke_config_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "planned_no_training_executed" or smoke.get("actual_results") is not None:
        raise ValueError("frozen 200-step smoke config must remain unexecuted")
    if int(smoke.get("optimizer_steps", -1)) != 200:
        raise ValueError("frozen B0 config no longer specifies 200 steps")
    audit = json.loads(data_audit_path.read_text(encoding="utf-8"))
    batch_readiness = json.loads(batch_readiness_path.read_text(encoding="utf-8"))
    image_readiness = json.loads(image_readiness_path.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise ValueError("Go Stanford data audit has not passed")
    if not batch_readiness["gates"]["canonical_batches_passed"]:
        raise ValueError("canonical batch readiness has not passed")
    if not all(image_readiness["gates"].values()):
        raise ValueError("IMAGE-FORWARD gate has not passed")

    seed = int(smoke["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
        subset_seed=seed,
    )
    cursor = DeterministicBatchCursor(
        len(dataset), int(smoke["batch_size_candidate"]), seed
    )
    model = TinyNavBrainImagePolicy(config, encoder_weights=None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    initial_model_hash = tensor_state_sha256(model.state_dict())
    torch.cuda.reset_peak_memory_stats(device)

    synthetic = synthetic_batch(device, config, seed + 100)
    h0, h0_ms = timed_step(
        device,
        lambda: optimizer_train_step(
            model, optimizer, synthetic, head="h0_deterministic", max_gradient_norm=10.0
        ),
    )
    after_h0_hash = tensor_state_sha256(model.state_dict())
    if after_h0_hash == initial_model_hash:
        raise ValueError("synthetic H0 optimizer step did not change model state")

    real_indices = cursor.next_indices()
    real_cpu = collate_go_stanford_canonical([dataset[index] for index in real_indices])
    real_identity = batch_identity(real_cpu)
    real_batch = move_batch(real_cpu, device)
    h1, h1_ms = timed_step(
        device,
        lambda: optimizer_train_step(
            model, optimizer, real_batch, head="h1_rectified_flow",
            max_gradient_norm=10.0,
        ),
    )
    after_h1_hash = tensor_state_sha256(model.state_dict())
    if after_h1_hash == after_h0_hash:
        raise ValueError("real H1 optimizer step did not change model state")
    optimizer_steps_executed = 2
    if optimizer_steps_executed != 2:
        raise ValueError("TRAIN-STEP gate must execute exactly two optimizer steps")
    train_step_peak_memory = torch.cuda.max_memory_allocated(device)

    commit, tracked_dirty, any_dirty = git_context(repo_root)
    metadata = {
        "schema_version": "0.1.0",
        "run_id": "tinynavbrain-train-step-gate-v0.1",
        "phase": "smoke",
        "head": "h1_rectified_flow",
        "seed": seed,
        "optimizer_step": optimizer_steps_executed,
        "git_commit": commit,
        "manifest_sha256": audit["provenance"]["manifest_sha256"],
        "split_sha256": audit["provenance"]["split_sha256"],
        "model_contract_sha256": canonical_text_sha256(model_contract_path),
        "config_sha256": canonical_text_sha256(smoke_config_path),
        "rng_state_present": True,
        "sampler_state_present": True,
    }
    save_exact_resume_checkpoint(
        checkpoint_path,
        metadata=metadata,
        model=model,
        optimizer=optimizer,
        sampler_state=cursor.state_dict(),
        extra_state={
            "gate_not_b0": True,
            "optimizer_steps_executed": optimizer_steps_executed,
            "heads": ["h0_deterministic", "h1_rectified_flow"],
        },
    )
    checkpoint_sha256 = file_sha256(checkpoint_path)

    expected_next_indices = cursor.next_indices()
    expected_next_batch = collate_go_stanford_canonical(
        [dataset[index] for index in expected_next_indices]
    )
    expected_next_identity = batch_identity(expected_next_batch)
    expected_rng = {
        "python": random.random(),
        "numpy": float(np.random.rand()),
        "torch_cpu": torch.rand(8),
        "torch_cuda": torch.rand(8, device=device),
    }

    resumed_model = TinyNavBrainImagePolicy(config, encoder_weights=None).to(device)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-4)
    resumed_cursor = DeterministicBatchCursor(
        len(dataset), int(smoke["batch_size_candidate"]), seed
    )
    loaded = load_exact_resume_checkpoint(
        checkpoint_path,
        expected_metadata=metadata,
        model=resumed_model,
        optimizer=resumed_optimizer,
        sampler=resumed_cursor,
        map_location=device,
    )
    actual_next_indices = resumed_cursor.next_indices()
    actual_next_batch = collate_go_stanford_canonical(
        [dataset[index] for index in actual_next_indices]
    )
    actual_next_identity = batch_identity(actual_next_batch)
    actual_rng = {
        "python": random.random(),
        "numpy": float(np.random.rand()),
        "torch_cpu": torch.rand(8),
        "torch_cuda": torch.rand(8, device=device),
    }
    model_exact = nested_equal(model.state_dict(), resumed_model.state_dict())
    optimizer_exact = nested_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
    sampler_exact = (
        expected_next_indices == actual_next_indices
        and expected_next_identity == actual_next_identity
    )
    rng_exact = (
        expected_rng["python"] == actual_rng["python"]
        and expected_rng["numpy"] == actual_rng["numpy"]
        and torch.equal(expected_rng["torch_cpu"], actual_rng["torch_cpu"])
        and torch.equal(expected_rng["torch_cuda"], actual_rng["torch_cuda"])
    )
    metadata_exact = loaded["metadata"] == metadata
    gates = {
        "exactly_two_optimizer_steps": optimizer_steps_executed == 2,
        "synthetic_h0_step_passed": h0.optimizer_step_performed,
        "real_h1_step_passed": h1.optimizer_step_performed,
        "finite_losses": bool(np.isfinite([h0.loss, h1.loss]).all()),
        "finite_positive_gradients": (
            h0.finite_gradients and h1.finite_gradients
            and h0.gradient_l2_norm > 0 and h1.gradient_l2_norm > 0
        ),
        "both_steps_changed_model_state": (
            initial_model_hash != after_h0_hash != after_h1_hash
        ),
        "checkpoint_metadata_valid_and_exact": metadata_exact,
        "model_state_exact_after_reload": model_exact,
        "optimizer_state_exact_after_reload": optimizer_exact,
        "rng_state_exact_after_reload": rng_exact,
        "next_batch_exact_after_reload": sampler_exact,
    }
    if not all(gates.values()):
        raise ValueError(f"TRAIN-STEP gate failed: {gates}")
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": "0.1.0",
        "gate_id": "tinynavbrain-train-step-exact-resume",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_boundary": {
            "optimizer_steps_executed": optimizer_steps_executed,
            "backward_calls_executed": 2,
            "synthetic_steps": 1,
            "real_batch_steps": 1,
            "b0_200_step_run_executed": False,
            "evaluation_executed": False,
            "pretrained_weights_downloaded": False,
            "checkpoint_written": True,
        },
        "provenance": {
            "git_commit": commit,
            "tracked_worktree_dirty": tracked_dirty,
            "worktree_dirty_including_preserved_untracked": any_dirty,
            "manifest_sha256": metadata["manifest_sha256"],
            "split_sha256": metadata["split_sha256"],
            "dataset_tree_sha256": audit["provenance"]["canonical_dataset_tree_sha256"],
            "model_contract_sha256": metadata["model_contract_sha256"],
            "smoke_config_sha256": metadata["config_sha256"],
            "adapter_sha256": file_sha256(Path(__file__).with_name("go_stanford_adapter.py")),
            "runtime_sha256": file_sha256(Path(__file__).with_name("train_step_runtime.py")),
            "image_policy_sha256": file_sha256(MODELS / "tinynavbrain_image_policy.py"),
            "gate_script_sha256": file_sha256(Path(__file__)),
        },
        "hardware": {
            "device_index": device_index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4, "amp": False},
        "steps": [
            {**h0.__dict__, "scope": "synthetic", "elapsed_ms": h0_ms},
            {
                **h1.__dict__, "scope": "go_stanford_real_canonical_batch",
                "elapsed_ms": h1_ms, "batch_identity": real_identity,
            },
        ],
        "state_hashes": {
            "initial_model": initial_model_hash,
            "after_h0": after_h0_hash,
            "after_h1": after_h1_hash,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_path.stat().st_size,
            "metadata": metadata,
            "expected_next_batch_identity": expected_next_identity,
            "resumed_next_batch_identity": actual_next_identity,
        },
        "measurements": {
            "peak_cuda_memory_bytes_during_two_steps": train_step_peak_memory,
            "scope": "two-step integration gate only; not B0 throughput or training feasibility",
        },
        "gates": gates,
        "ready_for_b0_execution": False,
        "remaining_before_b0": [
            "review and freeze the TRAIN-STEP implementation/report in Git",
            "obtain separate explicit authorization for the frozen 200-step B0 run",
        ],
        "claim_boundary": (
            "Two optimizer steps validate loss/backward/optimizer/checkpoint plumbing only. "
            "They are not a 200-step smoke, convergence evidence, model quality, or a method result."
        ),
    }


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    steps = report["steps"]
    checkpoint = report["checkpoint"]
    lines = [
        "# TinyNavBrain TRAIN-STEP Gate",
        "",
        f"- GPU: {report['hardware']['name']} (cuda:{report['hardware']['device_index']})",
        f"- Executed optimizer/backward steps: {report['execution_boundary']['optimizer_steps_executed']}",
        f"- Synthetic H0 loss: {steps[0]['loss']:.8f} / step: {steps[0]['elapsed_ms']:.3f} ms",
        f"- Real H1 loss: {steps[1]['loss']:.8f} / step: {steps[1]['elapsed_ms']:.3f} ms",
        f"- Peak CUDA memory: {report['measurements']['peak_cuda_memory_bytes_during_two_steps'] / (1024 ** 2):.2f} MiB",
        f"- Checkpoint: `{checkpoint['path']}` ({checkpoint['size_bytes'] / (1024 ** 2):.2f} MiB)",
        f"- Checkpoint SHA-256: `{checkpoint['sha256']}`",
        f"- Ready for B0 execution: {report['ready_for_b0_execution']}",
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
    parser.add_argument("--image-readiness", type=Path, required=True)
    parser.add_argument("--smoke-config", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_gate(
            repo_root=args.repo_root,
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            data_audit_path=args.data_audit,
            batch_readiness_path=args.batch_readiness,
            image_readiness_path=args.image_readiness,
            smoke_config_path=args.smoke_config,
            model_contract_path=args.model_contract,
            checkpoint_path=args.checkpoint,
            device_index=args.device_index,
        )
    except (OSError, ValueError, RuntimeError, KeyError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, report)
    print(
        f"TRAIN-STEP gate passed with {report['execution_boundary']['optimizer_steps_executed']} "
        f"optimizer steps; checkpoint={report['checkpoint']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
