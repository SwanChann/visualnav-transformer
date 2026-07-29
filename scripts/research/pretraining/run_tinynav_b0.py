#!/usr/bin/env python3
"""Dry-run or explicitly execute the frozen TinyNavBrain B0 overlay.

The default/authorized workflow in this revision is ``--dry-run``.  The
``--execute`` branch is fail-closed behind the frozen token, branch, clean-tree,
and origin-ref guards.  Importing this module does not import torch, instantiate
a model, create an optimizer, read dataset images, or perform training.
"""

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
import yaml

from validate_b0_execution_config import (
    canonical_text_sha256,
    validate_b0_config,
)


class B0RunnerError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    return {
        "commit": git_output(repo_root, "rev-parse", "HEAD"),
        "branch": git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "origin_commit": git_output(
            repo_root, "rev-parse", "origin/agent/ubuntu-sim-handoff"
        ),
        "tracked_worktree_dirty": bool(git_output(
            repo_root, "status", "--porcelain=v1", "--untracked-files=no"
        )),
        "worktree_dirty_including_preserved_untracked": bool(git_output(
            repo_root, "status", "--porcelain=v1"
        )),
    }


def load_frozen_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    errors = validate_b0_config(payload, repo_root)
    if errors:
        raise B0RunnerError(f"B0 config validation failed: {errors}")
    return payload


def load_prerequisites(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for name, receipt in config["prerequisites"].items():
        path = repo_root / receipt["path"]
        if canonical_text_sha256(path) != receipt["sha256"]:
            raise B0RunnerError(f"prerequisite hash mismatch: {name}")
        reports[name] = json.loads(path.read_text(encoding="utf-8"))
    if not reports["data_audit"].get("passed"):
        raise B0RunnerError("DATA-PILOT audit has not passed")
    if not reports["canonical_batch_readiness"]["gates"]["canonical_batches_passed"]:
        raise B0RunnerError("canonical batch readiness has not passed")
    if not all(reports["image_forward_readiness"]["gates"].values()):
        raise B0RunnerError("IMAGE-FORWARD readiness has not passed")
    train_step = reports["train_step_readiness"]
    if not all(train_step["gates"].values()):
        raise B0RunnerError("TRAIN-STEP readiness has not passed")
    boundary = train_step["execution_boundary"]
    if boundary["total_backward_optimizer_steps_across_gate"] != 2:
        raise B0RunnerError("TRAIN-STEP receipt has an unexpected step count")
    if boundary["b0_200_step_run_executed"] is not False:
        raise B0RunnerError("TRAIN-STEP receipt incorrectly records B0 execution")
    return reports


def dry_run_report(
    config: Mapping[str, Any], config_path: Path, repo_root: Path
) -> dict[str, Any]:
    reports = load_prerequisites(config, repo_root)
    snapshot = git_snapshot(repo_root)
    return {
        "schema_version": "0.1.0",
        "gate_id": "tinynavbrain-b0-static-dry-run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_id": config["config_id"],
        "config_sha256": canonical_text_sha256(config_path),
        "runner_sha256": file_sha256(Path(__file__)),
        "git": snapshot,
        "planned_runs": config["runs"],
        "planned_budget": {
            "optimizer_steps": config["training"]["total_optimizer_steps_if_authorized"],
            "backward_calls": config["training"]["total_backward_calls_if_authorized"],
            "micro_batch_size": config["training"]["micro_batch_size"],
            "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
            "effective_batch_size": config["training"]["effective_batch_size"],
        },
        "prerequisite_receipts": {
            name: {
                "path": config["prerequisites"][name]["path"],
                "sha256": config["prerequisites"][name]["sha256"],
            }
            for name in sorted(reports)
        },
        "execution_boundary": {
            "mode": "dry-run",
            "torch_imported_by_runner": False,
            "dataset_images_read": False,
            "model_instantiated": False,
            "optimizer_created": False,
            "forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "checkpoints_written": 0,
            "training_performed": False,
        },
        "gates": {
            "config_valid": True,
            "parent_contract_hashes_match": True,
            "prerequisite_report_hashes_match": True,
            "prerequisite_semantics_pass": True,
            "h0_h1_budget_frozen": True,
            "execution_still_requires_new_authorization": True,
        },
        "ready_for_execution_authorization": True,
        "execution_authorized": False,
        "claim_boundary": (
            "Static dry-run only. No torch model, optimizer, forward/backward, checkpoint, "
            "or B0 training was executed."
        ),
    }


def validate_execute_guard(
    config: Mapping[str, Any], repo_root: Path, authorization_token: str | None
) -> dict[str, Any]:
    guard = config["execution_guard"]
    if authorization_token != guard["required_authorization_token"]:
        raise B0RunnerError("missing or incorrect explicit B0 authorization token")
    snapshot = git_snapshot(repo_root)
    if snapshot["branch"] != guard["required_branch"]:
        raise B0RunnerError("B0 branch guard failed")
    if guard["require_tracked_worktree_clean"] and snapshot["tracked_worktree_dirty"]:
        raise B0RunnerError("B0 requires a clean tracked worktree")
    if guard["require_head_equals_origin_branch"] and snapshot["commit"] != snapshot["origin_commit"]:
        raise B0RunnerError("B0 requires HEAD to equal origin/agent/ubuntu-sim-handoff")
    return snapshot


class EMAState:
    """Trainable-parameter EMA with a fully serializable exact-resume state."""

    def __init__(self, model: Any, decay: float) -> None:
        import torch

        if not 0.0 < decay < 1.0:
            raise B0RunnerError("EMA decay must be within (0,1)")
        self.decay = decay
        self.updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self.shadow or not all(isinstance(value, torch.Tensor) for value in self.shadow.values()):
            raise B0RunnerError("EMA failed to capture trainable parameters")

    def update(self, model: Any) -> None:
        import torch

        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        parameter.detach(), alpha=1.0 - self.decay
                    )
        self.updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1.0",
            "decay": self.decay,
            "updates": self.updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != "0.1.0" or float(state.get("decay", -1)) != self.decay:
            raise B0RunnerError("EMA resume contract mismatch")
        if set(state.get("shadow", {})) != set(self.shadow):
            raise B0RunnerError("EMA parameter set mismatch")
        self.updates = int(state["updates"])
        for name, value in state["shadow"].items():
            self.shadow[name].copy_(value)


def nested_digest(value: Any) -> str:
    import torch

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=lambda entry: repr(entry)):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for child in item:
                update(child)
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def rng_probe(device: Any) -> dict[str, Any]:
    import torch

    return {
        "python": random.random(),
        "numpy": float(np.random.rand()),
        "torch_cpu": torch.rand(8),
        "torch_cuda": torch.rand(8, device=device),
    }


def rng_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    import torch

    return (
        left["python"] == right["python"]
        and left["numpy"] == right["numpy"]
        and torch.equal(left["torch_cpu"], right["torch_cpu"])
        and torch.equal(left["torch_cuda"], right["torch_cuda"])
    )


def sample_identity(dataset: Any, indices: list[int]) -> list[str]:
    return [
        f"{dataset.sample_index[index].trajectory_id}:{dataset.sample_index[index].current_index}"
        for index in indices
    ]


def move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def checkpoint_metadata(
    *,
    run: Mapping[str, Any],
    step: int,
    seed: int,
    git_commit: str,
    config_sha256: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "run_id": f"{config['config_id']}-{run['id']}",
        "phase": "smoke",
        "head": run["head"],
        "seed": seed,
        "optimizer_step": step,
        "git_commit": git_commit,
        "manifest_sha256": config["data"]["manifest_sha256"],
        "split_sha256": config["data"]["split_sha256"],
        "model_contract_sha256": config["parent_contracts"]["model_contract_sha256"],
        "config_sha256": config_sha256,
        "rng_state_present": True,
        "sampler_state_present": True,
    }


def save_b0_checkpoint(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    model: Any,
    optimizer: Any,
    scaler: Any,
    ema: EMAState,
    cursor: Any,
    exposure: Mapping[str, Any],
    step_records: list[dict[str, Any]],
) -> None:
    from train_step_runtime import save_exact_resume_checkpoint

    save_exact_resume_checkpoint(
        path,
        metadata=metadata,
        model=model,
        optimizer=optimizer,
        sampler_state=cursor.state_dict(),
        extra_state={
            "grad_scaler": scaler.state_dict(),
            "ema": ema.state_dict(),
            "exposure": dict(exposure),
            "accumulation_micro_step": 0,
            "step_records": list(step_records),
        },
    )


def load_b0_checkpoint(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    model: Any,
    optimizer: Any,
    scaler: Any,
    ema: EMAState,
    cursor: Any,
    device: Any,
) -> dict[str, Any]:
    from train_step_runtime import load_exact_resume_checkpoint

    payload = load_exact_resume_checkpoint(
        path,
        expected_metadata=metadata,
        model=model,
        optimizer=optimizer,
        sampler=cursor,
        map_location=device,
    )
    extra = payload["extra_state"]
    if int(extra.get("accumulation_micro_step", -1)) != 0:
        raise B0RunnerError("B0 checkpoint is not at an accumulation boundary")
    scaler.load_state_dict(extra["grad_scaler"])
    ema.load_state_dict(extra["ema"])
    return payload


def verify_exact_resume(
    checkpoint_path: Path,
    *,
    metadata: Mapping[str, Any],
    model: Any,
    optimizer: Any,
    scaler: Any,
    ema: EMAState,
    cursor: Any,
    dataset: Any,
    model_factory: Any,
    optimizer_factory: Any,
    scaler_factory: Any,
    cursor_factory: Any,
    device: Any,
) -> tuple[Any, Any, Any, EMAState, Any, dict[str, Any]]:
    import torch
    from train_step_runtime import restore_rng_state

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_cursor = cursor_factory()
    expected_cursor.load_state_dict(raw["sampler_state"])
    expected_indices = expected_cursor.next_indices()
    expected_identity = sample_identity(dataset, expected_indices)
    restore_rng_state(raw["rng_state"])
    expected_rng = rng_probe(device)

    resumed_model = model_factory()
    resumed_optimizer = optimizer_factory(resumed_model)
    resumed_scaler = scaler_factory()
    resumed_ema = EMAState(resumed_model, ema.decay)
    resumed_cursor = cursor_factory()
    loaded = load_b0_checkpoint(
        checkpoint_path,
        metadata=metadata,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=resumed_scaler,
        ema=resumed_ema,
        cursor=resumed_cursor,
        device=device,
    )
    actual_indices = resumed_cursor.next_indices()
    actual_identity = sample_identity(dataset, actual_indices)
    actual_rng = rng_probe(device)
    gates = {
        "model_state_exact": nested_digest(model.state_dict()) == nested_digest(resumed_model.state_dict()),
        "optimizer_state_exact": nested_digest(optimizer.state_dict()) == nested_digest(resumed_optimizer.state_dict()),
        "grad_scaler_state_exact": scaler.state_dict() == resumed_scaler.state_dict(),
        "ema_state_exact": nested_digest(ema.state_dict()) == nested_digest(resumed_ema.state_dict()),
        "rng_probe_exact": rng_equal(expected_rng, actual_rng),
        "next_batch_identity_exact": (
            expected_indices == actual_indices and expected_identity == actual_identity
        ),
        "metadata_exact": loaded["metadata"] == metadata,
    }
    if not all(gates.values()):
        raise B0RunnerError(f"step-100 exact-resume probe failed: {gates}")
    # Reload a second time to return to the exact checkpoint boundary after the probes.
    load_b0_checkpoint(
        checkpoint_path,
        metadata=metadata,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=resumed_scaler,
        ema=resumed_ema,
        cursor=resumed_cursor,
        device=device,
    )
    return (
        resumed_model, resumed_optimizer, resumed_scaler, resumed_ema, resumed_cursor,
        {"gates": gates, "next_batch_identity": expected_identity},
    )


def retain_last_checkpoints(directory: Path, keep_last: int) -> None:
    checkpoints = sorted(
        directory.glob("step-*.pt"), key=lambda path: int(path.stem.split("-")[-1])
    )
    for obsolete in checkpoints[:-keep_last]:
        obsolete.unlink()


def execute_one_run(
    *,
    run: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    repo_root: Path,
    git_commit: str,
    device: Any,
) -> dict[str, Any]:
    import torch

    pretraining_dir = Path(__file__).parent
    models_dir = pretraining_dir.parent / "models"
    sys.path.insert(0, str(models_dir))
    from go_stanford_adapter import GoStanfordCanonicalDataset, collate_go_stanford_canonical
    from train_step_runtime import DeterministicBatchCursor, compute_action_loss
    from tinynavbrain_image_policy import TinyNavBrainImagePolicy
    from tinynavbrain_scaffold import TinyNavConfig

    training = config["training"]
    precision = training["precision"]
    optimizer_config = training["optimizer"]
    checkpoint_config = config["checkpoint"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = GoStanfordCanonicalDataset(
        dataset_root=repo_root / config["data"]["dataset_root"],
        manifest_path=repo_root / config["data"]["manifest"],
        split=config["data"]["split"],
        observation_frames=int(config["data"]["observation_frames"]),
        action_history_steps=int(config["data"]["action_history_steps"]),
        action_horizon=int(config["data"]["action_horizon"]),
        image_size=tuple(config["data"]["image_size"]),
        data_fraction=float(config["data"]["data_fraction"]),
        subset_seed=int(config["data"]["subset_seed"]),
    )
    micro_batch_size = int(training["micro_batch_size"])

    def cursor_factory() -> Any:
        return DeterministicBatchCursor(len(dataset), micro_batch_size, seed)

    def model_factory() -> Any:
        return TinyNavBrainImagePolicy(TinyNavConfig(), encoder_weights=None).to(device)

    def optimizer_factory(target_model: Any) -> Any:
        return torch.optim.AdamW(
            target_model.parameters(),
            lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            betas=tuple(optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]),
        )

    def scaler_factory() -> Any:
        return torch.amp.GradScaler(
            "cuda",
            init_scale=float(precision["init_scale"]),
            growth_factor=float(precision["growth_factor"]),
            backoff_factor=float(precision["backoff_factor"]),
            growth_interval=int(precision["growth_interval"]),
            enabled=bool(precision["grad_scaler"]),
        )

    model = model_factory()
    optimizer = optimizer_factory(model)
    scaler = scaler_factory()
    ema = EMAState(model, float(training["ema"]["decay"]))
    cursor = cursor_factory()
    torch.cuda.reset_peak_memory_stats(device)
    run_root = repo_root / checkpoint_config["root"] / run["id"]
    run_root.mkdir(parents=True, exist_ok=True)
    if any(run_root.glob("step-*.pt")):
        raise B0RunnerError(
            f"refusing to overwrite existing B0 checkpoints in {run_root}"
        )
    step_records: list[dict[str, Any]] = []
    exposure = {
        "examples_seen": 0,
        "micro_batches_seen": 0,
        "optimizer_steps": 0,
        "target_max_abs_m": 0.0,
    }
    resume_probe: dict[str, Any] | None = None
    training_peak_cuda_memory = 0
    config_hash = canonical_text_sha256(config_path)

    for step in range(1, int(run["optimizer_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[float] = []
        step_identities: list[str] = []
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(int(training["gradient_accumulation_steps"])):
            indices = cursor.next_indices()
            cpu_batch = collate_go_stanford_canonical([dataset[index] for index in indices])
            step_identities.extend(sample_identity(dataset, indices))
            batch = move_batch(cpu_batch, device)
            exposure["target_max_abs_m"] = max(
                float(exposure["target_max_abs_m"]),
                float(batch["target_waypoints"].detach().abs().max()),
            )
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=bool(precision["autocast"])
            ):
                action_loss, _ = compute_action_loss(model, batch, head=run["head"])
                scaled_loss = action_loss / int(training["gradient_accumulation_steps"])
            if not bool(torch.isfinite(action_loss)):
                raise B0RunnerError(f"non-finite {run['id']} loss at step {step}")
            scaler.scale(scaled_loss).backward()
            step_losses.append(float(action_loss.detach()))
            exposure["examples_seen"] += len(indices)
            exposure["micro_batches_seen"] += 1
        scaler.unscale_(optimizer)
        parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
        if not parameters or not all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
            raise B0RunnerError(f"non-finite or missing gradients at {run['id']} step {step}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(training["gradients"]["max_l2_norm"])
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise B0RunnerError(f"non-finite gradient norm at {run['id']} step {step}")
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        if scale_after < scale_before:
            raise B0RunnerError(f"GradScaler skipped {run['id']} step {step}; fail closed")
        ema.update(model)
        exposure["optimizer_steps"] = step
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        step_records.append({
            "optimizer_step": step,
            "action_loss_mean": float(np.mean(step_losses)),
            "action_loss_micro_batches": step_losses,
            "gradient_l2_norm_preclip": float(gradient_norm.detach()),
            "grad_scaler_scale_before": scale_before,
            "grad_scaler_scale_after": scale_after,
            "examples_seen_total": exposure["examples_seen"],
            "elapsed_ms_including_data_excluding_checkpoint": elapsed_ms,
            "batch_identity": step_identities,
        })
        training_peak_cuda_memory = max(
            training_peak_cuda_memory, torch.cuda.max_memory_allocated(device)
        )

        should_save = (
            step % int(checkpoint_config["save_every_optimizer_steps"]) == 0
            or step == int(run["optimizer_steps"])
        )
        checkpoint_path = run_root / f"step-{step}.pt"
        metadata = checkpoint_metadata(
            run=run, step=step, seed=seed, git_commit=git_commit,
            config_sha256=config_hash, config=config,
        )
        if should_save:
            save_b0_checkpoint(
                checkpoint_path,
                metadata=metadata,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                cursor=cursor,
                exposure=exposure,
                step_records=step_records,
            )
            retain_last_checkpoints(run_root, int(checkpoint_config["keep_last"]))
        if step == int(checkpoint_config["exact_resume_probe_optimizer_step"]):
            if not checkpoint_path.is_file():
                raise B0RunnerError("resume probe step is not a checkpoint boundary")
            model, optimizer, scaler, ema, cursor, resume_probe = verify_exact_resume(
                checkpoint_path,
                metadata=metadata,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                cursor=cursor,
                dataset=dataset,
                model_factory=model_factory,
                optimizer_factory=optimizer_factory,
                scaler_factory=scaler_factory,
                cursor_factory=cursor_factory,
                device=device,
            )
            # Resume verification temporarily holds two full model/optimizer
            # copies. Exclude that deliberate duplication from training-step
            # peak memory while retaining its exactness gates separately.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

    checkpoints = sorted(run_root.glob("step-*.pt"))
    if len(checkpoints) != int(checkpoint_config["keep_last"]):
        raise B0RunnerError(f"unexpected retained checkpoint count for {run['id']}")
    if resume_probe is None or not all(resume_probe["gates"].values()):
        raise B0RunnerError(f"missing exact-resume probe for {run['id']}")
    return {
        "run_id": run["id"],
        "head": run["head"],
        "optimizer_steps": exposure["optimizer_steps"],
        "backward_calls": exposure["micro_batches_seen"],
        "examples_seen": exposure["examples_seen"],
        "target_max_abs_m": exposure["target_max_abs_m"],
        "step_records": step_records,
        "resume_probe": resume_probe,
        "peak_cuda_memory_bytes_training_steps": training_peak_cuda_memory,
        "retained_checkpoints": [
            {"path": str(path.relative_to(repo_root)), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
            for path in checkpoints
        ],
    }


def execute_b0(
    config: Mapping[str, Any], config_path: Path, repo_root: Path,
    authorization_token: str | None, device_index: int,
) -> dict[str, Any]:
    snapshot = validate_execute_guard(config, repo_root, authorization_token)
    load_prerequisites(config, repo_root)
    import torch

    if not torch.cuda.is_available() or not 0 <= device_index < torch.cuda.device_count():
        raise B0RunnerError(f"CUDA device {device_index} is unavailable")
    device = torch.device(f"cuda:{device_index}")
    run_results = [
        execute_one_run(
            run=run, config=config, config_path=config_path, repo_root=repo_root,
            git_commit=snapshot["commit"], device=device,
        )
        for run in config["runs"]
    ]
    total_steps = sum(result["optimizer_steps"] for result in run_results)
    total_backward = sum(result["backward_calls"] for result in run_results)
    if total_steps != config["training"]["total_optimizer_steps_if_authorized"]:
        raise B0RunnerError("executed optimizer-step count differs from frozen budget")
    if total_backward != config["training"]["total_backward_calls_if_authorized"]:
        raise B0RunnerError("executed backward count differs from frozen budget")
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": "0.1.0",
        "run_id": config["config_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_text_sha256(config_path),
        "runner_sha256": file_sha256(Path(__file__)),
        "git": snapshot,
        "hardware": {"device_index": device_index, "name": properties.name},
        "runs": run_results,
        "execution_boundary": {
            "mode": "execute",
            "optimizer_steps": total_steps,
            "backward_calls": total_backward,
            "evaluation_performed": False,
            "pretrained_weights_downloaded": False,
        },
        "claim_boundary": (
            "B0 is a plumbing smoke only. Loss trajectories, timing, and memory are not "
            "model-quality, convergence, deployment, or method evidence."
        ),
    }


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    boundary = report["execution_boundary"]
    if boundary["mode"] == "dry-run":
        lines = [
            "# TinyNavBrain B0 Static Readiness",
            "",
            f"- Config: `{report['config_id']}`",
            f"- Config SHA-256: `{report['config_sha256']}`",
            f"- Planned optimizer steps: {report['planned_budget']['optimizer_steps']}",
            f"- Planned backward calls: {report['planned_budget']['backward_calls']}",
            "- Model instantiated: False",
            "- Optimizer created: False",
            "- Forward/backward/optimizer steps: 0/0/0",
            f"- Ready for execution authorization: {report['ready_for_execution_authorization']}",
            "",
            "## Gates",
            "",
        ]
        lines.extend(f"- {name}: {value}" for name, value in report["gates"].items())
        lines.extend(["", "## Evidence boundary", "", report["claim_boundary"]])
    else:
        lines = [
            "# TinyNavBrain B0 Execution",
            "",
            f"- Optimizer steps: {boundary['optimizer_steps']}",
            f"- Backward calls: {boundary['backward_calls']}",
            "",
            "## Evidence boundary",
            "",
            report["claim_boundary"],
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-token")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_frozen_config(args.config, args.repo_root)
        if args.dry_run:
            report = dry_run_report(config, args.config, args.repo_root)
        else:
            report = execute_b0(
                config, args.config, args.repo_root,
                args.authorization_token, args.device_index,
            )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as exc:
        failure = {
            "schema_version": "0.1.0",
            "status": "failed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "dry-run" if args.dry_run else "execute",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "claim_boundary": "Failure receipt only; no successful B0 result may be inferred.",
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            "# TinyNavBrain B0 Failure\n\n"
            f"- Mode: {failure['mode']}\n"
            f"- Error type: {failure['error_type']}\n"
            f"- Error: {failure['error']}\n\n"
            "No successful B0 result may be inferred.\n",
            encoding="utf-8",
        )
        print(f"ERROR: {exc}")
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, report)
    print(
        "B0 static dry-run passed; no training executed"
        if args.dry_run else "B0 execution completed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
