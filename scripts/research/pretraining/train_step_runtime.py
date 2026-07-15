"""Executable TinyNavBrain train-step and exact-resume checkpoint utilities.

This module contains the smallest runtime needed to exercise the frozen H0/H1
loss contracts.  Importing it performs no model execution, optimizer creation,
checkpoint I/O, or training.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from checkpoint_contract import validate_checkpoint_metadata
from training_contract import (
    deterministic_h0_loss,
    rectified_flow_h1_loss,
    rectified_flow_state,
)


class TrainStepRuntimeError(RuntimeError):
    """Raised when an executable training or resume invariant fails."""


@dataclass(frozen=True)
class TrainStepResult:
    head: str
    loss: float
    per_dataset_loss: dict[str, float]
    gradient_l2_norm: float
    finite_gradients: bool
    optimizer_step_performed: bool


class DeterministicBatchCursor:
    """A serializable deterministic sample order with an exact batch cursor."""

    def __init__(self, dataset_length: int, batch_size: int, seed: int) -> None:
        if dataset_length < 1 or batch_size < 1 or seed < 0:
            raise ValueError("dataset_length/batch_size must be positive and seed non-negative")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.dataset_length = dataset_length
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.cursor = 0
        self.order = torch.randperm(dataset_length, generator=generator).tolist()

    def next_indices(self) -> list[int]:
        if self.cursor + self.batch_size > self.dataset_length:
            raise TrainStepRuntimeError("batch cursor reached the end of the frozen epoch")
        indices = self.order[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return indices

    def state_dict(self) -> dict[str, Any]:
        order_bytes = np.asarray(self.order, dtype=np.int64).tobytes()
        return {
            "schema_version": "0.1.0",
            "dataset_length": self.dataset_length,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": list(self.order),
            "order_sha256": hashlib.sha256(order_bytes).hexdigest(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {
            "schema_version", "dataset_length", "batch_size", "seed", "epoch",
            "cursor", "order", "order_sha256",
        }
        missing = sorted(required - set(state))
        if missing:
            raise TrainStepRuntimeError(f"sampler state missing fields: {missing}")
        if state["schema_version"] != "0.1.0":
            raise TrainStepRuntimeError("unsupported sampler state schema")
        immutable = ("dataset_length", "batch_size", "seed")
        for name in immutable:
            if int(state[name]) != getattr(self, name):
                raise TrainStepRuntimeError(f"sampler resume mismatch: {name}")
        order = [int(item) for item in state["order"]]
        if len(order) != self.dataset_length or sorted(order) != list(range(self.dataset_length)):
            raise TrainStepRuntimeError("sampler order is not a dataset permutation")
        actual_hash = hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest()
        if actual_hash != state["order_sha256"]:
            raise TrainStepRuntimeError("sampler order checksum mismatch")
        cursor = int(state["cursor"])
        if cursor < 0 or cursor > self.dataset_length:
            raise TrainStepRuntimeError("sampler cursor is outside the dataset")
        self.order = order
        self.epoch = int(state["epoch"])
        self.cursor = cursor


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required - set(state))
    if missing:
        raise TrainStepRuntimeError(f"RNG state missing fields: {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # A checkpoint loaded with ``map_location=cuda`` also moves RNG byte
    # tensors.  PyTorch's RNG setters require CPU ByteTensor state, so
    # normalize explicitly before restoring.
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if torch.cuda.is_available():
        cuda_state = state["torch_cuda"]
        if len(cuda_state) != torch.cuda.device_count():
            raise TrainStepRuntimeError("CUDA RNG state does not match visible device count")
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in cuda_state])


def compute_action_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    head: str,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    target = batch["target_waypoints"]
    target_mask = batch["target_mask"]
    dataset_ids = batch["dataset_id"]
    if not isinstance(target, Tensor) or not isinstance(target_mask, Tensor):
        raise TrainStepRuntimeError("canonical target tensors are missing")
    if not isinstance(dataset_ids, Sequence) or isinstance(dataset_ids, (str, bytes)):
        raise TrainStepRuntimeError("dataset_id must be a sequence")
    if head == "h0_deterministic":
        output = model(batch, head=head)
        return deterministic_h0_loss(
            output["waypoints_m"], target, target_mask, dataset_ids
        )
    if head == "h1_rectified_flow":
        time = torch.rand(
            target.shape[0], 1, device=target.device, dtype=target.dtype, generator=generator
        )
        noise = torch.randn(
            target.shape, device=target.device, dtype=target.dtype, generator=generator
        )
        state, velocity_target = rectified_flow_state(target, time, noise)
        output = model(batch, head="h1_velocity", x_t=state, t=time)
        return rectified_flow_h1_loss(
            output["velocity"], velocity_target, target_mask, dataset_ids
        )
    raise TrainStepRuntimeError(f"unsupported train head {head!r}")


def optimizer_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    *,
    head: str,
    generator: torch.Generator | None = None,
    max_gradient_norm: float | None = None,
) -> TrainStepResult:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, per_dataset = compute_action_loss(model, batch, head=head, generator=generator)
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise TrainStepRuntimeError("action loss is not a finite scalar")
    loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        raise TrainStepRuntimeError("backward produced no parameter gradients")
    finite_gradients = all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)
    if not finite_gradients:
        raise TrainStepRuntimeError("backward produced non-finite gradients")
    if max_gradient_norm is not None:
        if max_gradient_norm <= 0:
            raise TrainStepRuntimeError("max_gradient_norm must be positive")
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
    else:
        squared = torch.stack(
            [parameter.grad.detach().float().norm(2).square() for parameter in parameters]
        ).sum()
        gradient_norm = squared.sqrt()
    if not bool(torch.isfinite(gradient_norm)) or float(gradient_norm) <= 0:
        raise TrainStepRuntimeError("gradient L2 norm must be finite and positive")
    optimizer.step()
    return TrainStepResult(
        head=head,
        loss=float(loss.detach()),
        per_dataset_loss={name: float(value.detach()) for name, value in per_dataset.items()},
        gradient_l2_norm=float(gradient_norm.detach()),
        finite_gradients=finite_gradients,
        optimizer_step_performed=True,
    )


def save_exact_resume_checkpoint(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler_state: Mapping[str, Any],
    extra_state: Mapping[str, Any] | None = None,
) -> None:
    errors = validate_checkpoint_metadata(metadata)
    if errors:
        raise TrainStepRuntimeError(f"invalid checkpoint metadata: {errors}")
    payload = {
        "metadata": dict(metadata),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "sampler_state": dict(sampler_state),
        "extra_state": dict(extra_state or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_exact_resume_checkpoint(
    path: Path,
    *,
    expected_metadata: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: DeterministicBatchCursor,
    map_location: torch.device | str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, MutableMapping):
        raise TrainStepRuntimeError("checkpoint root must be a mapping")
    required = {"metadata", "model_state", "optimizer_state", "rng_state", "sampler_state"}
    missing = sorted(required - set(payload))
    if missing:
        raise TrainStepRuntimeError(f"checkpoint missing fields: {missing}")
    errors = validate_checkpoint_metadata(
        payload["metadata"], expected_resume=expected_metadata
    )
    if errors:
        raise TrainStepRuntimeError(f"checkpoint resume metadata failed: {errors}")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    sampler.load_state_dict(payload["sampler_state"])
    restore_rng_state(payload["rng_state"])
    return dict(payload)
