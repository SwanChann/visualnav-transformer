"""Pure H0/H1 loss contracts; no backward pass or optimizer is defined here."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


class TrainingContractError(ValueError):
    pass


def _dataset_macro_reduce(
    elementwise_loss: Tensor, target_mask: Tensor, dataset_ids: Sequence[str]
) -> tuple[Tensor, dict[str, Tensor]]:
    if elementwise_loss.ndim != 3:
        raise TrainingContractError("elementwise loss must be [B,T,D]")
    batch_size, horizon, dimensions = elementwise_loss.shape
    if tuple(target_mask.shape) != (batch_size, horizon) or target_mask.dtype != torch.bool:
        raise TrainingContractError("target_mask must be bool [B,T]")
    if len(dataset_ids) != batch_size:
        raise TrainingContractError("dataset_ids length must equal batch size")
    expanded_mask = target_mask[:, :, None].expand(-1, -1, dimensions)
    valid_counts = expanded_mask.sum(dim=(1, 2))
    sample_loss = (elementwise_loss * expanded_mask).sum(dim=(1, 2)) / valid_counts.clamp_min(1)
    per_dataset: dict[str, Tensor] = {}
    for dataset_id in sorted({str(item) for item in dataset_ids}):
        selection = torch.tensor(
            [str(item) == dataset_id for item in dataset_ids],
            device=elementwise_loss.device,
            dtype=torch.bool,
        ) & (valid_counts > 0)
        if bool(selection.any()):
            per_dataset[dataset_id] = sample_loss[selection].mean()
    if not per_dataset:
        raise TrainingContractError("batch has no valid action target")
    return torch.stack(list(per_dataset.values())).mean(), per_dataset


def deterministic_h0_loss(
    prediction_m: Tensor,
    target_m: Tensor,
    target_mask: Tensor,
    dataset_ids: Sequence[str],
) -> tuple[Tensor, dict[str, Tensor]]:
    if prediction_m.shape != target_m.shape or prediction_m.ndim != 3:
        raise TrainingContractError("H0 prediction and target must share [B,T,D]")
    elementwise = F.smooth_l1_loss(prediction_m, target_m, reduction="none")
    return _dataset_macro_reduce(elementwise, target_mask, dataset_ids)

def rectified_flow_state(target_m: Tensor, time: Tensor, noise: Tensor) -> tuple[Tensor, Tensor]:
    if target_m.shape != noise.shape or target_m.ndim != 3:
        raise TrainingContractError("flow target and noise must share [B,T,D]")
    if tuple(time.shape) != (target_m.shape[0], 1):
        raise TrainingContractError("flow time must be [B,1]")
    if bool(torch.any(time < 0)) or bool(torch.any(time > 1)):
        raise TrainingContractError("flow time must be within [0,1]")
    broadcast_time = time[:, :, None]
    state = (1.0 - broadcast_time) * target_m + broadcast_time * noise
    velocity_target = noise - target_m
    return state, velocity_target


def rectified_flow_h1_loss(
    velocity_prediction: Tensor,
    velocity_target: Tensor,
    target_mask: Tensor,
    dataset_ids: Sequence[str],
) -> tuple[Tensor, dict[str, Tensor]]:
    if velocity_prediction.shape != velocity_target.shape or velocity_prediction.ndim != 3:
        raise TrainingContractError("H1 velocity prediction and target must share [B,T,D]")
    elementwise = F.mse_loss(velocity_prediction, velocity_target, reduction="none")
    return _dataset_macro_reduce(elementwise, target_mask, dataset_ids)
