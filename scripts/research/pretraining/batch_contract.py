"""Canonical TinyNavBrain batch adapters and past-only action-history helpers.

These functions only transform or validate tensors. They do not load datasets,
run a model, call backward, or step an optimizer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


class BatchContractError(ValueError):
    """Raised when a batch could violate the frozen data contract."""


def _require_string_list(value: Any, name: str, batch_size: int) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BatchContractError(f"{name} must be a sequence of strings")
    result = [str(item).strip() for item in value]
    if len(result) != batch_size or any(not item for item in result):
        raise BatchContractError(f"{name} must contain {batch_size} non-empty strings")
    return result


def _batch_column(value: Any, name: str, batch_size: int) -> Tensor:
    if value is None:
        raise BatchContractError(f"{name} is required")
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise BatchContractError(f"{name} must contain numeric values") from exc
    if tuple(tensor.shape) == (batch_size,):
        tensor = tensor[:, None]
    if tuple(tensor.shape) != (batch_size, 1):
        raise BatchContractError(f"{name} must be [B,1]")
    if not bool(torch.isfinite(tensor).all()) or bool(torch.any(tensor <= 0)):
        raise BatchContractError(f"{name} must be finite and positive")
    return tensor


def adapt_legacy_vint_batch(
    legacy_batch: Sequence[Tensor],
    metadata: Mapping[str, Any],
    *,
    observation_frames: int = 6,
    action_history_steps: int = 4,
    action_horizon: int = 8,
) -> dict[str, Any]:
    """Convert the existing seven-item ViNT batch without changing its tuple API.

    ``target_scale_m`` is mandatory because legacy action labels may be either
    normalized or metric. The caller must provide the reversible scale explicitly;
    the adapter never guesses it from ``dataset_id``.
    """

    if len(legacy_batch) != 7:
        raise BatchContractError("legacy ViNT batch must contain exactly seven items")
    obs, goal, actions, distance, goal_pos, dataset_index, action_mask = legacy_batch
    if obs.ndim != 4:
        raise BatchContractError("legacy obs_images must be [B,3*frames,H,W]")
    batch_size, channels, height, width = obs.shape
    if channels != observation_frames * 3:
        raise BatchContractError(
            f"legacy obs channel count {channels} != 3*{observation_frames}"
        )
    if tuple(goal.shape) != (batch_size, 3, height, width):
        raise BatchContractError("legacy goal_image shape does not match observations")
    if actions.ndim != 3 or actions.shape[0] != batch_size:
        raise BatchContractError("legacy action_label must be [B,T,D]")
    if actions.shape[1] != action_horizon or actions.shape[2] < 2:
        raise BatchContractError(f"legacy action_label must provide [B,{action_horizon},>=2]")

    dataset_ids = _require_string_list(metadata.get("dataset_id"), "dataset_id", batch_size)
    trajectory_ids = _require_string_list(
        metadata.get("trajectory_id"), "trajectory_id", batch_size
    )
    dt_s = _batch_column(metadata.get("dt_s"), "dt_s", batch_size)
    waypoint_spacing_m = _batch_column(
        metadata.get("waypoint_spacing_m"), "waypoint_spacing_m", batch_size
    )
    target_scale_m = _batch_column(
        metadata.get("target_scale_m"), "target_scale_m", batch_size
    )

    goal_mask = torch.as_tensor(metadata.get("goal_mask"))
    if tuple(goal_mask.shape) != (batch_size,) or goal_mask.dtype != torch.bool:
        raise BatchContractError("goal_mask must be bool [B]")
    history = torch.as_tensor(metadata.get("action_history"), dtype=torch.float32)
    history_mask = torch.as_tensor(metadata.get("action_history_mask"))
    if tuple(history.shape) != (batch_size, action_history_steps, 3):
        raise BatchContractError(f"action_history must be [B,{action_history_steps},3]")
    if tuple(history_mask.shape) != (batch_size, action_history_steps):
        raise BatchContractError(f"action_history_mask must be [B,{action_history_steps}]")
    if history_mask.dtype != torch.bool:
        raise BatchContractError("action_history_mask must be bool")

    per_sample_action_mask = torch.as_tensor(action_mask).reshape(batch_size).to(torch.bool)
    target_mask = (per_sample_action_mask & ~goal_mask)[:, None].expand(-1, action_horizon).clone()
    batch = {
        "obs_images": obs.reshape(batch_size, observation_frames, 3, height, width),
        "goal_image": goal,
        "goal_mask": goal_mask,
        "action_history": history,
        "action_history_mask": history_mask,
        "dt_s": dt_s,
        "waypoint_spacing_m": waypoint_spacing_m,
        "target_waypoints": actions[..., :2].to(torch.float32) * target_scale_m[:, None, :],
        "target_mask": target_mask,
        "dataset_id": dataset_ids,
        "trajectory_id": trajectory_ids,
        "legacy_distance_label": distance,
        "legacy_goal_pos": goal_pos,
        "legacy_dataset_index": dataset_index,
    }
    validate_canonical_batch(
        batch,
        observation_frames=observation_frames,
        action_history_steps=action_history_steps,
        action_horizon=action_horizon,
    )
    return batch


def validate_canonical_batch(
    batch: Mapping[str, Any],
    *,
    observation_frames: int = 6,
    action_history_steps: int = 4,
    action_horizon: int = 8,
) -> None:
    required = {
        "obs_images",
        "goal_image",
        "goal_mask",
        "action_history",
        "action_history_mask",
        "dt_s",
        "waypoint_spacing_m",
        "target_waypoints",
        "target_mask",
        "dataset_id",
        "trajectory_id",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise BatchContractError(f"canonical batch missing fields: {missing}")
    obs = batch["obs_images"]
    if not isinstance(obs, Tensor) or obs.ndim != 5:
        raise BatchContractError("obs_images must be a tensor [B,F,3,H,W]")
    batch_size, frames, channels, height, width = obs.shape
    if frames != observation_frames or channels != 3 or height < 1 or width < 1:
        raise BatchContractError("obs_images violates the frozen image shape contract")
    expected_shapes = {
        "goal_image": (batch_size, 3, height, width),
        "goal_mask": (batch_size,),
        "action_history": (batch_size, action_history_steps, 3),
        "action_history_mask": (batch_size, action_history_steps),
        "dt_s": (batch_size, 1),
        "waypoint_spacing_m": (batch_size, 1),
        "target_waypoints": (batch_size, action_horizon, 2),
        "target_mask": (batch_size, action_horizon),
    }
    for name, expected in expected_shapes.items():
        value = batch[name]
        if not isinstance(value, Tensor) or tuple(value.shape) != expected:
            raise BatchContractError(f"{name} shape must be {expected}")
    for name in ("goal_mask", "action_history_mask", "target_mask"):
        if batch[name].dtype != torch.bool:
            raise BatchContractError(f"{name} must be bool")
    for name in ("obs_images", "goal_image", "action_history", "target_waypoints"):
        if not bool(torch.isfinite(batch[name]).all()):
            raise BatchContractError(f"{name} contains non-finite values")
    for name in ("dt_s", "waypoint_spacing_m"):
        value = batch[name]
        if not bool(torch.isfinite(value).all()) or bool(torch.any(value <= 0)):
            raise BatchContractError(f"{name} must be finite and positive")
    padded_history = batch["action_history"][~batch["action_history_mask"]]
    if padded_history.numel() and not bool(torch.equal(padded_history, torch.zeros_like(padded_history))):
        raise BatchContractError("padded action_history entries must be exactly zero")
    if bool(torch.any(batch["target_mask"][batch["goal_mask"]])):
        raise BatchContractError("goal-masked samples cannot contribute action targets")
    _require_string_list(batch["dataset_id"], "dataset_id", batch_size)
    _require_string_list(batch["trajectory_id"], "trajectory_id", batch_size)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def build_past_action_history(
    positions_xy: Tensor,
    yaw_rad: Tensor,
    *,
    current_index: int,
    history_steps: int = 4,
    frame_stride: int = 1,
) -> tuple[Tensor, Tensor]:
    """Build past relative actions using indices no later than ``current_index``.

    Each valid row is the motion from a past pose to its following sampled pose,
    expressed in the earlier pose's local frame. Left padding is exactly zero.
    """

    positions = torch.as_tensor(positions_xy, dtype=torch.float32)
    yaw = torch.as_tensor(yaw_rad, dtype=torch.float32).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] != yaw.shape[0]:
        raise BatchContractError("positions_xy and yaw_rad must be aligned [N,2] and [N]")
    if history_steps < 1 or frame_stride < 1:
        raise BatchContractError("history_steps and frame_stride must be positive")
    if current_index < 0 or current_index >= positions.shape[0]:
        raise BatchContractError("current_index is out of range")
    if not bool(torch.isfinite(positions).all()) or not bool(torch.isfinite(yaw).all()):
        raise BatchContractError("pose history contains non-finite values")

    history = torch.zeros(history_steps, 3, dtype=torch.float32)
    mask = torch.zeros(history_steps, dtype=torch.bool)
    interval_ends = list(
        range(current_index - (history_steps - 1) * frame_stride, current_index + 1, frame_stride)
    )
    for output_index, end_index in enumerate(interval_ends):
        start_index = end_index - frame_stride
        if start_index < 0:
            continue
        delta = positions[end_index] - positions[start_index]
        start_yaw = float(yaw[start_index])
        cosine, sine = math.cos(start_yaw), math.sin(start_yaw)
        local_x = float(delta[0]) * cosine + float(delta[1]) * sine
        local_y = -float(delta[0]) * sine + float(delta[1]) * cosine
        history[output_index] = torch.tensor(
            [local_x, local_y, _wrap_angle(float(yaw[end_index] - yaw[start_index]))]
        )
        mask[output_index] = True
    return history, mask
