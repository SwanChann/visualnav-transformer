"""Full image-to-waypoint TinyNavBrain interface with a shared EfficientNet-B0.

The encoder is deliberately constructed with ``weights=None`` so importing or
instantiating this module cannot download pretrained weights.  This file defines
model structure only; it does not create optimizers, run backward, train, or
write checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import efficientnet_b0

from tinynavbrain_scaffold import TinyNavBrainScaffold, TinyNavConfig


class TinyNavImagePolicyError(ValueError):
    """Raised when inputs or configuration violate the image-policy contract."""


class SharedEfficientNetB0(nn.Module):
    """One random-initialized RGB encoder shared by observations and goals."""

    output_dim = 1280

    def __init__(self, *, weights: None = None) -> None:
        super().__init__()
        if weights is not None:
            raise TinyNavImagePolicyError(
                "pretrained weights are disabled in the current no-download gate"
            )
        backbone = efficientnet_b0(weights=None)
        classifier = backbone.classifier
        if not isinstance(classifier, nn.Sequential) or not isinstance(classifier[-1], nn.Linear):
            raise TinyNavImagePolicyError("unexpected torchvision EfficientNet-B0 classifier")
        if classifier[-1].in_features != self.output_dim:
            raise TinyNavImagePolicyError("unexpected EfficientNet-B0 feature dimension")
        backbone.classifier = nn.Identity()
        self.backbone = backbone

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise TinyNavImagePolicyError("images must be [N,3,H,W]")
        if images.shape[2] < 32 or images.shape[3] < 32:
            raise TinyNavImagePolicyError("EfficientNet inputs must be at least 32x32")
        if not bool(torch.isfinite(images).all()):
            raise TinyNavImagePolicyError("images contain non-finite values")
        features = self.backbone(images)
        if tuple(features.shape) != (images.shape[0], self.output_dim):
            raise TinyNavImagePolicyError(
                f"encoder output shape {tuple(features.shape)} != {(images.shape[0], self.output_dim)}"
            )
        return features


class TinyNavBrainImagePolicy(nn.Module):
    """Shared-image-encoder policy exposing the frozen H0/H1 interfaces."""

    def __init__(
        self,
        config: TinyNavConfig = TinyNavConfig(),
        *,
        encoder_weights: None = None,
    ) -> None:
        super().__init__()
        if config.encoder_dim != SharedEfficientNetB0.output_dim:
            raise TinyNavImagePolicyError(
                f"config.encoder_dim must be {SharedEfficientNetB0.output_dim}"
            )
        self.config = config
        self.image_encoder = SharedEfficientNetB0(weights=encoder_weights)
        self.policy = TinyNavBrainScaffold(config)
        parameter_count = self.trainable_parameter_count()
        if parameter_count >= config.max_trainable_parameters:
            raise TinyNavImagePolicyError(
                f"trainable parameters {parameter_count} exceed budget "
                f"{config.max_trainable_parameters}"
            )

    def _validate_batch(self, batch: Mapping[str, Any]) -> None:
        required = {
            "obs_images",
            "goal_image",
            "goal_mask",
            "action_history",
            "action_history_mask",
            "dt_s",
            "waypoint_spacing_m",
        }
        missing = sorted(required - set(batch))
        if missing:
            raise TinyNavImagePolicyError(f"batch missing fields: {missing}")
        obs = batch["obs_images"]
        goal = batch["goal_image"]
        if not isinstance(obs, Tensor) or obs.ndim != 5:
            raise TinyNavImagePolicyError("obs_images must be [B,F,3,H,W]")
        batch_size, frames, channels, height, width = obs.shape
        if frames != self.config.observation_frames or channels != 3:
            raise TinyNavImagePolicyError("obs_images violates the frozen frame/channel contract")
        if not isinstance(goal, Tensor) or tuple(goal.shape) != (batch_size, 3, height, width):
            raise TinyNavImagePolicyError("goal_image shape does not match observations")
        devices = {
            value.device
            for name in required
            if isinstance((value := batch[name]), Tensor)
        }
        if len(devices) != 1:
            raise TinyNavImagePolicyError("all tensor inputs must be on one device")

    def encode(self, batch: Mapping[str, Any]) -> Tensor:
        self._validate_batch(batch)
        obs = batch["obs_images"]
        goal = batch["goal_image"]
        batch_size, frames, channels, height, width = obs.shape
        flattened_obs = obs.reshape(batch_size * frames, channels, height, width)
        # One call through one encoder instance enforces shared observation/goal weights.
        all_features = self.image_encoder(torch.cat([flattened_obs, goal], dim=0))
        obs_features = all_features[: batch_size * frames].reshape(
            batch_size, frames, self.config.encoder_dim
        )
        goal_features = all_features[batch_size * frames :]
        return self.policy.encode(
            obs_features=obs_features,
            goal_features=goal_features,
            goal_mask=batch["goal_mask"],
            action_history=batch["action_history"],
            action_history_mask=batch["action_history_mask"],
            dt_s=batch["dt_s"],
            waypoint_spacing_m=batch["waypoint_spacing_m"],
        )

    def predict_deterministic(self, batch: Mapping[str, Any]) -> Tensor:
        return self.policy.predict_deterministic(self.encode(batch))

    def flow_velocity(self, batch: Mapping[str, Any], x_t: Tensor, t: Tensor) -> Tensor:
        return self.policy.flow_velocity(x_t, t, self.encode(batch))

    def sample_flow(
        self,
        batch: Mapping[str, Any],
        *,
        nfe: int = 2,
        candidate_count: int = 1,
        seed: int | None = None,
    ) -> Tensor:
        return self.policy.sample_flow(
            self.encode(batch), nfe=nfe, candidate_count=candidate_count, seed=seed
        )

    def predict_progress(self, batch: Mapping[str, Any]) -> Tensor:
        return self.policy.predict_progress(self.encode(batch))

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        head: str = "h0_deterministic",
        x_t: Tensor | None = None,
        t: Tensor | None = None,
        nfe: int = 2,
        candidate_count: int = 1,
        seed: int | None = None,
    ) -> dict[str, Tensor]:
        context = self.encode(batch)
        output = {
            "context": context,
            "progress_m": self.policy.predict_progress(context),
        }
        if head == "h0_deterministic":
            output["waypoints_m"] = self.policy.predict_deterministic(context)
        elif head == "h1_velocity":
            if x_t is None or t is None:
                raise TinyNavImagePolicyError("h1_velocity requires x_t and t")
            output["velocity"] = self.policy.flow_velocity(x_t, t, context)
        elif head == "h1_rectified_flow":
            output["waypoint_candidates_m"] = self.policy.sample_flow(
                context,
                nfe=nfe,
                candidate_count=candidate_count,
                seed=seed,
            )
        else:
            raise TinyNavImagePolicyError(f"unknown head {head!r}")
        return output

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def encoder_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.image_encoder.parameters() if parameter.requires_grad
        )
