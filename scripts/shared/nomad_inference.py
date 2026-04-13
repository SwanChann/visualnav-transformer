#!/usr/bin/env python3
"""Shared NoMaD inference module for simulation and deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from torchvision import transforms

from project_paths import add_repo_paths, repo_path

add_repo_paths()

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn


DEFAULT_POLICY_CONFIG = repo_path("train", "config", "nomad.yaml")
DEFAULT_POLICY_CHECKPOINT_CANDIDATES = [
    repo_path("deployment", "model_weights", "nomad", "nomad.pth"),
    repo_path("deployment", "model_weights", "nomad.pth"),
]
DEFAULT_ACTION_STATS = {
    "min": np.array([-2.5, -4.0], dtype=np.float32),
    "max": np.array([5.0, 4.0], dtype=np.float32),
}


@dataclass(frozen=True)
class NoMaDInferenceSpec:
    policy_config: str | None = None
    policy_checkpoint: str | None = None
    scheduler_kind: str = "ddpm"
    ddim_steps: int = 10
    cfg_weight: float = 0.0
    device: str | None = None
    num_samples: int = 8


def resolve_repo_relative(path_value: str | None, default_path: Path | None = None) -> Path:
    """Resolve a repo-relative path."""
    if path_value is None:
        if default_path is None:
            raise ValueError("A path value or default_path is required.")
        return default_path.resolve()
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return repo_path(*path.parts).resolve()


def load_yaml_config(config_path: Path) -> dict:
    """Load one YAML config file."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def extract_state_dict(payload) -> Mapping[str, torch.Tensor]:
    """Extract a model state dict from common checkpoint formats."""
    if hasattr(payload, "state_dict"):
        payload = payload.state_dict()
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")

    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    elif "model" in payload:
        candidate = payload["model"]
        if hasattr(candidate, "state_dict"):
            payload = candidate.state_dict()
        elif isinstance(candidate, dict):
            payload = candidate

    cleaned = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


class NoMaDInferenceModule:
    """Unified policy loader plus DDPM/DDIM/CFG inference controller."""

    def __init__(self, spec: NoMaDInferenceSpec | None = None) -> None:
        self.spec = spec or NoMaDInferenceSpec()
        self.device = torch.device(
            self.spec.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config_path = resolve_repo_relative(self.spec.policy_config, DEFAULT_POLICY_CONFIG)
        self.config = load_yaml_config(self.config_path)
        self.checkpoint_path = self._resolve_checkpoint(self.spec.policy_checkpoint)
        self.image_size = tuple(int(value) for value in self.config.get("image_size", [96, 96]))
        self.context_size = int(self.config.get("context_size", 3))
        self.len_traj_pred = int(self.config.get("len_traj_pred", 8))
        self.num_diffusion_iters = int(self.config.get("num_diffusion_iters", 10))
        self.encoding_size = int(self.config.get("encoding_size", 256))
        self.num_samples = int(self.spec.num_samples)
        self.scheduler_kind = str(self.spec.scheduler_kind)
        self.ddim_steps = int(self.spec.ddim_steps)
        self.cfg_weight = float(self.spec.cfg_weight)
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.model = self._build_model()

    def _resolve_checkpoint(self, explicit_path: str | None) -> Path:
        if explicit_path:
            resolved = resolve_repo_relative(explicit_path)
            if not resolved.exists():
                raise FileNotFoundError(f"Policy checkpoint not found: {resolved}")
            return resolved
        for candidate in DEFAULT_POLICY_CHECKPOINT_CANDIDATES:
            if candidate.exists():
                return candidate.resolve()
        return DEFAULT_POLICY_CHECKPOINT_CANDIDATES[0].resolve()

    def _build_model(self) -> NoMaD:
        vision_encoder_name = str(self.config.get("vision_encoder", "nomad_vint"))

        if vision_encoder_name == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                context_size=self.context_size,
                obs_encoder=self.config.get("obs_encoder", "efficientnet-b0"),
                obs_encoding_size=self.encoding_size,
                mha_num_attention_heads=self.config.get("mha_num_attention_heads", 4),
                mha_num_attention_layers=self.config.get("mha_num_attention_layers", 4),
                mha_ff_dim_factor=self.config.get("mha_ff_dim_factor", 4),
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
            model = NoMaD(
                vision_encoder=vision_encoder,
                noise_pred_net=ConditionalUnet1D(
                    input_dim=2,
                    global_cond_dim=self.encoding_size,
                    down_dims=self.config.get("down_dims", [64, 128, 256]),
                    cond_predict_scale=self.config.get("cond_predict_scale", False),
                ),
                dist_pred_net=DenseNetwork(embedding_dim=self.encoding_size),
            )
        elif vision_encoder_name == "nomad_vint_dinov2":
            from nomad_vint_dinov2 import NoMaD_ViNT_DINOv2

            vision_encoder = NoMaD_ViNT_DINOv2(
                context_size=self.context_size,
                obs_encoding_size=self.encoding_size,
                mha_num_attention_heads=self.config.get("mha_num_attention_heads", 4),
                mha_num_attention_layers=self.config.get("mha_num_attention_layers", 4),
                mha_ff_dim_factor=self.config.get("mha_ff_dim_factor", 4),
                freeze_backbone=bool(self.config.get("freeze_backbone", True)),
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
            model = NoMaD(
                vision_encoder=vision_encoder,
                noise_pred_net=ConditionalUnet1D(
                    input_dim=2,
                    global_cond_dim=self.encoding_size,
                    down_dims=self.config.get("down_dims", [64, 128, 256]),
                    cond_predict_scale=self.config.get("cond_predict_scale", False),
                ),
                dist_pred_net=DenseNetwork(embedding_dim=self.encoding_size),
            )
        elif vision_encoder_name == "nomad_vint_backbone_suite":
            from nomad_vint_backbone_suite import build_backbone_nomad_model

            backbone_name = str(self.config.get("encoder_backbone", "dinov2_small"))
            model = build_backbone_nomad_model(
                config=self.config,
                backbone_name=backbone_name,
                freeze_backbone=bool(self.config.get("freeze_backbone", True)),
                pretrained_backbone=False,
                share_goal_encoder=bool(self.config.get("share_goal_encoder", True)),
            )
        else:
            raise ValueError(f"Unsupported vision encoder for inference: {vision_encoder_name}")

        state_dict = extract_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device).eval()
        return model

    def describe(self) -> str:
        vision_encoder_name = str(self.config.get("vision_encoder", "nomad_vint"))
        if vision_encoder_name == "nomad_vint_backbone_suite":
            policy_name = str(self.config.get("encoder_backbone", "unknown"))
        else:
            policy_name = str(self.config.get("obs_encoder", vision_encoder_name))
        return (
            f"policy={policy_name}, vision_encoder={vision_encoder_name}, "
            f"checkpoint={self.checkpoint_path}"
        )

    def pil_to_tensor(self, pil_img: PILImage.Image) -> torch.Tensor:
        """Convert one PIL image into a normalized tensor."""
        return self._transform(pil_img.convert("RGB").resize(self.image_size))

    def build_obs_tensor(self, frame_buffer) -> torch.Tensor:
        """Build the stacked observation tensor expected by NoMaD."""
        frames = list(frame_buffer)
        while len(frames) < self.context_size + 1:
            frames.insert(0, frames[0].clone())
        frames = frames[-(self.context_size + 1) :]
        return torch.cat(frames, dim=0).unsqueeze(0)

    def _make_scheduler(self):
        common_kwargs = dict(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        if self.scheduler_kind == "ddpm":
            return DDPMScheduler(**common_kwargs)
        if self.scheduler_kind == "ddim":
            return DDIMScheduler(**common_kwargs)
        raise ValueError(f"Unsupported scheduler kind: {self.scheduler_kind}")

    def effective_num_steps(self) -> int:
        return self.ddim_steps if self.scheduler_kind == "ddim" else self.num_diffusion_iters

    def encode_condition(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        goal_mask_value: int,
    ) -> torch.Tensor:
        """Encode the NoMaD visual condition."""
        batch_size = obs_img.shape[0]
        mask = torch.full((batch_size,), int(goal_mask_value), dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self.model(
                "vision_encoder",
                obs_img=obs_img.to(self.device),
                goal_img=goal_img.to(self.device),
                input_goal_mask=mask,
            )

    def predict_distance(self, condition: torch.Tensor) -> np.ndarray:
        """Run the NoMaD distance head."""
        with torch.no_grad():
            distances = self.model("dist_pred_net", obsgoal_cond=condition)
        return distances.detach().cpu().numpy().flatten()

    def unnormalize_action(self, deltas: np.ndarray) -> np.ndarray:
        """Map normalized diffusion output back to metric waypoint deltas."""
        normalized = (deltas + 1.0) / 2.0
        return normalized * (DEFAULT_ACTION_STATS["max"] - DEFAULT_ACTION_STATS["min"]) + DEFAULT_ACTION_STATS["min"]

    def sample_actions(
        self,
        condition: torch.Tensor,
        uncond_condition: torch.Tensor | None = None,
        num_samples: int | None = None,
    ) -> np.ndarray:
        """Run DDPM/DDIM sampling with optional CFG."""
        sample_count = int(num_samples or self.num_samples)
        scheduler = self._make_scheduler()
        condition_batch = condition.repeat(sample_count, 1)
        uncond_batch = None
        if uncond_condition is not None:
            uncond_batch = uncond_condition.repeat(sample_count, 1)

        action_noise = torch.randn(
            (sample_count, self.len_traj_pred, 2),
            device=self.device,
        )
        scheduler.set_timesteps(self.effective_num_steps())
        use_cfg = self.cfg_weight != 0.0 and uncond_batch is not None

        with torch.no_grad():
            for timestep in scheduler.timesteps:
                noise_pred = self.model(
                    "noise_pred_net",
                    sample=action_noise,
                    timestep=timestep,
                    global_cond=condition_batch,
                )
                if use_cfg:
                    noise_uncond = self.model(
                        "noise_pred_net",
                        sample=action_noise,
                        timestep=timestep,
                        global_cond=uncond_batch,
                    )
                    noise_pred = (1.0 + self.cfg_weight) * noise_pred - self.cfg_weight * noise_uncond
                action_noise = scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=action_noise,
                ).prev_sample

        action_deltas = self.unnormalize_action(action_noise.detach().cpu().numpy())
        return np.cumsum(action_deltas, axis=1)
