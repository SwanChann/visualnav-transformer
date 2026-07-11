"""Feature-level TinyNavBrain scaffold for interface and shape validation.

This module deliberately starts from visual features.  It does not claim to be
the complete image encoder or a trained navigation policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TinyNavConfig:
    encoder_dim: int = 1280
    d_model: int = 256
    context_size: int = 5
    action_history_length: int = 4
    action_horizon: int = 8
    waypoint_dim: int = 2
    transformer_layers: int = 4
    transformer_heads: int = 4
    transformer_ffn_dim: int = 1024
    dropout: float = 0.1
    max_trainable_parameters: int = 20_000_000

    @property
    def observation_frames(self) -> int:
        return self.context_size + 1

    @property
    def token_count(self) -> int:
        return 1 + self.observation_frames + 1 + self.action_history_length + 1


class TinyNavBrainScaffold(nn.Module):
    ALLOWED_NFE = {1, 2, 4, 8}

    def __init__(self, config: TinyNavConfig = TinyNavConfig()) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.image_projection = nn.Sequential(nn.Linear(config.encoder_dim, d), nn.LayerNorm(d))
        self.goal_relation = nn.Sequential(
            nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d)
        )
        self.action_projection = nn.Sequential(
            nn.Linear(3, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d)
        )
        self.scale_projection = nn.Sequential(
            nn.Linear(2, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.null_goal = nn.Parameter(torch.zeros(1, d))
        self.position = nn.Parameter(torch.zeros(1, config.token_count, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=config.transformer_layers)
        self.fusion_norm = nn.LayerNorm(d)

        action_dim = config.action_horizon * config.waypoint_dim
        self.deterministic_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, action_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64)
        )
        self.flow_head = nn.Sequential(
            nn.Linear(d + 64 + action_dim, 2 * d),
            nn.SiLU(),
            nn.Linear(2 * d, 2 * d),
            nn.SiLU(),
            nn.Linear(2 * d, action_dim),
        )
        self.progress_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.null_goal, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def _validate_inputs(
        self,
        obs_features: Tensor,
        goal_features: Tensor,
        goal_mask: Tensor,
        action_history: Tensor,
        action_history_mask: Tensor,
        dt_s: Tensor,
        waypoint_spacing_m: Tensor,
    ) -> None:
        cfg = self.config
        batch = obs_features.shape[0]
        expected = (batch, cfg.observation_frames, cfg.encoder_dim)
        if tuple(obs_features.shape) != expected:
            raise ValueError(f"obs_features shape {tuple(obs_features.shape)} != {expected}")
        if tuple(goal_features.shape) != (batch, cfg.encoder_dim):
            raise ValueError("goal_features has invalid shape")
        if tuple(goal_mask.shape) != (batch,) or goal_mask.dtype != torch.bool:
            raise ValueError("goal_mask must be bool [B]")
        if tuple(action_history.shape) != (batch, cfg.action_history_length, 3):
            raise ValueError("action_history has invalid shape")
        if tuple(action_history_mask.shape) != (batch, cfg.action_history_length):
            raise ValueError("action_history_mask has invalid shape")
        if tuple(dt_s.shape) != (batch, 1) or tuple(waypoint_spacing_m.shape) != (batch, 1):
            raise ValueError("dt_s and waypoint_spacing_m must be [B,1]")
        if bool(torch.any(dt_s <= 0)) or bool(torch.any(waypoint_spacing_m <= 0)):
            raise ValueError("physical scale metadata must be positive")

    def encode(
        self,
        obs_features: Tensor,
        goal_features: Tensor,
        goal_mask: Tensor,
        action_history: Tensor,
        action_history_mask: Tensor,
        dt_s: Tensor,
        waypoint_spacing_m: Tensor,
    ) -> Tensor:
        self._validate_inputs(
            obs_features,
            goal_features,
            goal_mask,
            action_history,
            action_history_mask,
            dt_s,
            waypoint_spacing_m,
        )
        batch = obs_features.shape[0]
        obs_tokens = self.image_projection(obs_features)
        goal_token = self.image_projection(goal_features)
        effective_goal = torch.where(goal_mask[:, None], self.null_goal.expand(batch, -1), goal_token)
        current = obs_tokens[:, -1]
        relation = self.goal_relation(
            torch.cat([current, effective_goal, current - effective_goal, current * effective_goal], dim=-1)
        ).unsqueeze(1)

        action_tokens = self.action_projection(action_history)
        action_tokens = action_tokens * action_history_mask.to(action_tokens.dtype).unsqueeze(-1)
        scale = torch.cat([torch.log(dt_s), torch.log(waypoint_spacing_m)], dim=-1)
        scale_token = self.scale_projection(scale).unsqueeze(1)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, obs_tokens, relation, action_tokens, scale_token], dim=1)
        if tokens.shape[1] != self.config.token_count:
            raise RuntimeError("internal token count does not match frozen contract")
        fused = self.fusion(tokens + self.position[:, : tokens.shape[1]])
        return self.fusion_norm(fused[:, 0])

    def predict_deterministic(self, context: Tensor) -> Tensor:
        cfg = self.config
        flat = self.deterministic_head(context)
        return flat.reshape(context.shape[0], cfg.action_horizon, cfg.waypoint_dim)

    def flow_velocity(self, x_t: Tensor, t: Tensor, context: Tensor) -> Tensor:
        cfg = self.config
        expected = (context.shape[0], cfg.action_horizon, cfg.waypoint_dim)
        if tuple(x_t.shape) != expected:
            raise ValueError(f"x_t shape {tuple(x_t.shape)} != {expected}")
        if tuple(t.shape) != (context.shape[0], 1):
            raise ValueError("t must be [B,1]")
        flat = x_t.reshape(context.shape[0], -1)
        velocity = self.flow_head(torch.cat([context, self.time_embedding(t), flat], dim=-1))
        return velocity.reshape(expected)

    @torch.no_grad()
    def sample_flow(
        self,
        context: Tensor,
        nfe: int = 2,
        candidate_count: int = 1,
        seed: Optional[int] = None,
    ) -> Tensor:
        if nfe not in self.ALLOWED_NFE:
            raise ValueError(f"nfe must be one of {sorted(self.ALLOWED_NFE)}")
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        cfg = self.config
        repeated = context.repeat_interleave(candidate_count, dim=0)
        generator = torch.Generator(device=context.device)
        if seed is not None:
            generator.manual_seed(seed)
        x = torch.randn(
            repeated.shape[0], cfg.action_horizon, cfg.waypoint_dim,
            device=context.device, dtype=context.dtype, generator=generator
        )
        step = 1.0 / nfe
        for index in range(nfe):
            time_value = 1.0 - index * step
            t = torch.full((repeated.shape[0], 1), time_value, device=context.device, dtype=context.dtype)
            x = x - step * self.flow_velocity(x, t, repeated)
        return x.reshape(context.shape[0], candidate_count, cfg.action_horizon, cfg.waypoint_dim)

    def predict_progress(self, context: Tensor) -> Tensor:
        return self.progress_head(context)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
