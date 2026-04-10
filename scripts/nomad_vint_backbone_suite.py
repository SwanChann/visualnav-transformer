#!/usr/bin/env python3
"""Backbone suite for NoMaD visual encoder experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from project_paths import add_repo_paths

add_repo_paths()

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
from vint_train.models.nomad.nomad_vint import replace_bn_with_gn
from vint_train.models.vint.self_attention import PositionalEncoding


@dataclass(frozen=True)
class BackboneSpec:
    """Single backbone specification."""

    timm_name: str
    input_size: int
    global_pool: str = "avg"
    dynamic_img_size: bool = False


BACKBONE_SPECS: Dict[str, BackboneSpec] = {
    "dinov2_small": BackboneSpec(
        timm_name="vit_small_patch14_dinov2.lvd142m",
        input_size=98,
        global_pool="token",
        dynamic_img_size=True,
    ),
    "convnext_tiny": BackboneSpec(
        timm_name="convnext_tiny.in12k_ft_in1k",
        input_size=96,
    ),
    "resnet50": BackboneSpec(
        timm_name="resnet50.a1_in1k",
        input_size=96,
    ),
}


def get_backbone_names() -> list[str]:
    """Return the supported backbone names."""
    return list(BACKBONE_SPECS.keys())


def get_backbone_input_size(backbone_name: str) -> int:
    """Return the expected input size for a backbone."""
    if backbone_name not in BACKBONE_SPECS:
        raise KeyError(f"Unsupported backbone: {backbone_name}")
    return BACKBONE_SPECS[backbone_name].input_size


class TimmBackboneEncoder(nn.Module):
    """Shared timm encoder wrapper."""

    def __init__(
        self,
        backbone_name: str,
        output_dim: int = 256,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        if backbone_name not in BACKBONE_SPECS:
            raise KeyError(f"Unsupported backbone: {backbone_name}")

        self.spec = BACKBONE_SPECS[backbone_name]
        create_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "global_pool": self.spec.global_pool,
        }
        if self.spec.dynamic_img_size:
            create_kwargs["dynamic_img_size"] = True

        self.backbone = timm.create_model(self.spec.timm_name, **create_kwargs)
        self.feature_dim = self._infer_feature_dim()

        if freeze_backbone:
            # 中文注释：默认冻结主干，只训练投影层和导航头
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def _infer_feature_dim(self) -> int:
        # 中文注释：通过一次虚拟前向自动推断 backbone 输出维度
        sample = torch.zeros(1, 3, self.spec.input_size, self.spec.input_size)
        with torch.no_grad():
            features = self.backbone(sample)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return int(features.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.spec.input_size or x.shape[-2] != self.spec.input_size:
            x = F.interpolate(
                x,
                size=(self.spec.input_size, self.spec.input_size),
                mode="bilinear",
                align_corners=False,
            )
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return self.projection(features)


class NoMaD_ViNT_BackboneSuite(nn.Module):
    """General NoMaD visual encoder with pluggable backbones."""

    def __init__(
        self,
        backbone_name: str,
        context_size: int = 3,
        obs_encoding_size: int = 256,
        mha_num_attention_heads: int = 4,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.context_size = context_size

        self.obs_encoder = TimmBackboneEncoder(
            backbone_name=backbone_name,
            output_dim=obs_encoding_size,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
        )
        self.goal_encoder = TimmBackboneEncoder(
            backbone_name=backbone_name,
            output_dim=obs_encoding_size,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
        )

        self.goal_fusion = nn.Sequential(
            nn.Linear(obs_encoding_size * 2, obs_encoding_size),
            nn.GELU(),
        )

        self.positional_encoding = PositionalEncoding(
            obs_encoding_size,
            max_seq_len=context_size + 2,
        )
        sa_layer = nn.TransformerEncoderLayer(
            d_model=obs_encoding_size,
            nhead=mha_num_attention_heads,
            dim_feedforward=mha_ff_dim_factor * obs_encoding_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_encoder = nn.TransformerEncoder(
            sa_layer,
            num_layers=mha_num_attention_layers,
        )

        # 中文注释：mask=0 表示导航模式，mask=1 表示隐藏目标图像
        self.goal_mask = torch.zeros((1, context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True
        self.no_mask = torch.zeros((1, context_size + 2), dtype=torch.bool)
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat(
            [
                1 - self.no_mask.float(),
                (1 - self.goal_mask.float()) * ((context_size + 2) / (context_size + 1)),
            ],
            dim=0,
        )

    def forward(self, obs_img: torch.Tensor, goal_img: torch.Tensor, input_goal_mask: torch.Tensor | None = None):
        device = obs_img.device

        obs_frames = torch.split(obs_img, 3, dim=1)
        obs_tokens = []
        for frame in obs_frames:
            obs_tokens.append(self.obs_encoder(frame).unsqueeze(1))
        obs_encoding = torch.cat(obs_tokens, dim=1)

        last_obs = obs_frames[-1]
        last_obs_feat = self.obs_encoder(last_obs)
        goal_feat = self.goal_encoder(goal_img)
        goal_encoding = self.goal_fusion(
            torch.cat([last_obs_feat, goal_feat], dim=-1)
        ).unsqueeze(1)

        tokens = torch.cat([obs_encoding, goal_encoding], dim=1)

        if input_goal_mask is not None:
            idx = input_goal_mask.long()
            src_key_padding_mask = torch.index_select(
                self.all_masks.to(device),
                0,
                idx,
            )
        else:
            idx = None
            src_key_padding_mask = None

        tokens = self.positional_encoding(tokens)
        tokens = self.sa_encoder(tokens, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None and idx is not None:
            avg_mask = torch.index_select(
                self.avg_pool_mask.to(device),
                0,
                idx,
            ).unsqueeze(-1)
            tokens = tokens * avg_mask

        return torch.mean(tokens, dim=1)


def build_backbone_nomad_model(
    config: dict,
    backbone_name: str,
    freeze_backbone: bool = True,
    pretrained_backbone: bool = True,
) -> NoMaD:
    """Build a NoMaD model with a selected timm backbone."""
    vision_encoder = NoMaD_ViNT_BackboneSuite(
        backbone_name=backbone_name,
        context_size=config["context_size"],
        obs_encoding_size=config["encoding_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        freeze_backbone=freeze_backbone,
        pretrained=pretrained_backbone,
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config.get("cond_predict_scale", False),
    )
    dist_pred_net = DenseNetwork(embedding_dim=config["encoding_size"])

    return NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_net,
    )
