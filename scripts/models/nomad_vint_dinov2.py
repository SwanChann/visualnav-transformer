#!/usr/bin/env python3
# Allow direct execution from any scripts/<category>/ path.
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve()
while SCRIPTS_ROOT.name != "scripts" and SCRIPTS_ROOT.parent != SCRIPTS_ROOT:
    SCRIPTS_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
"""
Day 4: 将 EfficientNet-B0 替换为 DINOv2-Small 作为 NoMaD 视觉编码器。

核心变更:
  - obs_encoder / goal_encoder 从 EfficientNet-B0 切换到 DINOv2-Small
  - 新增 384->256 投影层
  - DINOv2 要求输入尺寸为 14 的倍数: 96->98 (14x7)

注意: 使用 timm 加载 DINOv2（而非 torch.hub）。
  torch.hub.load('facebookresearch/dinov2', ...) 会拉取最新仓库代码，
  其中含有 'float | None' 联合类型语法（Python 3.10+），
  在 Python 3.8 上会导致 TypeError。timm 的 DINOv2 实现完全兼容。
"""

import sys
import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tooling.project_paths import add_repo_paths

add_repo_paths()


class DINOv2Encoder(nn.Module):
    """DINOv2-Small 图像编码器，输出固定维度特征向量。

    使用 timm 而非 torch.hub 以避免 Python 3.8 不兼容问题。
    timm 模型: vit_small_patch14_dinov2.lvd142m
    """

    def __init__(self, output_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()
        # timm 加载 DINOv2-Small 预训练权重，不带分类头
        # num_classes=0      → forward() 返回 CLS token [B, 384]
        # dynamic_img_size=True → 禁用严格的 518×518 尺寸断言；
        #   位置编码在运行时插值以匹配输入尺寸
        #   （DINOv2 原生支持任何 patch_size=14 的倍数）
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,   # 允许 98×98 (= 14×7) 输入
        )
        self.feature_dim = 384              # DINOv2-Small CLS token 维度

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[DINOv2] Backbone frozen — only projection layers trainable")

        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] → [B, output_dim]"""
        # DINOv2 patch_size=14，输入必须是 14 的倍数
        # NoMaD 默认 96x96 -> 调整为 98x98 (14x7)
        if x.shape[-1] != 98 or x.shape[-2] != 98:
            x = F.interpolate(x, size=(98, 98), mode="bilinear",
                              align_corners=False)
        features  = self.backbone(x)           # [B, 384]  CLS token via timm
        projected = self.projection(features)  # [B, output_dim]
        return projected


class NoMaD_ViNT_DINOv2(nn.Module):
    """
    使用 DINOv2 替换 EfficientNet 的 NoMaD 视觉编码器。
    保持与原始 NoMaD_ViNT 相同的输入/输出接口。
    """

    def __init__(
        self,
        context_size: int = 3,
        obs_encoding_size: int = 256,
        mha_num_attention_heads: int = 4,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.context_size = context_size
        self.obs_encoding_size = obs_encoding_size

        # ── 图像编码器 ──
        self.obs_encoder  = DINOv2Encoder(obs_encoding_size, freeze_backbone)
        self.goal_encoder = DINOv2Encoder(obs_encoding_size, freeze_backbone)

        # 目标融合: concat(obs_feat, goal_feat) -> 融合特征
        self.goal_fusion = nn.Sequential(
            nn.Linear(obs_encoding_size * 2, obs_encoding_size),
            nn.GELU(),
        )

        # ── 时序自注意力（与原始结构相同）──
        from vint_train.models.vint.self_attention import PositionalEncoding

        self.positional_encoding = PositionalEncoding(
            obs_encoding_size, max_seq_len=context_size + 2,
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
            sa_layer, num_layers=mha_num_attention_layers,
        )

        # ── 目标遮罩（与原始结构相同）──
        self.goal_mask = torch.zeros((1, context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True
        self.no_mask   = torch.zeros((1, context_size + 2), dtype=torch.bool)
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat([
            1 - self.no_mask.float(),
            (1 - self.goal_mask.float())
            * ((context_size + 2) / (context_size + 1)),
        ], dim=0)

    def forward(self, obs_img, goal_img, input_goal_mask=None):
        device = obs_img.device

        # 将 [B, (C+1)*3, H, W] 拆分为 C+1 张图像
        obs_imgs_split = torch.split(obs_img, 3, dim=1)
        obs_encodings  = []
        for img in obs_imgs_split:
            enc = self.obs_encoder(img)             # [B, 256]
            obs_encodings.append(enc.unsqueeze(1))  # [B, 1, 256]
        obs_encoding = torch.cat(obs_encodings, dim=1)  # [B, C+1, 256]

        # 编码目标
        last_obs  = obs_imgs_split[-1]
        obs_feat  = self.obs_encoder(last_obs)      # [B, 256]
        goal_feat = self.goal_encoder(goal_img)     # [B, 256]
        goal_encoding = self.goal_fusion(
            torch.cat([obs_feat, goal_feat], dim=-1)
        ).unsqueeze(1)                              # [B, 1, 256]

        tokens = torch.cat([obs_encoding, goal_encoding], dim=1)  # [B, C+2, 256]

        # 目标遮罩
        if input_goal_mask is not None:
            idx = input_goal_mask.to(device=device, dtype=torch.bool).to(dtype=torch.long)
            src_key_padding_mask = torch.index_select(
                self.all_masks.to(device), 0, idx,
            ).to(dtype=torch.bool)
        else:
            src_key_padding_mask = None

        # 位置编码 + 自注意力
        tokens = self.positional_encoding(tokens)
        tokens = self.sa_encoder(
            tokens, src_key_padding_mask=src_key_padding_mask,
        )

        # 加权平均池化
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(
                self.avg_pool_mask.to(device), 0, idx,
            ).unsqueeze(-1)
            tokens = tokens * avg_mask

        return torch.mean(tokens, dim=1)            # [B, 256]
