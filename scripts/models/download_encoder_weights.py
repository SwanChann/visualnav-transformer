#!/usr/bin/env python3
"""Download pretrained timm weights for visual encoder experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List


BACKBONE_TIMM_NAMES: Dict[str, str] = {
    "efficientnet_b0_suite": "efficientnet_b0",
    "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
    "convnext_tiny": "convnext_tiny.in12k_ft_in1k",
    "resnet50": "resnet50.a1_in1k",
}

BACKBONE_INPUT_SIZES: Dict[str, int] = {
    "efficientnet_b0_suite": 96,
    "dinov2_small": 98,
    "convnext_tiny": 96,
    "resnet50": 96,
}

BACKBONE_GLOBAL_POOLS: Dict[str, str] = {
    "efficientnet_b0_suite": "avg",
    "dinov2_small": "token",
    "convnext_tiny": "avg",
    "resnet50": "avg",
}


def parse_backbones(raw_backbones: List[str] | None) -> List[str]:
    """Parse requested backbone names."""
    if raw_backbones is None:
        return list(BACKBONE_TIMM_NAMES.keys())
    if "all" in raw_backbones:
        return list(BACKBONE_TIMM_NAMES.keys())
    return raw_backbones


def configure_cache(cache_dir: str | None) -> None:
    """Configure local cache directories."""
    if cache_dir is None:
        return

    cache_path = Path(cache_dir).expanduser().resolve()
    torch_cache = cache_path / "torch"
    huggingface_cache = cache_path / "huggingface"
    torch_cache.mkdir(parents=True, exist_ok=True)
    huggingface_cache.mkdir(parents=True, exist_ok=True)

    # 中文注释：timm 可能通过 torch hub 或 huggingface hub 下载权重
    os.environ["TORCH_HOME"] = str(torch_cache)
    os.environ["HF_HOME"] = str(huggingface_cache)


def create_timm_model(backbone_name: str):
    """Create one pretrained timm model."""
    import timm

    create_kwargs = {
        "pretrained": True,
        "num_classes": 0,
        "global_pool": BACKBONE_GLOBAL_POOLS[backbone_name],
    }
    if backbone_name == "dinov2_small":
        # 中文注释：DINOv2 使用 token pooling，避免预训练权重的 norm 键和 fc_norm 键不匹配
        create_kwargs["dynamic_img_size"] = True

    return timm.create_model(BACKBONE_TIMM_NAMES[backbone_name], **create_kwargs)


def verify_forward(backbone_name: str, model) -> None:
    """Run a tiny forward pass to verify the cached weights."""
    import torch

    input_size = BACKBONE_INPUT_SIZES[backbone_name]
    model.eval()
    with torch.no_grad():
        # 中文注释：只用零张量检查网络结构和权重是否能正常前向传播
        sample = torch.zeros(1, 3, input_size, input_size)
        output = model(sample)
    print(f"{backbone_name}: forward_shape={tuple(output.shape)}")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Download visual encoder pretrained weights")
    parser.add_argument(
        "--backbone",
        action="append",
        default=None,
        choices=["all", *BACKBONE_TIMM_NAMES.keys()],
        help="Backbone to download, can be repeated",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache root, for example ~/.cache/nomad_encoder",
    )
    parser.add_argument(
        "--verify-forward",
        action="store_true",
        help="Run a small CPU forward pass after downloading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the timm model names",
    )
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    backbones = parse_backbones(args.backbone)

    for backbone_name in backbones:
        timm_name = BACKBONE_TIMM_NAMES[backbone_name]
        if args.dry_run:
            print(f"{backbone_name}: {timm_name}")
            continue

        print(f"Downloading {backbone_name}: {timm_name}")
        model = create_timm_model(backbone_name)
        if args.verify_forward:
            verify_forward(backbone_name, model)

    print("Finished visual encoder weight preparation.")


if __name__ == "__main__":
    main()
