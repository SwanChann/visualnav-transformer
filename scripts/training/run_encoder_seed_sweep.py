#!/usr/bin/env python3
"""Run visual encoder training with multiple random seeds."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import yaml


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_ROOT.parent


def resolve_path(path_value: str) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def parse_seeds(raw_seeds: List[str]) -> List[int]:
    """Parse seed values."""
    seeds: List[int] = []
    for raw_seed in raw_seeds:
        for part in raw_seed.split(","):
            part = part.strip()
            if part:
                seeds.append(int(part))
    return seeds


def write_seed_config(base_config_path: Path, output_dir: Path, backbone: str, seed: int) -> Path:
    """Create a seed-specific config file."""
    with base_config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config = copy.deepcopy(config)
    config["seed"] = seed
    config["encoder_backbone"] = backbone
    config["run_name"] = f"{config.get('run_name', backbone)}_seed_{seed}"

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"{backbone}_seed_{seed}.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        # 中文注释：为每个随机种子生成独立配置，便于复现实验
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return config_path


def make_seed_config_path(output_dir: Path, backbone: str, seed: int) -> Path:
    """Return the seed-specific config path."""
    return output_dir / f"{backbone}_seed_{seed}.yaml"


def build_train_command(args, config_path: Path) -> List[str]:
    """Build one training command."""
    train_entry = SCRIPTS_ROOT / "training" / "train_backbone_suite.py"
    command = [
        sys.executable,
        str(train_entry),
        "--config",
        str(config_path),
        "--backbone",
        args.backbone,
    ]
    if args.freeze_backbone:
        command.append("--freeze-backbone")
    if args.pretrained_backbone:
        command.append("--pretrained-backbone")
    return command


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run NoMaD encoder training across seeds")
    parser.add_argument("--base-config", required=True, help="Base YAML config path")
    parser.add_argument(
        "--backbone",
        required=True,
        choices=["dinov2_small", "convnext_tiny", "resnet50"],
        help="Backbone name",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0", "1", "2"],
        help="Seed list, for example 0 1 2 or 0,1,2",
    )
    parser.add_argument(
        "--output-dir",
        default="results/config_runs/vision_encoder",
        help="Directory for generated seed configs",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the visual backbone during training",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Load pretrained timm weights",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running training",
    )
    args = parser.parse_args()

    base_config_path = resolve_path(args.base_config)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_path(args.output_dir) / f"{args.backbone}_{timestamp}"
    seeds = parse_seeds(args.seeds)

    for seed in seeds:
        if args.dry_run:
            # 中文注释：dry-run 只预览路径和命令，不写入临时配置文件
            config_path = make_seed_config_path(output_dir, args.backbone, seed)
        else:
            config_path = write_seed_config(base_config_path, output_dir, args.backbone, seed)
        command = build_train_command(args, config_path)
        print(" ".join(command))
        if not args.dry_run:
            # 中文注释：逐个种子串行运行，避免多个训练任务争抢同一块 GPU
            subprocess.run(command, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
