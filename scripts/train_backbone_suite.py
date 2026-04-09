#!/usr/bin/env python3
"""Generic backbone training entry for NoMaD."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb
import yaml
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from project_paths import REPO_ROOT, add_repo_paths

PROJECT_ROOT = str(REPO_ROOT)
add_repo_paths()

from nomad_vint_backbone_suite import (
    build_backbone_nomad_model,
    get_backbone_input_size,
    get_backbone_names,
)
from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.training.train_eval_loop import train_eval_loop_nomad


def resolve_repo_relative(path_value: str) -> str:
    """Resolve a repo-relative path."""
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def validate_existing_path(path_value: str, label: str) -> str:
    """Resolve and validate a required path."""
    resolved = resolve_repo_relative(path_value)
    if not Path(resolved).exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def build_noise_scheduler(config: dict) -> DDPMScheduler:
    """Create the DDPM scheduler used by NoMaD training."""
    return DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )


def build_dataloaders(config: dict):
    """Construct train and test dataloaders."""
    transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_datasets = []
    test_dataloaders = {}

    for dataset_name, dataset_cfg in config["datasets"].items():
        dataset_cfg.setdefault("negative_mining", True)
        dataset_cfg.setdefault("goals_per_obs", 1)
        dataset_cfg.setdefault("end_slack", 0)
        dataset_cfg.setdefault("waypoint_spacing", 1)

        for split_name in ("train", "test"):
            if split_name not in dataset_cfg:
                continue
            dataset = ViNT_Dataset(
                data_folder=validate_existing_path(dataset_cfg["data_folder"], f"{dataset_name} data_folder"),
                data_split_folder=validate_existing_path(dataset_cfg[split_name], f"{dataset_name} {split_name} split"),
                dataset_name=dataset_name,
                image_size=config["image_size"],
                waypoint_spacing=dataset_cfg["waypoint_spacing"],
                min_dist_cat=config["distance"]["min_dist_cat"],
                max_dist_cat=config["distance"]["max_dist_cat"],
                min_action_distance=config["action"]["min_dist_cat"],
                max_action_distance=config["action"]["max_dist_cat"],
                negative_mining=dataset_cfg["negative_mining"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                context_size=config["context_size"],
                context_type=config.get("context_type", "temporal"),
                end_slack=dataset_cfg["end_slack"],
                goals_per_obs=dataset_cfg["goals_per_obs"],
                normalize=config["normalize"],
                goal_type=config.get("goal_type", "image"),
            )
            if split_name == "train":
                train_datasets.append(dataset)
            else:
                test_dataloaders[f"{dataset_name}_test"] = dataset

    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
    )

    eval_batch_size = config.get("eval_batch_size", config["batch_size"])
    for key in list(test_dataloaders.keys()):
        test_dataloaders[key] = DataLoader(
            test_dataloaders[key],
            batch_size=eval_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    return train_loader, test_dataloaders, transform


def update_config_for_backbone(config: dict, backbone_name: str, freeze_backbone: bool) -> dict:
    """Inject backbone-specific settings into the config."""
    config = dict(config)
    config["project_name"] = "nomad-backbone-suite"
    config["run_name"] = f"nomad_{backbone_name}"
    config["vision_encoder"] = "nomad_vint_backbone_suite"
    config["encoder_backbone"] = backbone_name
    config["freeze_backbone"] = freeze_backbone
    size = get_backbone_input_size(backbone_name)
    config["image_size"] = [size, size]
    if "encoding_size" not in config:
        config["encoding_size"] = 256
    return config


def main(config: dict, backbone_name: str, freeze_backbone: bool, pretrained_backbone: bool) -> None:
    """Training entry."""
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpu_ids = config.get("gpu_ids", [0])
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True
    cudnn.benchmark = True

    model = build_backbone_nomad_model(
        config=config,
        backbone_name=backbone_name,
        freeze_backbone=freeze_backbone,
        pretrained_backbone=pretrained_backbone,
    )
    noise_scheduler = build_noise_scheduler(config)

    optimizer = AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=float(config["lr"]),
    )

    scheduler = None
    if config.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"],
        )
    elif config.get("scheduler") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.get("plateau_factor", 0.5),
            patience=config.get("plateau_patience", 3),
        )
    if config.get("warmup", False):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=config.get("warmup_epochs", 2),
            after_scheduler=scheduler,
        )

    train_loader, test_dataloaders, transform = build_dataloaders(config)

    gpu_ids = config.get("gpu_ids", [0])
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    train_eval_loop_nomad(
        train_model=config["train"],
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        noise_scheduler=noise_scheduler,
        train_loader=train_loader,
        test_dataloaders=test_dataloaders,
        transform=transform,
        goal_mask_prob=config["goal_mask_prob"],
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        print_log_freq=config["print_log_freq"],
        wandb_log_freq=config["wandb_log_freq"],
        image_log_freq=config["image_log_freq"],
        num_images_log=config["num_images_log"],
        current_epoch=0,
        alpha=float(config["alpha"]),
        use_wandb=config["use_wandb"],
        eval_fraction=config["eval_fraction"],
        eval_freq=config["eval_freq"],
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Generic NoMaD backbone training")
    parser.add_argument(
        "--config",
        "-c",
        default="config/nomad_dinov2.yaml",
        help="Training config path",
    )
    parser.add_argument(
        "--backbone",
        choices=get_backbone_names(),
        required=True,
        help="Backbone name to train",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the backbone and only train projection plus navigation heads",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Load pretrained timm weights for the selected backbone",
    )
    args = parser.parse_args()

    defaults_path = Path(PROJECT_ROOT) / "train" / "config" / "defaults.yaml"
    with defaults_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / "train" / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle)
    config.update(user_config)
    config = update_config_for_backbone(
        config=config,
        backbone_name=args.backbone,
        freeze_backbone=args.freeze_backbone,
    )

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        PROJECT_ROOT,
        "train",
        "logs",
        config["project_name"],
        config["run_name"],
    )
    os.makedirs(config["project_folder"], exist_ok=True)

    config_snapshot = Path(config["project_folder"]) / "config.yaml"
    with config_snapshot.open("w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)

    if config.get("use_wandb", False):
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.run.name = config["run_name"]
        wandb.config.update(config)
        wandb.save(str(config_path), policy="now")

    main(
        config=config,
        backbone_name=args.backbone,
        freeze_backbone=args.freeze_backbone,
        pretrained_backbone=args.pretrained_backbone,
    )
