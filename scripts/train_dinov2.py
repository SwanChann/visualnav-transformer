#!/usr/bin/env python3
"""
Day 4: NoMaD + DINOv2 训练脚本

独立训练脚本，将 EfficientNet-B0 替换为 DINOv2-Small 作为视觉编码器。
复用原始训练基础设施（ViNT_Dataset, train_eval_loop_nomad,
DDPMScheduler, ConditionalUnet1D）。

与原始 train.py 的主要区别:
  1. 视觉编码器: NoMaD_ViNT_DINOv2 (scripts/nomad_vint_dinov2.py)
  2. 图像尺寸: 98×98 (DINOv2 patch_size=14, 98=14×7)
  3. 默认冻结主干 → 仅训练投影层 + 扩散头
  4. 学习率: 5e-5 (冻结主干微调时降低)

用法:
  cd train
  python ../scripts/train_dinov2.py -c config/nomad_dinov2.yaml
"""

import os
import time
import argparse
from pathlib import Path

import yaml
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from project_paths import REPO_ROOT, add_repo_paths

# ── 项目设置 ──────────────────────────────────────────
PROJECT_ROOT = str(REPO_ROOT)
add_repo_paths()

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.training.train_eval_loop import train_eval_loop_nomad

# DINOv2 视觉编码器 (来自 scripts/nomad_vint_dinov2.py)
from nomad_vint_dinov2 import NoMaD_ViNT_DINOv2


def resolve_repo_relative(path_value: str) -> str:
    """Resolve a repo-relative path for runtime use without rewriting config files."""
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def validate_existing_path(path_value: str, label: str) -> str:
    """Resolve and validate that a required path exists before training starts."""
    resolved = resolve_repo_relative(path_value)
    if not Path(resolved).exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def build_dinov2_model(config: dict) -> tuple:
    """
    构建使用 DINOv2 视觉编码器的 NoMaD 模型。

    返回:
        model: NoMaD 模型
        noise_scheduler: DDPMScheduler
        trainable_params: 可训练参数数量
    """
    freeze = config.get("freeze_backbone", True)

    # 1) 视觉编码器 — DINOv2
    vision_encoder = NoMaD_ViNT_DINOv2(
        context_size=config["context_size"],
        obs_encoding_size=config["encoding_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        freeze_backbone=freeze,
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # 2) 噪声预测网络（与原始架构相同）
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config.get("cond_predict_scale", False),
    )

    # 3) 距离预测网络（与原始结构相同）
    dist_pred_net = DenseNetwork(embedding_dim=config["encoding_size"])

    # 4) 组装 NoMaD
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_net,
    )

    # 5) 噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # 统计可训练参数
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] Total params: {total:,}")
    print(f"[Model] Trainable params: {trainable:,} "
          f"({100*trainable/total:.1f}%)")
    if freeze:
        print("[Model] DINOv2 backbone FROZEN — only projection + heads trained")
    else:
        print("[Model] DINOv2 backbone UNFROZEN — full fine-tuning")

    return model, noise_scheduler, trainable


def build_dataloaders(config: dict):
    """根据配置构建训练和测试数据加载器。"""
    transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_datasets = []
    test_dataloaders = {}

    for ds_name, ds_cfg in config["datasets"].items():
        ds_cfg.setdefault("negative_mining", True)
        ds_cfg.setdefault("goals_per_obs", 1)
        ds_cfg.setdefault("end_slack", 0)
        ds_cfg.setdefault("waypoint_spacing", 1)

        for split in ["train", "test"]:
            if split not in ds_cfg:
                continue
            dataset = ViNT_Dataset(
                data_folder=validate_existing_path(ds_cfg["data_folder"], f"{ds_name} data_folder"),
                data_split_folder=validate_existing_path(ds_cfg[split], f"{ds_name} {split} split"),
                dataset_name=ds_name,
                image_size=config["image_size"],
                waypoint_spacing=ds_cfg["waypoint_spacing"],
                min_dist_cat=config["distance"]["min_dist_cat"],
                max_dist_cat=config["distance"]["max_dist_cat"],
                min_action_distance=config["action"]["min_dist_cat"],
                max_action_distance=config["action"]["max_dist_cat"],
                negative_mining=ds_cfg["negative_mining"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                context_size=config["context_size"],
                context_type=config.get("context_type", "temporal"),
                end_slack=ds_cfg["end_slack"],
                goals_per_obs=ds_cfg["goals_per_obs"],
                normalize=config["normalize"],
                goal_type=config.get("goal_type", "image"),
            )
            if split == "train":
                train_datasets.append(dataset)
            else:
                test_dataloaders[f"{ds_name}_test"] = dataset

    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
    )

    eval_bs = config.get("eval_batch_size", config["batch_size"])
    for key in test_dataloaders:
        test_dataloaders[key] = DataLoader(
            test_dataloaders[key],
            batch_size=eval_bs,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    print(f"[Data] Train samples: {len(train_dataset):,}")
    for k, v in test_dataloaders.items():
        print(f"[Data] Test '{k}': {len(v.dataset):,} samples")

    return train_loader, test_dataloaders, transform


def main(config: dict):
    """训练主入口。"""
    # ── 设备 ──
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpu_ids = config.get("gpu_ids", [0])
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        device = torch.device(f"cuda:{gpu_ids[0]}")
        print(f"[Device] CUDA: {torch.cuda.get_device_name(gpu_ids[0])}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU (no CUDA)")

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True
    cudnn.benchmark = True

    # ── 构建模型 ──
    model, noise_scheduler, n_trainable = build_dinov2_model(config)

    # ── 梯度裁剪 ──
    if config.get("clipping", False):
        max_norm = config.get("max_norm", 1.0)
        print(f"[Train] Gradient clipping: max_norm={max_norm}")
        for p in model.parameters():
            if p.requires_grad:
                p.register_hook(
                    lambda grad, mn=max_norm: torch.clamp(grad, -mn, mn)
                )

    # ── 优化器 ──
    lr = float(config["lr"])
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    print(f"[Train] Optimizer: AdamW, lr={lr}")

    # ── 学习率调度器 ──
    scheduler = None
    sched_name = config.get("scheduler")
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"],
        )
        print(f"[Train] Scheduler: CosineAnnealing, T_max={config['epochs']}")
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.get("plateau_factor", 0.5),
            patience=config.get("plateau_patience", 3),
        )

    if config.get("warmup", False):
        warmup_epochs = config.get("warmup_epochs", 4)
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1,
            total_epoch=warmup_epochs,
            after_scheduler=scheduler,
        )
        print(f"[Train] Warmup: {warmup_epochs} epochs")

    # ── 数据 ──
    train_loader, test_dataloaders, transform = build_dataloaders(config)

    # ── 多 GPU ──
    gpu_ids = config.get("gpu_ids", [0])
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    # ── 训练 ──
    print(f"\n{'='*60}")
    print(f"Starting DINOv2 Training: {config['epochs']} epochs, "
          f"batch_size={config['batch_size']}")
    print(f"{'='*60}\n")

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

    print("\n[DONE] DINOv2 training complete!")
    print(f"  Checkpoints saved to: {config['project_folder']}")
    print(f"  Best model: look for lowest test loss in wandb, use ema_<epoch>.pth")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="NoMaD + DINOv2 训练脚本",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/nomad_dinov2.yaml",
        type=str,
        help="YAML 配置文件路径",
    )
    args = parser.parse_args()

    # 加载默认配置 + 用户配置
    defaults_path = os.path.join(PROJECT_ROOT, "train/config/defaults.yaml")
    with open(defaults_path, "r") as f:
        config = yaml.safe_load(f)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = Path.cwd() / config_path
        if candidate.exists():
            config_path = candidate
        else:
            config_path = (REPO_ROOT / "train" / config_path).resolve()

    with config_path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    # 创建输出目录
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        PROJECT_ROOT, "train", "logs",
        config["project_name"], config["run_name"],
    )
    os.makedirs(config["project_folder"], exist_ok=True)

    # 保存配置快照
    cfg_path = os.path.join(config["project_folder"], "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved: {cfg_path}")

    # 初始化 wandb
    if config.get("use_wandb", False):
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.run.name = config["run_name"]
        wandb.config.update(config)
        wandb.save(str(config_path), policy="now")

    main(config)
