#!/usr/bin/env python3
"""
Day 1: NoMaD 离线推理基线
在数据集图像上运行 DDPM 推理（无需 ROS），并可视化生成轨迹。
"""

import os
import sys
import glob
import time
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from torchvision import transforms
from project_paths import REPO_ROOT, add_repo_paths

# ── 运行时间戳（时间戳 + 脚本名）─────────────────────
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── 项目路径 ─────────────────────────────────────────
PROJECT_ROOT = str(REPO_ROOT)
add_repo_paths()

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# ── 配置参数 ────────────────────────────────────────
# 自训练权重：将 logs/ 中的 ema_N.pth 复制到此路径（见 §1.2）
MODEL_WEIGHTS = os.path.join(
    PROJECT_ROOT, "deployment/model_weights/nomad/nomad.pth"
)
DATASET_ROOT = os.path.join(PROJECT_ROOT, "nomad_dataset/go_stanford")

IMAGE_SIZE        = (96, 96)   # (宽, 高)
CONTEXT_SIZE      = 3          # 历史帧数
LEN_TRAJ_PRED     = 8          # 预测路径点数
NUM_DIFFUSION_ITERS = 10       # DDPM 去噪步数
NUM_SAMPLES       = 8          # 并行采样数
ENCODING_SIZE     = 256        # 视觉特征维度


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  模型构建 —— 对应 deployment/src/utils.py → load_model()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_model(device: torch.device) -> NoMaD:
    """构建并加载 NoMaD 模型，逻辑与 utils.py 中的 load_model 一致"""
    print("Building model...")

    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_WEIGHTS}. "
            "Copy a trained checkpoint to deployment/model_weights/nomad/nomad.pth first."
        )

    # 1) 视觉编码器
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=ENCODING_SIZE,
        context_size=CONTEXT_SIZE,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # 2) 噪声预测网络 (Conditional UNet1D)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,                     # 每个路径点为 (x, y)
        global_cond_dim=ENCODING_SIZE,   # 条件向量维度
        down_dims=[64, 128, 256],        # 来自 nomad.yaml
        cond_predict_scale=False,        # 来自 nomad.yaml
    )

    # 3) 距离预测网络
    dist_pred_net = DenseNetwork(embedding_dim=ENCODING_SIZE)

    # 4) 组装模型
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_net,
    )

    # 5) 加载预训练权重
    print(f"Loading weights: {MODEL_WEIGHTS}")
    state_dict = torch.load(MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print("Model loaded successfully!")
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  图像处理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _find_image(traj_path: str, idx: int) -> str:
    """在轨迹目录中查找第 idx 帧图像（支持 .jpg / .png）"""
    for ext in (".jpg", ".png", ".jpeg"):
        p = os.path.join(traj_path, f"{idx}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find frame {idx} image: {traj_path}")


def load_and_transform(image_path: str) -> torch.Tensor:
    """加载单张图像 → [3, 96, 96] 归一化张量"""
    img = PILImage.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    return _TRANSFORM(img)


def prepare_observation(traj_path: str, t: int) -> torch.Tensor:
    """
    准备观测输入：取 [t-CONTEXT_SIZE, ..., t] 共 CONTEXT_SIZE+1 帧，
    拼接为 [1, (CONTEXT_SIZE+1)*3, H, W]。
    """
    frames = []
    for i in range(max(0, t - CONTEXT_SIZE), t + 1):
        frames.append(load_and_transform(_find_image(traj_path, i)))

    # 帧数不足时用最早的帧填充
    while len(frames) < CONTEXT_SIZE + 1:
        frames.insert(0, frames[0].clone())

    frames = frames[-(CONTEXT_SIZE + 1):]      # 只保留最近 C+1 帧
    return torch.cat(frames, dim=0).unsqueeze(0)  # [1, 12, 96, 96]


def prepare_goal(traj_path: str, t: int) -> torch.Tensor:
    """准备目标图像 → [1, 3, 96, 96]"""
    return load_and_transform(_find_image(traj_path, t)).unsqueeze(0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DDPM 扩散推理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def diffusion_inference(
    model: NoMaD,
    obs_cond: torch.Tensor,   # [1, 256]
    device: torch.device,
    num_samples: int = NUM_SAMPLES,
    num_steps: int = NUM_DIFFUSION_ITERS,
) -> torch.Tensor:
    """
    通过 DDPM 反向去噪从高斯噪声生成路径点轨迹。
    返回: [num_samples, LEN_TRAJ_PRED, 2]
    """
    scheduler = DDPMScheduler(
        num_train_timesteps=num_steps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    cond = obs_cond.repeat(num_samples, 1)            # [N, 256]
    naction = torch.randn(
        (num_samples, LEN_TRAJ_PRED, 2), device=device
    )

    scheduler.set_timesteps(num_steps)
    with torch.no_grad():
        for k in scheduler.timesteps:
            noise_pred = model(
                "noise_pred_net",
                sample=naction,
                timestep=k,
                global_cond=cond,
            )
            naction = scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
            ).prev_sample

    return naction                                     # [N, 8, 2]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  可视化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def visualize_trajectories(
    obs_path: str,
    goal_path: str,
    trajectories: torch.Tensor,
    save_path: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(PILImage.open(obs_path))
    axes[0].set_title("Observation (current)", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(PILImage.open(goal_path))
    axes[1].set_title("Goal", fontsize=14)
    axes[1].axis("off")

    trajs = trajectories.cpu().numpy()
    for i in range(trajs.shape[0]):
        t = trajs[i]
        axes[2].plot(
            t[:, 1], t[:, 0], "o-", alpha=0.5, markersize=3,
            label=f"sample {i}" if i < 4 else None,
        )
    axes[2].plot(0, 0, "r*", markersize=15, label="robot")
    axes[2].set_title(f"Generated Trajectories (n={trajs.shape[0]})")
    axes[2].set_xlabel("y lateral (m)")
    axes[2].set_ylabel("x forward (m)")
    axes[2].legend(fontsize=8)
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved: {save_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.isdir(DATASET_ROOT):
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    # 1. 构建并加载模型
    model = build_model(device)

    # 2. 选择一条轨迹
    traj_names = sorted(os.listdir(DATASET_ROOT))
    traj_path  = os.path.join(DATASET_ROOT, traj_names[0])
    print(f"\nTest trajectory: {traj_names[0]}")

    images = sorted(glob.glob(os.path.join(traj_path, "*.jpg")))
    if not images:
        images = sorted(glob.glob(os.path.join(traj_path, "*.png")))
    n_images = len(images)
    print(f"Trajectory contains {n_images} frames")

    # 3. 设置观测/目标时刻
    obs_time  = CONTEXT_SIZE + 2
    goal_time = min(obs_time + 15, n_images - 1)
    print(f"Observation frame: t={obs_time}, Goal frame: t={goal_time}")

    # 4. 准备输入张量
    obs_img  = prepare_observation(traj_path, obs_time).to(device)
    goal_img = prepare_goal(traj_path, goal_time).to(device)

    # 5. 视觉编码（导航模式 mask=0）
    print("\n--- Vision Encoding ---")
    mask = torch.zeros(1).long().to(device)
    with torch.no_grad():
        obsgoal_cond = model(
            "vision_encoder",
            obs_img=obs_img,
            goal_img=goal_img,
            input_goal_mask=mask,
        )
    print(f"Condition vector shape: {obsgoal_cond.shape}")   # → [1, 256]

    # 6. 距离预测
    print("\n--- Distance Prediction ---")
    with torch.no_grad():
        dist = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
    print(f"Predicted distance: {dist.item():.4f}")

    # 7. DDPM 扩散推理
    print("\n--- Diffusion Inference (DDPM, 10 steps) ---")
    t0 = time.time()
    trajectories = diffusion_inference(model, obsgoal_cond, device)
    elapsed = time.time() - t0
    print(f"Inference time: {elapsed:.4f} s")
    print(f"Trajectory shape: {trajectories.shape}")       # → [8, 8, 2]

    # 8. 可视化（时间戳_脚本名 输出目录）
    result_dir = os.path.join(PROJECT_ROOT, f"results/day1/{RUN_TAG}_offline_inference")
    os.makedirs(result_dir, exist_ok=True)
    obs_img_path  = _find_image(traj_path, obs_time)
    goal_img_path = _find_image(traj_path, goal_time)

    visualize_trajectories(
        obs_img_path, goal_img_path, trajectories,
        os.path.join(result_dir, "baseline_trajectories.png"),
    )

    # 9. 探索模式对比 (mask=1)
    print("\n--- Exploration Mode (mask=1) ---")
    mask_explore = torch.ones(1).long().to(device)
    with torch.no_grad():
        explore_cond = model(
            "vision_encoder",
            obs_img=obs_img,
            goal_img=goal_img,
            input_goal_mask=mask_explore,
        )
    explore_trajs = diffusion_inference(model, explore_cond, device)
    visualize_trajectories(
        obs_img_path, goal_img_path, explore_trajs,
        os.path.join(result_dir, "explore_trajectories.png"),
    )

    # 10. 保存运行摘要到 txt
    summary_path = os.path.join(result_dir, "summary.txt")
    n_params = sum(p.numel() for p in model.parameters())
    lines = [
        f"Run time: {RUN_TAG}",
        f"Script: offline_inference.py",
        f"Test trajectory: {traj_names[0]}",
        f"Model params: {n_params:,}",
        f"DDPM inference time: {elapsed:.4f} s ({NUM_DIFFUSION_ITERS} steps)",
        f"Inference frequency: {1.0/elapsed:.1f} Hz",
        f"Generated trajectories: {NUM_SAMPLES} x {LEN_TRAJ_PRED} waypoints",
        f"Predicted distance: {dist.item():.4f}",
    ]
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary saved: {summary_path}")

    # 11. 打印摘要
    print("\n" + "=" * 60)
    print("Day 1 baseline verification complete!")
    for l in lines:
        print(f"  {l}")
    print(f"  Results directory: {result_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
