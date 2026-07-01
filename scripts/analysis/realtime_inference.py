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
Day 2.4: 实时摄像头推理
从 Ubuntu 系统摄像头采集画面，实时运行 NoMaD + DDIM 推理，
将预测的路径点轨迹叠加在画面上并保存为视频。

功能:
  1. USB 摄像头实时采集 (640x480 @ 30fps)
  2. 滑动窗口维护 context_size+1 帧历史
  3. DDIM-5 步快速推理（Day 2 验证的最佳配置）
  4. 实时显示轨迹叠加
  5. 按键控制: 'g' 捕捉目标帧, 'q' 退出, 's' 截图
  6. 完整录制 + 推理 FPS 统计

用法:
  python scripts/analysis/realtime_inference.py [--camera 0] [--ddim-steps 5]
"""

import os
import sys
import time
import argparse
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless rendering, avoid GUI conflicts
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image as PILImage
from tooling.project_paths import REPO_ROOT, add_repo_paths

# ── 运行时间戳 ───────────────────────────────────────────────
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── 项目路径 ─────────────────────────────────────────
PROJECT_ROOT = str(REPO_ROOT)
add_repo_paths()

from offline_inference import (
    build_model, ENCODING_SIZE, LEN_TRAJ_PRED, IMAGE_SIZE,
    CONTEXT_SIZE, NUM_DIFFUSION_ITERS,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  图像预处理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """OpenCV BGR 帧 -> NoMaD 输入张量 [3, 96, 96]"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb).resize(IMAGE_SIZE)
    return _TRANSFORM(pil)


def build_obs_tensor(frame_buffer: deque) -> torch.Tensor:
    """从滑动窗口构建 [1, (C+1)*3, 96, 96] 观测张量"""
    frames = list(frame_buffer)
    # 帧数不足时用最早的帧填充
    while len(frames) < CONTEXT_SIZE + 1:
        frames.insert(0, frames[0].clone())
    frames = frames[-(CONTEXT_SIZE + 1):]
    return torch.cat(frames, dim=0).unsqueeze(0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DDIM 推理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def realtime_diffusion(model, obs_cond, device, num_steps=5, num_samples=8):
    """DDIM 快速推理，返回 [num_samples, 8, 2] 轨迹"""
    scheduler = DDIMScheduler(
        num_train_timesteps=NUM_DIFFUSION_ITERS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    cond = obs_cond.repeat(num_samples, 1)
    naction = torch.randn((num_samples, LEN_TRAJ_PRED, 2), device=device)
    scheduler.set_timesteps(num_steps)
    with torch.no_grad():
        for k in scheduler.timesteps:
            noise_pred = model("noise_pred_net", sample=naction,
                               timestep=k, global_cond=cond)
            naction = scheduler.step(model_output=noise_pred,
                                     timestep=k, sample=naction).prev_sample
    return naction


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  轨迹绘制
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_trajectories(frame: np.ndarray, trajectories: torch.Tensor,
                      fps: float, mode: str) -> np.ndarray:
    """在帧上叠加路径点轨迹和状态信息"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    trajs = trajectories.cpu().numpy()
    center_x, center_y = w // 2, h - 30  # 底部中心作为机器人位置

    # 坐标映射: NoMaD 路径点 (x=前进, y=左) -> 像素 (u=右, v=上)
    scale = min(w, h) / 2.0  # 缩放因子

    colors = [
        (52, 152, 219), (46, 204, 113), (155, 89, 182),
        (241, 196, 15), (230, 126, 34), (231, 76, 60),
        (26, 188, 156), (149, 165, 166),
    ]

    # 绘制平均轨迹（粗线）
    mean_traj = trajs.mean(axis=0)
    pts_mean = []
    for wp in mean_traj:
        px = int(center_x + wp[1] * scale)  # y->right
        py = int(center_y - wp[0] * scale)  # x->up
        pts_mean.append((px, py))
    for i in range(len(pts_mean) - 1):
        cv2.line(overlay, pts_mean[i], pts_mean[i+1], (0, 255, 0), 3)
    for pt in pts_mean:
        cv2.circle(overlay, pt, 5, (0, 255, 0), -1)

    # 绘制单个采样轨迹（细线，半透明）
    for s in range(min(trajs.shape[0], 4)):
        pts = []
        for wp in trajs[s]:
            px = int(center_x + wp[1] * scale)
            py = int(center_y - wp[0] * scale)
            pts.append((px, py))
        color = colors[s % len(colors)]
        for i in range(len(pts) - 1):
            cv2.line(overlay, pts[i], pts[i+1], color, 1)

    # 机器人位置标记
    cv2.circle(overlay, (center_x, center_y), 8, (0, 0, 255), -1)
    cv2.putText(overlay, "R", (center_x - 5, center_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 屏幕信息叠加
    cv2.putText(overlay, f"Mode: {mode}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(overlay, f"DDIM Steps: 5", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(overlay, "[G]oal  [S]creenshot  [Q]uit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  主循环
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(description="NoMaD Realtime Camera Inference")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default /dev/video0)")
    parser.add_argument("--ddim-steps", type=int, default=5,
                        help="DDIM denoising steps (default 5)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode (for SSH without GUI)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 加载模型
    model = build_model(device)

    # 2. 打开摄像头
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera /dev/video{args.camera}")
        print("Hint: Check camera connection, or try --camera 1")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {actual_w}x{actual_h}")

    # 3. 输出目录
    out_dir = os.path.join(PROJECT_ROOT, f"results/realtime/{RUN_TAG}_realtime")
    os.makedirs(out_dir, exist_ok=True)

    # 4. 视频录制器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(out_dir, "recording.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, 15.0,
                             (actual_w, actual_h))

    # 5. 初始化
    frame_buffer = deque(maxlen=CONTEXT_SIZE + 1)
    goal_frame = None       # 目标帧（按 G 设置）
    goal_tensor = None
    mode = "Explore (mask=1)"
    screenshot_count = 0
    fps_list = []

    print("\n" + "=" * 50)
    print("Realtime inference started!")
    print("  [G] Grab current frame as goal (switch to navigation mode)")
    print("  [S] Save screenshot")
    print("  [Q] Quit")
    print("=" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            t0 = time.time()

            # 预处理当前帧
            tensor = frame_to_tensor(frame)
            frame_buffer.append(tensor)

            # 等待至少 context_size+1 帧
            if len(frame_buffer) < CONTEXT_SIZE + 1:
                cv2.putText(frame, f"Buffering... {len(frame_buffer)}/{CONTEXT_SIZE+1}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if not args.no_display:
                    cv2.imshow("NoMaD Realtime", frame)
                    cv2.waitKey(1)
                continue

            # 构建输入
            obs_tensor = build_obs_tensor(frame_buffer).to(device)

            if goal_tensor is not None:
                # 导航模式
                goal_input = goal_tensor.to(device)
                mask = torch.zeros(1).long().to(device)
                mode = "Navigate (mask=0)"
            else:
                # 探索模式 — 用当前帧作为虚拟目标
                goal_input = tensor.unsqueeze(0).to(device)
                mask = torch.ones(1).long().to(device)
                mode = "Explore (mask=1)"

            # 视觉编码
            with torch.no_grad():
                obs_cond = model("vision_encoder",
                                 obs_img=obs_tensor,
                                 goal_img=goal_input,
                                 input_goal_mask=mask)

            # DDIM 推理
            trajectories = realtime_diffusion(
                model, obs_cond, device,
                num_steps=args.ddim_steps,
            )

            elapsed = time.time() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_list.append(fps)

            # 在帧上叠加轨迹
            display_frame = draw_trajectories(frame, trajectories, fps, mode)

            # 显示目标帧缩略图
            if goal_frame is not None:
                thumb = cv2.resize(goal_frame, (120, 90))
                display_frame[5:95, actual_w-125:actual_w-5] = thumb
                cv2.putText(display_frame, "GOAL", (actual_w-120, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 写入录制
            writer.write(display_frame)

            # 显示
            if not args.no_display:
                cv2.imshow("NoMaD Realtime", display_frame)

            # 键盘事件
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nExiting realtime inference")
                break
            elif key == ord("g"):
                goal_frame = frame.copy()
                goal_tensor = tensor.unsqueeze(0)
                print(f"  -> Goal frame set (switching to navigation mode)")
                # 保存目标帧
                cv2.imwrite(os.path.join(out_dir, "goal_frame.jpg"), goal_frame)
            elif key == ord("s"):
                screenshot_count += 1
                ss_path = os.path.join(out_dir,
                    f"screenshot_{screenshot_count:03d}.jpg")
                cv2.imwrite(ss_path, display_frame)
                print(f"  -> Screenshot saved: {ss_path}")

    finally:
        cap.release()
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # ── 保存统计摘要 ──────────────────────
    if fps_list:
        summary_path = os.path.join(out_dir, "summary.txt")
        avg_fps = np.mean(fps_list)
        with open(summary_path, "w") as f:
            f.write(f"Run Time: {RUN_TAG}\n")
            f.write(f"Script: realtime_inference.py\n")
            f.write(f"Camera: /dev/video{args.camera}\n")
            f.write(f"Resolution: {actual_w}x{actual_h}\n")
            f.write(f"DDIM Steps: {args.ddim_steps}\n")
            f.write(f"Total Frames: {len(fps_list)}\n")
            f.write(f"Avg Inference FPS: {avg_fps:.1f}\n")
            f.write(f"Min FPS: {min(fps_list):.1f}\n")
            f.write(f"Max FPS: {max(fps_list):.1f}\n")
            f.write(f"Recording: {video_path}\n")
            f.write(f"Screenshots: {screenshot_count}\n")
            f.write(f"\nLegged Deployment Evaluation:\n")
            if avg_fps >= 15:
                f.write(f"  ✅ Avg {avg_fps:.1f} Hz meets legged navigation requirement (≥15 Hz)\n")
            else:
                f.write(f"  ⚠️ Avg {avg_fps:.1f} Hz below legged threshold 15 Hz, "
                        f"reduce DDIM steps or lower sample count\n")

        print(f"\n{'='*50}")
        print(f"Realtime inference statistics:")
        print(f"  Avg FPS: {avg_fps:.1f}")
        print(f"  Recording saved: {video_path}")
        print(f"  Summary saved: {summary_path}")
        print(f"{'='*50}")

    # ── 生成 FPS 趋势图 ──────────────────────
    if len(fps_list) > 10:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(fps_list, color="#3498db", alpha=0.6, linewidth=0.5, label="instant FPS")
        # 移动平均
        window = min(30, len(fps_list) // 3)
        if window > 1:
            ma = np.convolve(fps_list, np.ones(window)/window, mode="valid")
            ax.plot(range(window-1, len(fps_list)), ma,
                    color="#e74c3c", linewidth=2, label=f"MA-{window}")
        ax.axhline(y=15, color="orange", linestyle="--", label="Legged threshold 15Hz")
        ax.axhline(y=10, color="red", linestyle=":", alpha=0.5, label="Wheeled baseline 10Hz")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("FPS")
        ax.set_title("Realtime Inference FPS Trend")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fps_trend.png"), dpi=150)
        plt.close()
        print(f"  FPS trend: {out_dir}/fps_trend.png")


if __name__ == "__main__":
    main()
