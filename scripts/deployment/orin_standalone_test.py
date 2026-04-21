#!/usr/bin/env python3
"""Orin standalone diagnostic script.

在 Orin 上独立运行，不需要连接 Lite3 运动主机。
用于验证：
  1. 相机是否正常工作
  2. NoMaD 模型是否能正常加载和推理
  3. DDIM 加速是否正常
  4. 端到端推理延迟测量

使用方式：
  # 步骤一：仅测试相机
  python scripts/deployment/orin_standalone_test.py --test camera

  # 步骤二：测试模型加载
  python scripts/deployment/orin_standalone_test.py --test model \
    --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
    --policy-checkpoint deployment/model_weights/nomad/nomad.pth

  # 步骤三：端到端推理基准测试
  python scripts/deployment/orin_standalone_test.py --test benchmark \
    --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
    --policy-checkpoint deployment/model_weights/nomad/nomad.pth

  # 步骤四：完整流水线测试（相机 + 推理 + 可视化）
  python scripts/deployment/orin_standalone_test.py --test pipeline \
    --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
    --policy-checkpoint deployment/model_weights/nomad/nomad.pth
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for p in (SCRIPTS_ROOT, SCRIPTS_ROOT / "shared", SCRIPTS_ROOT / "simulation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def test_camera(args):
    """测试相机读取。"""
    print("=" * 60)
    print("测试一：相机读取")
    print("=" * 60)

    from deployment.lite3_real_bridge import OrinCamera

    try:
        cam = OrinCamera(
            device=args.camera_device,
            width=args.camera_width,
            height=args.camera_height,
            use_csi=args.use_csi,
            calibration_path=args.camera_calibration_path,
            undistort_alpha=args.undistort_alpha,
        )
    except RuntimeError as e:
        print(f"❌ 相机打开失败: {e}")
        return False

    latencies = []
    for i in range(20):
        t0 = time.time()
        img = cam.read()
        dt = (time.time() - t0) * 1000
        latencies.append(dt)
        if i < 3:
            print(f"  帧 {i}: size={img.size}, latency={dt:.1f}ms")

    cam.close()
    avg = np.mean(latencies)
    print(f"\n  ✅ 相机正常 | 平均延迟: {avg:.1f}ms | FPS: {1000/avg:.1f}")
    return True


def test_model(args):
    """测试模型加载。"""
    print("=" * 60)
    print("测试二：模型加载")
    print("=" * 60)

    if not args.policy_config or not args.policy_checkpoint:
        print("❌ 需要 --policy-config 和 --policy-checkpoint")
        return False

    import torch
    from shared.nomad_inference import NoMaDInferenceModule, NoMaDInferenceSpec

    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print("  Note: On Jetson Orin this CUDA device is the integrated NVIDIA GPU, not a desktop RTX card.")
    else:
        print("  Warning: Orin can still fall back to CPU, but this is usually not suitable for real-time deployment.")
        print("  Warning: On Jetson Orin, 'CUDA available: False' usually means a JetPack / PyTorch mismatch, not missing hardware.")

    t0 = time.time()
    spec = NoMaDInferenceSpec(
        policy_config=args.policy_config,
        policy_checkpoint=args.policy_checkpoint,
        scheduler_kind="ddim",
        ddim_steps=args.ddim_steps,
        image_resize_mode=args.image_resize_mode,
    )
    inference = NoMaDInferenceModule(spec)
    load_time = time.time() - t0

    print(f"\n  ✅ 模型加载成功 | 耗时: {load_time:.2f}s")
    print(f"  Device: {inference.device}")
    print(f"  Image size: {inference.image_size}")
    print(f"  Resize mode: {inference.image_resize_mode}")
    print(f"  Context size: {inference.context_size}")
    print(f"  Trajectory length: {inference.len_traj_pred}")
    return True


def test_benchmark(args):
    """端到端推理基准测试。"""
    print("=" * 60)
    print("测试三：推理基准测试")
    print("=" * 60)

    if not args.policy_config or not args.policy_checkpoint:
        print("❌ 需要 --policy-config 和 --policy-checkpoint")
        return False

    import torch
    from shared.nomad_inference import NoMaDInferenceModule, NoMaDInferenceSpec

    spec = NoMaDInferenceSpec(
        policy_config=args.policy_config,
        policy_checkpoint=args.policy_checkpoint,
        scheduler_kind="ddim",
        ddim_steps=args.ddim_steps,
        image_resize_mode=args.image_resize_mode,
    )
    inference = NoMaDInferenceModule(spec)

    # 构造虚拟输入
    h, w = int(inference.image_size[1]), int(inference.image_size[0])
    dummy_obs = torch.randn(1, 3 * (inference.context_size + 1), h, w).to(inference.device)
    dummy_goal = torch.randn(1, 3, h, w).to(inference.device)

    # 预热
    print("  预热 (5 次)...")
    for _ in range(5):
        obs_cond = inference.encode_condition(dummy_obs, dummy_goal, goal_mask_value=0)
        inference.sample_actions(obs_cond)

    # 基准测试
    n_runs = 20
    encode_times = []
    sample_times = []
    total_times = []

    print(f"  运行 {n_runs} 次基准测试...")
    for i in range(n_runs):
        t_total = time.time()

        t0 = time.time()
        obs_cond = inference.encode_condition(dummy_obs, dummy_goal, goal_mask_value=0)
        encode_dt = (time.time() - t0) * 1000

        t0 = time.time()
        inference.sample_actions(obs_cond)
        sample_dt = (time.time() - t0) * 1000

        total_dt = (time.time() - t_total) * 1000
        encode_times.append(encode_dt)
        sample_times.append(sample_dt)
        total_times.append(total_dt)

    print(f"\n  📊 结果 (DDIM-{args.ddim_steps}):")
    print(f"     视觉编码:   {np.mean(encode_times):.1f} ± {np.std(encode_times):.1f} ms")
    print(f"     扩散采样:   {np.mean(sample_times):.1f} ± {np.std(sample_times):.1f} ms")
    print(f"     端到端:     {np.mean(total_times):.1f} ± {np.std(total_times):.1f} ms")
    print(f"     推理频率:   ~{1000/np.mean(total_times):.1f} Hz")
    print(f"\n  ✅ 基准测试完成")
    return True


def test_pipeline(args):
    """End-to-end pipeline test with steady-state latency reporting."""
    print("=" * 60)
    print("测试四：完整流水线 (相机 + 推理)")
    print("=" * 60)

    if not args.policy_config or not args.policy_checkpoint:
        print("❌ 需要 --policy-config 和 --policy-checkpoint")
        return False

    import torch
    from shared.nomad_inference import NoMaDInferenceModule, NoMaDInferenceSpec
    from deployment.lite3_real_bridge import OrinCamera

    cam = OrinCamera(
        device=args.camera_device,
        width=args.camera_width,
        height=args.camera_height,
        use_csi=args.use_csi,
        calibration_path=args.camera_calibration_path,
        undistort_alpha=args.undistort_alpha,
    )

    spec = NoMaDInferenceSpec(
        policy_config=args.policy_config,
        policy_checkpoint=args.policy_checkpoint,
        scheduler_kind="ddim",
        ddim_steps=args.ddim_steps,
        image_resize_mode=args.image_resize_mode,
    )
    inference = NoMaDInferenceModule(spec)

    context_size = inference.context_size
    frame_buffer = deque(maxlen=context_size + 1)

    print(f"  填充帧缓冲 (需要 {context_size + 1} 帧)...")
    for i in range(context_size + 1):
        img = cam.read()
        tensor = inference.pil_to_tensor(img)
        frame_buffer.append(tensor)
        print(f"    帧 {i + 1}/{context_size + 1}")

    n_runs = 10
    latencies = []
    print(f"\n  运行 {n_runs} 次端到端流水线...")

    for i in range(n_runs):
        t0 = time.time()

        img = cam.read()
        tensor = inference.pil_to_tensor(img)
        frame_buffer.append(tensor)

        obs_tensor = inference.build_obs_tensor(frame_buffer).to(inference.device)
        h, w = int(inference.image_size[1]), int(inference.image_size[0])
        fake_goal = torch.randn(1, 3, h, w).to(inference.device)

        obs_cond = inference.encode_condition(obs_tensor, fake_goal, goal_mask_value=1)
        actions = inference.sample_actions(obs_cond)
        mean_action = actions.mean(axis=0)
        waypoint = mean_action[min(2, inference.len_traj_pred - 1)]

        dt = (time.time() - t0) * 1000
        latencies.append(dt)
        print(f"    [{i + 1}/{n_runs}] {dt:.1f}ms | waypoint=({waypoint[0]:.3f}, {waypoint[1]:.3f})")

    cam.close()

    print("\n  📊 端到端流水线结果:")
    print(f"     平均延迟: {np.mean(latencies):.1f} ± {np.std(latencies):.1f} ms")
    print(f"     推理频率: ~{1000 / np.mean(latencies):.1f} Hz")
    if len(latencies) > 1:
        steady_latencies = latencies[1:]
        print(f"     稳态延迟(不含首轮): {np.mean(steady_latencies):.1f} ± {np.std(steady_latencies):.1f} ms")
        print(f"     稳态推理频率(不含首轮): ~{1000 / np.mean(steady_latencies):.1f} Hz")
        print("     注意: 首轮通常包含 CUDA 和调度器冷启动开销，请优先参考稳态数据。")

    print("\n  ✅ 流水线测试完成")
    return True


def main():
    parser = argparse.ArgumentParser(description="Orin 独立调试脚本")
    parser.add_argument("--test", choices=["camera", "model", "benchmark", "pipeline", "all"], default="all")
    parser.add_argument("--policy-config", type=str, default=None)
    parser.add_argument("--policy-checkpoint", type=str, default=None)
    parser.add_argument("--ddim-steps", type=int, default=5)
    parser.add_argument("--camera-device", default="0")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--use-csi", action="store_true")
    parser.add_argument("--camera-calibration-path", type=str, default=None)
    parser.add_argument("--undistort-alpha", type=float, default=0.0)
    parser.add_argument(
        "--image-resize-mode",
        choices=["stretch", "center_crop", "letterbox"],
        default="stretch",
        help="Preprocess camera images before NoMaD inference.",
    )
    args = parser.parse_args()

    tests = {
        "camera": test_camera,
        "model": test_model,
        "benchmark": test_benchmark,
        "pipeline": test_pipeline,
    }

    if args.test == "all":
        order = ["camera", "model", "benchmark", "pipeline"]
    else:
        order = [args.test]

    results = {}
    for name in order:
        try:
            ok = tests[name](args)
            results[name] = "✅ 通过" if ok else "❌ 失败"
        except Exception as e:
            results[name] = f"❌ 异常: {e}"
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:12s}: {status}")


if __name__ == "__main__":
    main()
