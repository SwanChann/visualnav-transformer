#!/usr/bin/env python3
"""Capture a real-world topomap with the Orin camera."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from deployment.lite3_real_bridge import OrinCamera


def _next_index(output_dir: Path) -> int:
    existing_indices = []
    for image_path in output_dir.glob("*.png"):
        try:
            existing_indices.append(int(image_path.stem))
        except ValueError:
            continue
    return max(existing_indices, default=-1) + 1


def _write_metadata(output_dir: Path, args: argparse.Namespace, saved_files: list[Path]) -> None:
    metadata = {
        "domain": "real",
        "generated_by": "scripts/deployment/capture_real_topomap.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "map_name": args.map_name,
        "camera_device": str(args.camera_device),
        "use_csi": bool(args.use_csi),
        "image_width": int(args.camera_width),
        "image_height": int(args.camera_height),
        "capture_count": len(saved_files),
        "total_png_count": len(list(output_dir.glob("*.png"))),
        "note": "Real-world topomap captured by Orin camera. Use this directory with --topomap-dir on the real backend.",
    }
    (output_dir / "topomap_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture real-world topomap images on Jetson Orin.")
    parser.add_argument("--output-dir", default="deployment/topomaps/images/real_hallway")
    parser.add_argument("--map-name", default="real_hallway")
    parser.add_argument("--camera-device", default="0")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--use-csi", action="store_true")
    parser.add_argument("--csi-sensor-id", type=int, default=0)
    parser.add_argument("--csi-flip", type=int, default=0)
    parser.add_argument("--camera-calibration-path", default=None)
    parser.add_argument("--undistort-alpha", type=float, default=0.0)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--manual", action="store_true", help="Press Enter before each capture.")
    parser.add_argument("--start-index", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    start_index = _next_index(output_dir) if args.start_index is None else int(args.start_index)
    camera = OrinCamera(
        device=args.camera_device,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        use_csi=args.use_csi,
        csi_sensor_id=args.csi_sensor_id,
        csi_flip=args.csi_flip,
        calibration_path=args.camera_calibration_path,
        undistort_alpha=args.undistort_alpha,
    )

    print(f"[TopomapCapture] 输出目录: {output_dir}")
    print(f"[TopomapCapture] 起始编号: {start_index:03d}")
    print("[TopomapCapture] 沿真实路线缓慢移动相机，保持高度和朝向接近机器人第一视角。")
    saved_files: list[Path] = []
    try:
        for offset in range(max(int(args.count), 1)):
            image_index = start_index + offset
            if args.manual:
                input(f"[TopomapCapture] 移动到节点 {image_index:03d} 后按 Enter 保存...")
            else:
                if offset > 0:
                    time.sleep(max(float(args.interval), 0.0))

            image = camera.read()
            image_path = output_dir / f"{image_index:03d}.png"
            image.save(image_path)
            saved_files.append(image_path)
            print(f"[TopomapCapture] 保存 {image_path.name}: size={image.size}")
    finally:
        camera.close()

    _write_metadata(output_dir, args, saved_files)
    print(f"[TopomapCapture] 完成: 新增 {len(saved_files)} 张图像")
    print(f"[TopomapCapture] 已写入: {output_dir / 'topomap_meta.json'}")
    print(f"[TopomapCapture] 导航时使用: --topomap-dir {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
