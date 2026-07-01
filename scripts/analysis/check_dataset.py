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
Day 1: 数据集完整性检查

检查 go_stanford 数据集中每条轨迹的完整性：
  - traj_data.pkl 是否存在
  - position / yaw 数组长度
  - 图像文件数量
"""

import os
import pickle
import glob
from datetime import datetime
from pathlib import Path

from tooling.project_paths import REPO_ROOT, repo_path

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_ROOT = str(REPO_ROOT)
DATASET_ROOT = str(repo_path("nomad_dataset", "go_stanford"))

trajs = sorted(path.name for path in Path(DATASET_ROOT).iterdir() if path.is_dir())
print(f"Found {len(trajs)} trajectories")

ok_count, bad_count = 0, 0
records = []

for traj in trajs[:5]:                         # 只检查前 5 条
    traj_path = Path(DATASET_ROOT) / traj
    pkl_path = traj_path / "traj_data.pkl"

    if not pkl_path.exists():
        print(f"  [MISSING] {traj}: no traj_data.pkl")
        bad_count += 1
        records.append((traj, "MISSING", 0, 0, 0))
        continue

    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    n_pos    = len(data.get("position", []))
    n_yaw    = len(data.get("yaw", []))
    n_images = len(glob.glob(str(traj_path / "*.jpg")))

    print(f"  [OK] {traj}: {n_pos} positions, {n_yaw} yaws, {n_images} images")
    ok_count += 1
    records.append((traj, "OK", n_pos, n_yaw, n_images))

print(f"\nCheck complete: {ok_count} OK, {bad_count} errors")

# ── 保存检查结果到 txt ──
out_dir = repo_path("results", "day1")
out_dir.mkdir(parents=True, exist_ok=True)
txt_path = out_dir / f"{RUN_TAG}_dataset_check.txt"
with txt_path.open("w", encoding="utf-8") as f:
    f.write(f"Run Time: {RUN_TAG}\n")
    f.write(f"Script: check_dataset.py\n")
    f.write(f"Dataset: {DATASET_ROOT}\n\n")
    f.write("# 列说明:\n")
    f.write("#   traj_name  - 轨迹子目录名（如 no10vc_10_0）\n")
    f.write("#   status     - OK = 有效，MISSING = 未找到 traj_data.pkl\n")
    f.write("#   n_pos      - traj_data.pkl 中 (x,y,z) 位置条目数\n")
    f.write("#   n_yaw      - traj_data.pkl 中航向角条目数\n")
    f.write("#   n_images   - 轨迹目录中 .jpg 帧数量\n")
    f.write("#   注意: n_pos 应等于 n_yaw，且与 n_images 近似\n\n")
    f.write(f"{'Trajectory':<22} {'Status':<10} {'Positions':<12} {'Yaws':<8} {'Images':<8}\n")
    f.write("-" * 60 + "\n")
    for traj_name, status, n_pos, n_yaw, n_img in records:
        f.write(f"{traj_name:<22} {status:<10} {n_pos:<12} {n_yaw:<8} {n_img:<8}\n")
    f.write(f"\nSummary: {ok_count} OK, {bad_count} errors out of {len(records)} checked\n")
    f.write(f"Total trajectories in dataset: {len(trajs)}\n")

print(f"Results saved: {txt_path}")
