#!/usr/bin/env python3
"""Environment and project readiness checker for the thesis workflow."""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple


# 中文注释：统一通过脚本位置推导仓库根目录，避免任何绝对路径依赖
REPO_ROOT = Path(__file__).resolve().parents[2]


def format_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check repository structure, Python environment, dataset, and common files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "nomad_dataset" / "go_stanford",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "deployment" / "model_weights" / "nomad" / "nomad.pth",
        help="Expected model weight file.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the report under results/tools/.",
    )
    return parser.parse_args()


def status_line(name: str, ok: bool, detail: str) -> str:
    state = "OK" if ok else "WARN"
    return f"[{state}] {name}: {detail}"


def collect_structure_checks(repo_root: Path) -> List[str]:
    checks: List[Tuple[str, Path]] = [
        ("repo_root", repo_root),
        ("train_dir", repo_root / "train"),
        ("deployment_dir", repo_root / "deployment"),
        ("scripts_dir", repo_root / "scripts"),
        ("results_dir", repo_root / "results"),
        ("docs_dir", repo_root / "docs"),
        ("thesis_dir", repo_root / "毕设冲刺"),
        ("readme", repo_root / "README.md"),
        ("train_config", repo_root / "train" / "config" / "nomad.yaml"),
        ("deployment_config", repo_root / "deployment" / "config" / "models.yaml"),
        ("shared_inference", repo_root / "scripts" / "shared" / "nomad_inference.py"),
        ("navigation_host", repo_root / "scripts" / "deployment" / "nomad_navigation_host.py"),
        ("lite3_state_machine", repo_root / "scripts" / "simulation" / "nomad_mujoco_lite3_state_machine.py"),
        ("navigation_host_plan", repo_root / "scripts" / "configs" / "navigation_host" / "lite3_navigation_host_plan.json"),
    ]
    lines = []
    for name, path in checks:
        lines.append(status_line(name, path.exists(), format_path(path, repo_root)))
    return lines


def collect_python_checks() -> List[str]:
    lines = [
        status_line("python_version", sys.version_info >= (3, 8), platform.python_version()),
        status_line("platform", True, platform.platform()),
    ]

    try:
        import torch  # type: ignore

        cuda_detail = "available" if torch.cuda.is_available() else "not available"
        if torch.cuda.is_available():
            cuda_detail = f"available, device_count={torch.cuda.device_count()}"
        lines.append(status_line("torch", True, torch.__version__))
        lines.append(status_line("cuda", torch.cuda.is_available(), cuda_detail))
    except Exception as exc:  # pragma: no cover
        lines.append(status_line("torch", False, f"import failed: {exc}"))

    return lines


def count_items(directory: Path) -> int:
    try:
        return sum(1 for entry in directory.iterdir() if entry.is_dir())
    except Exception:
        return 0


def collect_dataset_checks(dataset_root: Path, repo_root: Path) -> List[str]:
    lines: List[str] = []
    exists = dataset_root.exists()
    display = format_path(dataset_root, repo_root)
    lines.append(status_line("dataset_root", exists, display))

    if not exists:
        return lines

    traj_count = count_items(dataset_root)
    lines.append(status_line("trajectory_count", traj_count > 0, str(traj_count)))

    sample_dirs = sorted([entry.name for entry in dataset_root.iterdir() if entry.is_dir()])[:3]
    sample_text = ", ".join(sample_dirs) if sample_dirs else "none"
    lines.append(status_line("sample_trajectories", len(sample_dirs) > 0, sample_text))
    return lines


def collect_weight_checks(weights: Path, repo_root: Path) -> List[str]:
    exists = weights.exists()
    display = format_path(weights, repo_root)
    return [status_line("model_weights", exists, display)]


def collect_results_checks(results_root: Path, repo_root: Path) -> List[str]:
    lines: List[str] = []
    if not results_root.exists():
        lines.append(status_line("results_root", False, str(results_root)))
        return lines

    txt_files = list(results_root.rglob("*.txt"))
    png_files = list(results_root.rglob("*.png"))
    lines.append(status_line("results_text_files", len(txt_files) > 0, str(len(txt_files))))
    lines.append(status_line("results_figures", len(png_files) > 0, str(len(png_files))))

    for target in [
        results_root / "day1",
        results_root / "day2",
        results_root / "day3",
        results_root / "day4",
        results_root / "day6",
        results_root / "deployment",
        results_root / "nomad_mujoco",
        results_root / "summary",
    ]:
        label = str(target.relative_to(repo_root))
        lines.append(status_line(label, target.exists(), "present" if target.exists() else "missing"))
    return lines


def build_report(args: argparse.Namespace) -> List[str]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "NoMaD Thesis Environment Check",
        f"Time: {now}",
        f"Repo root: {REPO_ROOT}",
        "",
        "## Structure",
    ]
    lines.extend(collect_structure_checks(REPO_ROOT))
    lines.extend(["", "## Python"])
    lines.extend(collect_python_checks())
    lines.extend(["", "## Dataset"])
    lines.extend(collect_dataset_checks(args.dataset_root, REPO_ROOT))
    lines.extend(["", "## Weights"])
    lines.extend(collect_weight_checks(args.weights, REPO_ROOT))
    lines.extend(["", "## Results"])
    lines.extend(collect_results_checks(REPO_ROOT / "results", REPO_ROOT))
    return lines


def save_report(lines: Iterable[str]) -> Path:
    out_dir = REPO_ROOT / "results" / "tools"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"env_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    return out_file


def main() -> int:
    args = parse_args()
    lines = build_report(args)
    text = "\n".join(lines)
    print(text)

    if args.save:
        out_file = save_report(lines)
        print(f"\nSaved report: {out_file.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
