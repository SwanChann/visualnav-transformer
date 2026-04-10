#!/usr/bin/env python3
"""Collect thesis-ready summaries from the current result files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# 中文注释：扫描现有结果文件，优先抽取正式统计实验结果，避免引用快速验证旧口径
REPO_ROOT = Path(__file__).resolve().parent.parent


def format_path(path: Path | None, base: Path) -> str:
    if path is None:
        return "N/A"
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize thesis-ready baseline, statistical experiments, and platform results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results",
        help="Results root directory.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the markdown report under results/summary/.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def format_number(value: str | float | int | None, digits: int = 3) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{digits}f}"


def latest_file(results_root: Path, pattern: str, path_fragment: str | None = None) -> Path | None:
    candidates: List[Path] = []
    for path in results_root.rglob(pattern):
        normalized = str(path).replace("\\", "/")
        if path_fragment is not None and path_fragment not in normalized:
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def read_csv_rows(path: Path | None) -> List[Dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def extract_baseline(results_root: Path) -> Dict[str, str]:
    candidates: List[Tuple[float, Dict[str, str]]] = []
    patterns = {
        "model_params": r"Model params:\s*([0-9,]+)",
        "inference_time": r"DDPM inference time:\s*([0-9.]+\s*s)",
        "frequency": r"Inference frequency:\s*([0-9.]+\s*Hz)",
        "predicted_distance": r"Predicted distance:\s*([0-9.]+)",
    }

    for path in results_root.rglob("summary.txt"):
        text = read_text(path)
        data: Dict[str, str] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                data[key] = match.group(1)
        if data:
            data["source"] = format_path(path, REPO_ROOT)
            candidates.append((path.stat().st_mtime, data))

    if not candidates:
        return {}
    return max(candidates, key=lambda item: item[0])[1]


def extract_ddim(results_root: Path) -> tuple[Path | None, List[Dict[str, str]]]:
    path = latest_file(results_root, "ddim_overall_summary.csv", "ddim_stat_experiment")
    return path, read_csv_rows(path)


def extract_cfg(results_root: Path) -> tuple[Path | None, List[Dict[str, str]]]:
    path = latest_file(results_root, "cfg_overall_summary.csv", "cfg_stat_experiment")
    return path, read_csv_rows(path)


def extract_joint(results_root: Path) -> tuple[Path | None, List[Dict[str, str]]]:
    path = latest_file(results_root, "overall_summary.csv", "joint_ddim_cfg_experiment")
    return path, read_csv_rows(path)


def extract_encoder(results_root: Path) -> tuple[Path | None, List[Dict[str, str]]]:
    path = latest_file(results_root, "encoder_overall_summary.csv", "encoder_comparison_experiment")
    return path, read_csv_rows(path)


def parse_key_value_summary(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {"source": format_path(path, REPO_ROOT)}
    for line in read_text(path).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip().lower().replace(" ", "_")] = value.strip()
    return data


def extract_lite3(results_root: Path) -> List[Dict[str, str]]:
    summaries = [parse_key_value_summary(path) for path in results_root.glob("nomad_mujoco/*/summary.txt")]
    return sorted(summaries, key=lambda row: row["source"])[-12:]


def extract_tron1(results_root: Path) -> tuple[Path | None, List[Dict[str, object]]]:
    path = latest_file(results_root, "tron1_manifest.json", "tron1_tron1_plan")
    if path is None:
        return None, []
    payload = json.loads(read_text(path))
    return path, list(payload.get("asset_checks", []))


def build_markdown(results_root: Path) -> List[str]:
    baseline = extract_baseline(results_root)
    ddim_path, ddim_rows = extract_ddim(results_root)
    cfg_path, cfg_rows = extract_cfg(results_root)
    joint_path, joint_rows = extract_joint(results_root)
    encoder_path, encoder_rows = extract_encoder(results_root)
    lite3_rows = extract_lite3(results_root)
    tron1_path, tron1_rows = extract_tron1(results_root)

    lines = [
        "# Thesis Result Summary",
        "",
        f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Results root: {format_path(results_root, REPO_ROOT)}",
        "",
        "## Baseline",
    ]

    if baseline:
        lines.extend(
            [
                f"- Source: `{baseline.get('source', 'unknown')}`",
                f"- Model params: {baseline.get('model_params', 'N/A')}",
                f"- DDPM inference time: {baseline.get('inference_time', 'N/A')}",
                f"- Inference frequency: {baseline.get('frequency', 'N/A')}",
                f"- Predicted distance: {baseline.get('predicted_distance', 'N/A')}",
            ]
        )
    else:
        lines.append("- No baseline summary file detected.")

    lines.extend(
        [
            "",
            "## DDIM Statistical Result",
            "",
            f"- Source: `{format_path(ddim_path, REPO_ROOT)}`",
            "",
            "| Config | Cases | Latency(ms) | CI95(ms) | MSE vs DDPM-10 | Diversity | Speedup |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if ddim_rows:
        for row in ddim_rows:
            lines.append(
                f"| {row.get('config_name', 'N/A')} | "
                f"{row.get('num_cases', 'N/A')} | "
                f"{format_number(row.get('latency_ms_mean'), 2)} | "
                f"{format_number(row.get('latency_ms_ci95'), 2)} | "
                f"{format_number(row.get('mse_vs_baseline_mean'), 6)} | "
                f"{format_number(row.get('diversity_mean'), 4)} | "
                f"{format_number(row.get('speedup_vs_ddpm10'), 2)}x |"
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

    lines.extend(
        [
            "",
            "## CFG Statistical Result",
            "",
            f"- Source: `{format_path(cfg_path, REPO_ROOT)}`",
            "",
            "| w | Cases | Latency(ms) | Shift vs w=0 | Diversity | Forward Progress | Lateral Abs |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if cfg_rows:
        for row in cfg_rows:
            lines.append(
                f"| {row.get('guidance_scale', 'N/A')} | "
                f"{row.get('num_cases', 'N/A')} | "
                f"{format_number(row.get('latency_ms_mean'), 2)} | "
                f"{format_number(row.get('shift_vs_w0_mean'), 6)} | "
                f"{format_number(row.get('diversity_mean'), 4)} | "
                f"{format_number(row.get('forward_progress_mean_mean'), 3)} | "
                f"{format_number(row.get('lateral_abs_mean_mean'), 3)} |"
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

    lines.extend(
        [
            "",
            "## Joint DDIM x CFG",
            "",
            f"- Source: `{format_path(joint_path, REPO_ROOT)}`",
            "",
            "| Rank | Config | Latency(ms) | Forward Progress | Smoothness | Diversity | Score |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    if joint_rows:
        for rank, row in enumerate(joint_rows[:5], 1):
            lines.append(
                f"| {rank} | {row.get('config_name', 'N/A')} | "
                f"{format_number(row.get('latency_ms_mean'), 2)} | "
                f"{format_number(row.get('forward_progress_mean_mean'), 3)} | "
                f"{format_number(row.get('smoothness_mean_mean'), 4)} | "
                f"{format_number(row.get('diversity_mean'), 4)} | "
                f"{format_number(row.get('composite_score'), 4)} |"
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

    lines.extend(
        [
            "",
            "## Encoder Comparison",
            "",
            f"- Source: `{format_path(encoder_path, REPO_ROOT)}`",
            "",
            "| Model | Cases | Latency(ms) | Diversity | Forward Progress | Smoothness |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    if encoder_rows:
        for row in encoder_rows:
            lines.append(
                f"| {row.get('model_name', 'N/A')} | "
                f"{row.get('num_cases', 'N/A')} | "
                f"{format_number(row.get('latency_ms_mean'), 2)} | "
                f"{format_number(row.get('diversity_mean'), 4)} | "
                f"{format_number(row.get('forward_progress_mean_mean'), 3)} | "
                f"{format_number(row.get('smoothness_mean_mean'), 4)} |"
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A |")

    lines.extend(
        [
            "",
            "## Lite3 MuJoCo Results",
            "",
            "| Source | Mode | Steps | Distance | Reached Goal |",
            "|---|---|---:|---:|---|",
        ]
    )
    if lite3_rows:
        for row in lite3_rows:
            lines.append(
                f"| `{row.get('source', 'N/A')}` | "
                f"{row.get('mode', 'N/A')} | "
                f"{row.get('steps', 'N/A')} | "
                f"{row.get('total_distance', 'N/A')} | "
                f"{row.get('reached_goal', 'N/A')} |"
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A |")

    lines.extend(
        [
            "",
            "## Tron1 Asset Status",
            "",
            f"- Source: `{format_path(tron1_path, REPO_ROOT)}`",
            "",
            "| Asset | Exists | Required | Path |",
            "|---|---|---|---|",
        ]
    )
    if tron1_rows:
        for row in tron1_rows:
            lines.append(
                f"| {row.get('name', 'N/A')} | "
                f"{'yes' if row.get('exists') else 'no'} | "
                f"{'yes' if row.get('required') else 'no'} | "
                f"`{row.get('path', 'N/A')}` |"
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A |")

    return lines


def save_report(lines: Iterable[str]) -> Path:
    out_dir = REPO_ROOT / "results" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"thesis_result_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    return out_file


def main() -> int:
    args = parse_args()
    lines = build_markdown(args.results_root)
    print("\n".join(lines))

    if args.save:
        out_file = save_report(lines)
        print(f"\nSaved report: {format_path(out_file, REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
