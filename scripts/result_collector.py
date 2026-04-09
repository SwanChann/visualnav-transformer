#!/usr/bin/env python3
"""Deprecated wrapper. Use thesis_result_summary.py instead."""

from __future__ import annotations

from thesis_result_summary import main


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


# 中文注释：扫描现有结果文件，抽取可直接写入论文或汇报的关键指标
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize baseline, DDIM, and CFG results into a markdown report."
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


def extract_baseline(results_root: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for path in results_root.rglob("summary.txt"):
        text = read_text(path)
        patterns = {
            "model_params": r"Model params:\s*([0-9,]+)",
            "inference_time": r"DDPM inference time:\s*([0-9.]+\s*s)",
            "frequency": r"Inference frequency:\s*([0-9.]+\s*Hz)",
            "predicted_distance": r"Predicted distance:\s*([0-9.]+)",
        }
        found = False
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                data[key] = match.group(1)
                found = True
        if found:
            data["source"] = str(path.relative_to(REPO_ROOT))
            return data
    return data


def extract_ddim(results_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in results_root.rglob("*ddim_results.txt"):
        text = read_text(path)
        for line in text.splitlines():
            if not line.startswith("DD"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            row = {
                "config": " ".join(parts[0:1]) if parts[0].startswith("DDIM-") else " ".join(parts[0:2]),
                "time_ms": parts[-6],
                "mse": parts[-4],
                "diversity": parts[-3],
                "steps": parts[-2],
                "speedup": parts[-1],
            }
            rows.append(row)
        if rows:
            return rows
    return rows


def extract_cfg(results_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in results_root.rglob("results.txt"):
        if "day3" not in str(path).replace("\\", "/"):
            continue
        text = read_text(path)
        for line in text.splitlines():
            match = re.match(
                r"\s*(-?[0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
                line,
            )
            if not match:
                continue
            rows.append(
                {
                    "w": match.group(1),
                    "diversity": match.group(2),
                    "mean_displacement": match.group(3),
                    "time_ms": match.group(4),
                }
            )
        if rows:
            return rows
    return rows


def build_markdown(results_root: Path) -> List[str]:
    baseline = extract_baseline(results_root)
    ddim_rows = extract_ddim(results_root)
    cfg_rows = extract_cfg(results_root)

    lines = [
        "# Thesis Result Summary",
        "",
        f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Results root: {results_root.relative_to(REPO_ROOT) if results_root.is_relative_to(REPO_ROOT) else results_root}",
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

    lines.extend(["", "## DDIM", "", "| Config | Time(ms) | MSE | Diversity | Steps | Speedup |", "|---|---:|---:|---:|---:|---:|"])
    if ddim_rows:
        for row in ddim_rows:
            lines.append(
                f"| {row['config']} | {row['time_ms']} | {row['mse']} | {row['diversity']} | {row['steps']} | {row['speedup']} |"
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A |")

    lines.extend(["", "## CFG", "", "| w | Diversity | Mean Displacement | Time(ms) |", "|---:|---:|---:|---:|"])
    if cfg_rows:
        for row in cfg_rows:
            lines.append(
                f"| {row['w']} | {row['diversity']} | {row['mean_displacement']} | {row['time_ms']} |"
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
        print(f"\nSaved report: {out_file.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
