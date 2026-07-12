#!/usr/bin/env python3
"""Generate controlled-only RA-L Table III and figures from U11 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIG_LABELS = {
    "ddpm10_cfg0_standard_k8": "DDPM-10",
    "ddim2_cfg0_standard_k8": "DDIM-2",
    "ddim3_cfg0_standard_k8": "DDIM-3",
    "ddim2_cfg0_tts8": "DDIM-2 + TTS-8",
    "ddim2_cfg2_tts8": "DDIM-2 + CFG-2 + TTS-8",
}
CONFIG_ORDER = list(CONFIG_LABELS)
STRATA = ["policy_only_off", "system_stabilizer_on"]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_table(summary: list[dict]) -> list[dict]:
    rows = []
    for item in summary:
        rows.append({
            "configuration": CONFIG_LABELS[item["config_id"]], "config_id": item["config_id"], "stabilizer": item["stabilizer_mode"],
            "success_n": item["success_n"], "scientific_n": item["scientific_n"], "success_rate": item["success_rate"],
            "success_ci_low": item["success_clopper_pearson_95"][0], "success_ci_high": item["success_clopper_pearson_95"][1],
            "spl_mean": item["spl_mean"], "final_distance_m": item["final_distance_m_mean"], "path_efficiency": item["path_efficiency_mean"],
            "collision_rate": item["collision_rate"], "fall_rate": item["fall_rate"], "stuck_rate": item["stuck_rate"], "timeout_rate": item["timeout_rate"],
            "sampler_p95_ms": item["sampler_p95_ms_mean"], "full_loop_p95_ms": item["full_loop_p95_ms_mean"], "loop_hz": item["loop_hz_mean_mean"],
            "invalid_n": item["infrastructure_invalid_n"], "crash_n": item["crash_n"],
            "trajectory_equivalent_across_methods": item["trajectory_equivalent_across_methods"],
        })
    return rows


def write_table(output_dir: Path, rows: list[dict], policy_action_scale_m: float) -> None:
    with (output_dir / "table_iii.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output_dir / "table_iii.md").open("w", encoding="utf-8") as handle:
        handle.write("| Configuration | Stabilizer | Success | SPL | Final dist. m | Full-loop p95 ms | Stuck | Timeout |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(f"| {row['configuration']} | {row['stabilizer']} | {row['success_n']}/{row['scientific_n']} | {row['spl_mean']:.3f} | {row['final_distance_m']:.3f} | {row['full_loop_p95_ms']:.1f} | {row['stuck_rate']:.2f} | {row['timeout_rate']:.2f} |\n")
        handle.write(f"\nNote: native NoMaD actions are converted with target-platform scale {policy_action_scale_m:g} m before waypoint control. All system-stabilizer trajectories are byte-identical across the five methods for every scene/seed; those rows differ only in measured compute latency and are not independent navigation outcomes.\n")


def plot_success(output_dir: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, stratum in zip(axes, STRATA):
        bucket = [next(row for row in rows if row["config_id"] == config and row["stabilizer"] == stratum) for config in CONFIG_ORDER]
        rates = np.asarray([row["success_rate"] for row in bucket])
        lows = np.asarray([row["success_ci_low"] for row in bucket]); highs = np.asarray([row["success_ci_high"] for row in bucket])
        x = np.arange(len(bucket))
        axis.bar(x, rates, color="#4472C4")
        axis.errorbar(x, rates, yerr=np.vstack([rates - lows, highs - rates]), fmt="none", color="black", capsize=3)
        axis.set_xticks(x, [CONFIG_LABELS[config] for config in CONFIG_ORDER], rotation=30, ha="right")
        title = stratum.replace("_", " ")
        if stratum == "system_stabilizer_on":
            title += "\n(method trajectories identical)"
        axis.set_title(title)
        axis.set_ylim(0, 1.05); axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Episode success rate")
    fig.suptitle("Controlled MuJoCo success (descriptive; 3 seeds, 2 scenes)")
    fig.tight_layout(); fig.savefig(output_dir / "fig_controlled_success.pdf", bbox_inches="tight"); plt.close(fig)


def plot_pareto(output_dir: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, stratum in zip(axes, STRATA):
        for config in CONFIG_ORDER:
            row = next(item for item in rows if item["config_id"] == config and item["stabilizer"] == stratum)
            axis.scatter(row["full_loop_p95_ms"], row["spl_mean"], s=55, label=CONFIG_LABELS[config])
        axis.set_title(stratum.replace("_", " ")); axis.set_xlabel("Full-loop p95 latency (ms)"); axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean SPL")
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("Closed-loop latency–efficiency trade-off")
    fig.tight_layout(); fig.savefig(output_dir / "fig_controlled_latency_spl.pdf", bbox_inches="tight"); plt.close(fig)


def plot_trajectories(output_dir: Path, episodes: list[dict[str, str]]) -> None:
    candidates = [row for row in episodes if row["scene_id"] == "medium" and row["diffusion_seed"] == "11" and row["stabilizer_mode"] == "policy_only_off"]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for row in candidates:
        record = json.loads((Path(__file__).resolve().parents[2] / row["source_trial_json"]).read_text(encoding="utf-8"))
        xy = np.asarray([[point["x_m"], point["y_m"]] for point in record["trajectory"]])
        axis.plot(xy[:, 0], xy[:, 1], label=CONFIG_LABELS[row["config_id"]])
    axis.scatter([0, 7], [0, 0], marker="*", s=100, c=["black", "green"], label="start / goal")
    axis.set_xlabel("x (m)"); axis.set_ylabel("y (m)"); axis.grid(alpha=0.25); axis.legend(fontsize=7)
    axis.set_title("Stored controlled trajectories: medium, seed 11, policy-only")
    fig.tight_layout(); fig.savefig(output_dir / "fig_controlled_trajectories.pdf", bbox_inches="tight"); plt.close(fig)


def generate(analysis_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((analysis_dir / "summary_by_config_stratum.json").read_text(encoding="utf-8"))
    audit = json.loads((analysis_dir / "audit.json").read_text(encoding="utf-8"))
    episodes = load_csv(analysis_dir / "controlled_episodes.csv")
    rows = build_table(summary)
    write_table(output_dir, rows, float(audit.get("policy_action_scale_m", 1.0))); plot_success(output_dir, rows); plot_pareto(output_dir, rows); plot_trajectories(output_dir, episodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args(); generate(args.analysis_dir.resolve(), args.output_dir.resolve()); return 0


if __name__ == "__main__":
    raise SystemExit(main())
