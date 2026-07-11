#!/usr/bin/env python3
"""Aggregate frozen Lite3 timing logs and prepare a manual outcome template."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


TIMING_FIELDS = (
    "timestamp", "label", "tick", "camera_ms", "ui_ms", "preprocess_ms",
    "infer_ms", "command_ms", "total_ms", "infer_fps", "loop_fps", "camera_status"
)
NUMERIC_FIELDS = (
    "tick", "camera_ms", "ui_ms", "preprocess_ms", "infer_ms", "command_ms",
    "total_ms", "infer_fps", "loop_fps"
)


class TimingError(ValueError):
    pass


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise TimingError("Cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def infer_config_id(trial_id: str) -> tuple[str, str]:
    match = re.fullmatch(r"ddim(\d+)cfg([0-9.]+)tts(\d+)", trial_id.lower())
    if match:
        return f"ddim{match.group(1)}_cfg{match.group(2)}_tts{match.group(3)}", "directory_name_exact_pattern"
    if trial_id.lower() == "ddpm":
        return "ddpm_steps_unknown_cfg_unknown_tts_unknown", "directory_name_partial"
    return "unknown", "not_encoded_in_log"


def _number(row: dict[str, str], field: str, location: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise TimingError(f"{location}: invalid {field}") from exc
    if not math.isfinite(value):
        raise TimingError(f"{location}: non-finite {field}")
    return value


def load_timing_files(root: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in sorted(root.rglob("timing_profile.csv")):
        trial_dir = path.parent.parent
        trial_id = trial_dir.name
        config_id, config_source = infer_config_id(trial_id)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [field for field in TIMING_FIELDS if field not in (reader.fieldnames or [])]
            if missing:
                raise TimingError(f"{path}: missing columns {missing}")
            for index, row in enumerate(reader, start=2):
                location = f"{path} row {index}"
                sample = {
                    "trial_id": trial_id,
                    "trial_path": trial_dir.as_posix(),
                    "timing_path": path.as_posix(),
                    "config_id": config_id,
                    "config_source": config_source,
                    "timestamp": row["timestamp"],
                    "timestamp_clock_valid": not row["timestamp"].startswith("1970-"),
                    "label": row["label"],
                    "camera_status": row["camera_status"],
                }
                for field in NUMERIC_FIELDS:
                    sample[field] = _number(row, field, location)
                samples.append(sample)
    if not samples:
        raise TimingError(f"No timing_profile.csv records under {root}")
    return samples


def summarize(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in samples:
        grouped.setdefault(row["trial_path"], []).append(row)
    output = []
    for trial_path, rows in sorted(grouped.items()):
        result: dict[str, Any] = {
            "trial_id": rows[0]["trial_id"],
            "trial_path": trial_path,
            "config_id": rows[0]["config_id"],
            "config_source": rows[0]["config_source"],
            "num_profiled_loops": len(rows),
            "timestamp_clock_valid": all(row["timestamp_clock_valid"] for row in rows),
        }
        for field in ("camera_ms", "ui_ms", "preprocess_ms", "infer_ms", "command_ms", "total_ms", "infer_fps", "loop_fps"):
            values = [float(row[field]) for row in rows]
            result[f"{field}_mean"] = statistics.mean(values)
            result[f"{field}_p50"] = percentile(values, 0.50)
            result[f"{field}_p90"] = percentile(values, 0.90)
            result[f"{field}_p95"] = percentile(values, 0.95)
            result[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(result)
    return output


def build_annotation_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted(root.rglob("interactive_summary.txt")):
        trial_dir = summary_path.parent
        text = summary_path.read_text(encoding="utf-8", errors="replace")
        backend = re.search(r"^Backend:\s*(.+?),\s*Map:\s*(.+)$", text, re.MULTILINE)
        auto = re.search(r"status=([^\s]+)", text)
        timing = trial_dir / "logs" / "timing_profile.csv"
        meta_files = sorted(trial_dir.rglob("recording_meta.json"))
        rows.append(
            {
                "trial_id": trial_dir.name,
                "trial_path": trial_dir.as_posix(),
                "backend": backend.group(1).strip() if backend else "unknown",
                "map": backend.group(2).strip() if backend else "unknown",
                "auto_status_untrusted": auto.group(1) if auto else "unknown",
                "timing_available": timing.is_file(),
                "timing_path": timing.as_posix() if timing.is_file() else "NA",
                "recording_meta_count": len(meta_files),
                "manual_outcome": "",
                "failure_mode": "",
                "video_reviewed_by": "",
                "video_review_date": "",
                "evidence_usable": "",
                "notes": "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise TimingError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Frozen Real-robot Timing Summary",
        "",
        "Computed from existing sampled loop records; no robot experiment was run.",
        "",
        "| Trial label | Config ID | N | Total mean / p50 / p95 (ms) | Infer p95 (ms) | Loop FPS mean | Clock valid |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['trial_id']} | {row['config_id']} | {row['num_profiled_loops']} | "
            f"{row['total_ms_mean']:.1f} / {row['total_ms_p50']:.1f} / {row['total_ms_p95']:.1f} | "
            f"{row['infer_ms_p95']:.1f} | {row['loop_fps_mean']:.2f} | {row['timestamp_clock_valid']} |"
        )
    lines.extend(
        [
            "",
            "`N` is the number of profiled loop samples, not navigation trials. The profiler records "
            "one loop every configured interval (`profile_interval`), so these rows are sampled "
            "per-loop timings rather than a continuous trace.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_notes(path: Path, samples: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> None:
    invalid_clock_trials = sorted({row["trial_id"] for row in samples if not row["timestamp_clock_valid"]})
    unknown_configs = sorted({row["trial_id"] for row in samples if row["config_id"] == "unknown"})
    lines = [
        "# Real-robot Timing Evidence Notes",
        "",
        f"- Timing files: {len(set(row['timing_path'] for row in samples))}",
        f"- Profiled loop samples: {len(samples)}",
        f"- Frozen trial folders with interactive summaries: {len(annotations)}",
        f"- Trials with invalid 1970 wall-clock timestamps: {invalid_clock_trials}",
        f"- Timing trials whose exact inference configuration is not encoded in the log/path: {unknown_configs}",
        "",
        "## Interpretation boundary",
        "",
        "- `total_ms` is the measured profiled loop for that sampled tick; `loop_fps=1000/total_ms`.",
        "- p50/p90/p95 are across sampled individual loop records, not across trial means.",
        "- The CSV is sampled at `profile_interval`, so it may miss unprofiled spikes and is not a complete tail-latency trace.",
        "- Absolute timestamps beginning in 1970 are invalid due to the robot clock; duration fields remain usable if the profiler clock was monotonic.",
        "- Directory names matching `ddim<steps>cfg<weight>tts<budget>` are parsed as configuration metadata. Other labels remain `unknown`; no configuration is guessed from words such as baseline/optimal.",
        "- `interactive_summary.txt status=success` is copied only as `auto_status_untrusted`. It must not populate Table IV until a human reviews the video.",
        "- One timing file is one frozen execution folder, not a randomized repeated trial. Do not calculate success rate from timing-file count.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(root: Path, out: Path) -> dict[str, int]:
    samples = load_timing_files(root)
    summaries = summarize(samples)
    annotations = build_annotation_rows(root)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "real_timing_samples.csv", samples)
    write_csv(out / "real_timing_summary.csv", summaries)
    write_csv(out / "real_trial_annotation_template.csv", annotations)
    write_table(out / "table_real_timing.md", summaries)
    write_notes(out / "real_timing_notes.md", samples, annotations)
    audit = {
        "timing_files": len({row["timing_path"] for row in samples}),
        "profiled_loop_samples": len(samples),
        "timing_trials": len(summaries),
        "annotation_trials": len(annotations),
        "manually_labeled_trials": sum(bool(row["manual_outcome"]) for row in annotations),
    }
    (out / "real_timing_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate existing real-robot timing logs only.")
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audit = run(args.trial_root, args.out)
    except (TimingError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"Wrote real-robot timing package: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
