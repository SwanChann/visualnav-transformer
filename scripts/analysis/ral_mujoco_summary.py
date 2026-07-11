#!/usr/bin/env python3
"""Unify existing MuJoCo benchmark JSON without running simulation.

The script is intentionally standard-library only.  It treats every benchmark
directory as a separate source because historical runs may use different maps,
controller/stabilizer settings, or success definitions.  Missing provenance is
reported, never imputed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = (
    "encoder",
    "map_name",
    "scheduler",
    "ddim_steps",
    "cfg_weight",
    "run_index",
    "success",
    "steps",
    "path_distance",
    "final_goal_dist",
    "wall_time",
    "exit_code",
)

OPTIONAL_FIELDS = ("seed", "latency_ms", "stuck", "fall", "stabilizer_mode")

CSV_FIELDS = (
    "source_experiment",
    "source_path",
    "encoder",
    "map",
    "scheduler",
    "ddim_steps",
    "cfg_weight",
    "run_index",
    "seed",
    "success",
    "final_distance",
    "steps",
    "path_length",
    "wall_time_s",
    "latency_ms",
    "stuck",
    "fall",
    "stabilizer_mode",
    "exit_code",
    "record_flags",
)


class DataError(ValueError):
    """Raised when a source record cannot be interpreted without guessing."""


def _require_number(record: dict[str, Any], field: str, source: str) -> float:
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"{source}: field {field!r} must be numeric, got {value!r}")
    if not math.isfinite(float(value)):
        raise DataError(f"{source}: field {field!r} is not finite")
    return float(value)


def _format_optional(value: Any) -> Any:
    return "NA" if value is None else value


def discover_sources(bench_root: Path) -> list[Path]:
    if not bench_root.is_dir():
        raise DataError(f"Benchmark root does not exist: {bench_root}")
    return sorted(bench_root.rglob("raw_results.json"))


def load_records(
    bench_root: Path, success_distance_threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unified: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    for raw_path in discover_sources(bench_root):
        source_id = raw_path.parent.relative_to(bench_root).as_posix()
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"Cannot read {raw_path}: {exc}") from exc
        if not isinstance(payload, list):
            raise DataError(f"{raw_path}: top-level JSON must be a list")

        source_rows: list[dict[str, Any]] = []
        for index, record in enumerate(payload):
            location = f"{raw_path} record {index}"
            if not isinstance(record, dict):
                raise DataError(f"{location}: record must be an object")
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                raise DataError(f"{location}: missing required fields {missing}")
            if not isinstance(record["success"], bool):
                raise DataError(f"{location}: success must be a JSON boolean")

            steps = _require_number(record, "steps", location)
            path_distance = _require_number(record, "path_distance", location)
            final_distance = _require_number(record, "final_goal_dist", location)
            wall_time = _require_number(record, "wall_time", location)
            _require_number(record, "ddim_steps", location)
            _require_number(record, "cfg_weight", location)
            _require_number(record, "run_index", location)
            _require_number(record, "exit_code", location)

            flags = [f"missing_{field}" for field in OPTIONAL_FIELDS if field not in record]
            if record["success"] and final_distance > success_distance_threshold:
                flags.append("success_final_distance_inconsistent")
            if int(record["exit_code"]) != 0:
                flags.append("nonzero_exit_code")

            latency = record.get("latency_ms", record.get("inference_latency_ms"))
            row = {
                "source_experiment": source_id,
                "source_path": raw_path.as_posix(),
                "encoder": str(record["encoder"]),
                "map": str(record["map_name"]),
                "scheduler": str(record["scheduler"]),
                "ddim_steps": int(record["ddim_steps"]),
                "cfg_weight": float(record["cfg_weight"]),
                "run_index": int(record["run_index"]),
                "seed": _format_optional(record.get("seed")),
                "success": record["success"],
                "final_distance": final_distance,
                "steps": int(steps),
                "path_length": path_distance,
                "wall_time_s": wall_time,
                "latency_ms": _format_optional(latency),
                "stuck": _format_optional(record.get("stuck")),
                "fall": _format_optional(record.get("fall")),
                "stabilizer_mode": _format_optional(record.get("stabilizer_mode")),
                "exit_code": int(record["exit_code"]),
                "record_flags": ";".join(flags) if flags else "none",
            }
            unified.append(row)
            source_rows.append(row)

        sources.append(
            {
                "source_experiment": source_id,
                "raw_path": raw_path.as_posix(),
                "records": len(source_rows),
                "all_success": bool(source_rows) and all(row["success"] for row in source_rows),
            }
        )

    if not unified:
        raise DataError(f"No raw_results.json records found under {bench_root}")
    return unified, sources


def config_key(row: dict[str, Any], include_source: bool = True) -> tuple[Any, ...]:
    key: tuple[Any, ...] = (
        row["encoder"],
        row["map"],
        row["scheduler"],
        row["ddim_steps"],
        row["cfg_weight"],
    )
    return (row["source_experiment"],) + key if include_source else key


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    items = list(values)
    return statistics.mean(items), statistics.stdev(items) if len(items) > 1 else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[config_key(row)].append(row)

    summaries = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        success_count = sum(bool(row["success"]) for row in group)
        steps_mean, steps_std = _mean_std(float(row["steps"]) for row in group)
        path_mean, path_std = _mean_std(float(row["path_length"]) for row in group)
        final_mean, final_std = _mean_std(float(row["final_distance"]) for row in group)
        wall_mean, wall_std = _mean_std(float(row["wall_time_s"]) for row in group)
        summaries.append(
            {
                "source_experiment": key[0],
                "encoder": key[1],
                "map": key[2],
                "scheduler": key[3],
                "ddim_steps": key[4],
                "cfg_weight": key[5],
                "runs": len(group),
                "success_count": success_count,
                "success_pct": 100.0 * success_count / len(group),
                "steps_mean": steps_mean,
                "steps_std": steps_std,
                "path_mean": path_mean,
                "path_std": path_std,
                "final_mean": final_mean,
                "final_std": final_std,
                "wall_mean": wall_mean,
                "wall_std": wall_std,
                "flagged_records": sum(row["record_flags"] != "none" for row in group),
            }
        )
    return summaries


def build_audit(
    rows: list[dict[str, Any]], sources: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    missing_counts = {
        field: sum(row[field] == "NA" for row in rows) for field in OPTIONAL_FIELDS
    }
    inconsistent = [
        row
        for row in rows
        if "success_final_distance_inconsistent" in row["record_flags"]
    ]

    cross_source: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in rows:
        cross_source[config_key(row, include_source=False)].add(row["source_experiment"])
    duplicate_configs = [
        {"config": list(key), "sources": sorted(source_ids)}
        for key, source_ids in cross_source.items()
        if len(source_ids) > 1
    ]

    repeated: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    for row in rows:
        signature = (
            row["source_experiment"],
            row["map"],
            row["run_index"],
            row["steps"],
            round(float(row["path_length"]), 6),
            round(float(row["final_distance"]), 6),
        )
        repeated[signature].add(config_key(row, include_source=False))
    repeated_signatures = [
        {"signature": list(signature), "distinct_configs": len(configs)}
        for signature, configs in repeated.items()
        if len(configs) > 1
    ]

    return {
        "success_distance_threshold": threshold,
        "source_count": len(sources),
        "record_count": len(rows),
        "config_group_count": len(summarize(rows)),
        "sources": sources,
        "missing_optional_field_counts": missing_counts,
        "success_final_distance_inconsistency_count": len(inconsistent),
        "all_success_source_count": sum(source["all_success"] for source in sources),
        "duplicate_config_across_source_count": len(duplicate_configs),
        "duplicate_configs_across_sources": duplicate_configs,
        "repeated_trajectory_signature_count": len(repeated_signatures),
        "repeated_trajectory_signatures": repeated_signatures,
        "trajectory_coordinates_available": False,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Existing MuJoCo Closed-loop Summary",
        "",
        "This table is recomputed from existing `raw_results.json`; no simulation was run.",
        "Rows remain source-scoped because historical controller/stabilizer provenance is missing.",
        "",
        "| Source | Encoder | Map | Sampler | CFG | Success | Steps | Path (m) | Final dist. (m) | Wall time (s) | Flagged records |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        sampler = row["scheduler"]
        if sampler == "ddim":
            sampler = f"ddim-{row['ddim_steps']}"
        lines.append(
            "| {source} | {encoder} | {map_name} | {sampler} | {cfg:.2g} | "
            "{success}/{runs} ({pct:.1f}%) | {steps:.1f}±{steps_std:.1f} | "
            "{path_mean:.2f}±{path_std:.2f} | {final_mean:.2f}±{final_std:.2f} | "
            "{wall:.1f}±{wall_std:.1f} | {flags} |".format(
                source=row["source_experiment"],
                encoder=row["encoder"],
                map_name=row["map"],
                sampler=sampler,
                cfg=row["cfg_weight"],
                success=row["success_count"],
                runs=row["runs"],
                pct=row["success_pct"],
                steps=row["steps_mean"],
                steps_std=row["steps_std"],
                path_mean=row["path_mean"],
                path_std=row["path_std"],
                final_mean=row["final_mean"],
                final_std=row["final_std"],
                wall=row["wall_mean"],
                wall_std=row["wall_std"],
                flags=row["flagged_records"],
            )
        )
    lines.extend(
        [
            "",
            "Table note: these runs can support closed-loop integration evidence only. Missing "
            "stabilizer metadata prevents attribution to the navigation policy alone.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_note(path: Path, audit: dict[str, Any]) -> None:
    missing = audit["missing_optional_field_counts"]
    lines = [
        "# MuJoCo Evidence and Stabilizer Audit",
        "",
        "## What was found",
        "",
        f"- Sources with raw records: {audit['source_count']}",
        f"- Raw records: {audit['record_count']}",
        f"- Source-scoped configuration/map groups: {audit['config_group_count']}",
        f"- Sources whose recorded runs are all successful: {audit['all_success_source_count']}",
        f"- `success=true` records above {audit['success_distance_threshold']:.2f} m final distance: "
        f"{audit['success_final_distance_inconsistency_count']}",
        f"- Config/map identities repeated across sources: {audit['duplicate_config_across_source_count']}",
        f"- Trajectory-summary signatures repeated across distinct configs: "
        f"{audit['repeated_trajectory_signature_count']}",
        "",
        "## Missing provenance",
        "",
    ]
    for field in OPTIONAL_FIELDS:
        lines.append(f"- `{field}` missing in {missing[field]}/{audit['record_count']} records.")
    lines.extend(
        [
            "",
            "The current JSON cannot determine route-stabilizer mode, per-step inference latency, "
            "stuck/fall events, or randomized seed identity. Therefore the paper must not interpret "
            "these results as pure policy optimality or as a complete safety evaluation.",
            "",
            "## Semantic warnings",
            "",
            "- A success/final-distance inconsistency is flagged when a record is marked successful "
            "but its final distance exceeds the declared 0.6 m threshold. This may indicate a "
            "different historical success rule or post-processing; it must be resolved from the "
            "runner before using success rate.",
            "- Repeated trajectory summaries across distinct encoders/samplers/CFG values indicate "
            "that the benchmark may be dominated by deterministic route/controller behavior. They "
            "are an audit signal, not proof of falsification.",
            "- Source-scoped rows are not pooled because duplicate configurations occur in multiple "
            "benchmark directories with incomplete provenance.",
            "",
            "## Figure availability",
            "",
            "`fig_mujoco_paths.pdf` was not generated. Existing raw records contain scalar path "
            "lengths but no trajectory coordinates, so a path figure would be fabricated. Supply "
            "time-indexed positions from the original run logs before drawing it.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(bench_root: Path, out_dir: Path, threshold: float) -> dict[str, Any]:
    rows, sources = load_records(bench_root, threshold)
    summaries = summarize(rows)
    audit = build_audit(rows, sources, threshold)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "mujoco_closed_loop_unified.csv", rows)
    write_table(out_dir / "table_mujoco_results.md", summaries)
    write_note(out_dir / "stabilizer_note.md", audit)
    (out_dir / "mujoco_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate existing MuJoCo JSON; this never launches simulation."
    )
    parser.add_argument("--bench", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--success-distance-threshold", type=float, default=0.6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audit = run(args.bench, args.out, args.success_distance_threshold)
    except DataError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        f"Wrote {audit['record_count']} records from {audit['source_count']} sources "
        f"to {args.out}"
    )
    print(
        "Audit warnings: "
        f"success-distance={audit['success_final_distance_inconsistency_count']}, "
        f"repeated-signatures={audit['repeated_trajectory_signature_count']}, "
        f"missing-stabilizer={audit['missing_optional_field_counts']['stabilizer_mode']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
