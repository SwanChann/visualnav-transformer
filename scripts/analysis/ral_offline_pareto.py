#!/usr/bin/env python3
"""Build the unified RA-L offline latency/quality analysis.

The script intentionally uses only the Python standard library.  The repository's
analysis environment is split across several conda environments, while this
aggregation should remain runnable without the training/deployment stack.

Quality proxy definition
------------------------
For every source experiment independently, case-level metrics are z-normalized
over all case/config records in that experiment.  A case score is then

    z(forward_progress) - z(smoothness) - z(lateral_abs)
    + 0.5 * z(diversity)

``invalid_or_saturation`` is absent from the source records, so it is reported as
NA and omitted.  Config scores and 95% confidence intervals are the mean and
normal-approximation CI over the case scores.  Pareto membership is computed
within each source experiment (minimize mean latency, maximize quality proxy),
because the six experiments use different sampled evaluation cases.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


NA = "NA"
METRIC_FIELDS = ("forward_progress", "smoothness", "lateral_abs", "diversity")
METRIC_WEIGHTS = {
    "forward_progress": 1.0,
    "smoothness": -1.0,
    "lateral_abs": -1.0,
    "diversity": 0.5,
}

OUTPUT_FIELDS = (
    "source_experiment",
    "comparison_group",
    "config_id",
    "encoder",
    "scheduler",
    "num_steps",
    "cfg_weight",
    "tts_budget",
    "tts_topk",
    "tts_verifier",
    "num_cases",
    "latency_mean_ms",
    "latency_ci95_ms",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p95_ms",
    "hz_mean",
    "forward_progress",
    "smoothness",
    "lateral_abs",
    "diversity",
    "endpoint_norm",
    "invalid_or_saturation",
    "quality_proxy",
    "quality_proxy_ci95",
    "quality_proxy_scope",
    "pareto_frontier",
    "budget_mode",
    "risk_note",
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    family: str
    summary_relpath: str
    cases_relpath: str
    key: Callable[[Mapping[str, str]], str]
    normalize: Callable[[Mapping[str, str]], dict[str, object]]


def as_float(value: object, default: float | None = None) -> float | None:
    if value is None or str(value).strip() in {"", NA}:
        return default
    try:
        return float(str(value))
    except ValueError:
        return default


def as_int(value: object, default: int = 0) -> int:
    number = as_float(value)
    return default if number is None else int(number)


def fkey(value: object) -> str:
    number = as_float(value, 0.0)
    assert number is not None
    return f"{number:.8g}"


def row_metric(row: Mapping[str, str], name: str) -> float | None:
    for field in (name, f"{name}_mean", f"{name}_mean_mean"):
        value = as_float(row.get(field))
        if value is not None:
            return value
    return None


def common_metrics(row: Mapping[str, str]) -> dict[str, object]:
    latency = as_float(row.get("latency_ms_mean"))
    return {
        "num_cases": as_int(row.get("num_cases")),
        "latency_mean_ms": latency,
        "latency_ci95_ms": as_float(row.get("latency_ms_ci95")),
        "forward_progress": row_metric(row, "forward_progress"),
        "smoothness": row_metric(row, "smoothness"),
        "lateral_abs": row_metric(row, "lateral_abs"),
        "diversity": row_metric(row, "diversity"),
        "endpoint_norm": row_metric(row, "endpoint_norm"),
        "invalid_or_saturation": None,
        "hz_mean": None if not latency or latency <= 0 else 1000.0 / latency,
    }


def normalize_ddim(row: Mapping[str, str]) -> dict[str, object]:
    config = row["config_name"]
    scheduler, steps = config.split("_", 1)
    return {
        "config_id": config,
        "encoder": "efficientnet_b0",
        "scheduler": scheduler,
        "num_steps": as_int(steps),
        "cfg_weight": 0.0,
        "tts_budget": 0,
        "tts_topk": 0,
        "tts_verifier": "none",
        **common_metrics(row),
    }


def normalize_cfg(row: Mapping[str, str]) -> dict[str, object]:
    cfg = as_float(row.get("guidance_scale"), 0.0)
    return {
        "config_id": f"ddpm10_cfg{fkey(cfg)}_tts0",
        "encoder": "efficientnet_b0",
        "scheduler": "ddpm",
        "num_steps": 10,
        "cfg_weight": cfg,
        "tts_budget": 0,
        "tts_topk": 0,
        "tts_verifier": "none",
        **common_metrics(row),
    }


def normalize_joint(row: Mapping[str, str]) -> dict[str, object]:
    cfg = as_float(row.get("guidance_scale"), 0.0)
    return {
        "config_id": row["config_name"],
        "encoder": "efficientnet_b0",
        "scheduler": row.get("scheduler", "unknown"),
        "num_steps": as_int(row.get("num_steps")),
        "cfg_weight": cfg,
        "tts_budget": 0,
        "tts_topk": 0,
        "tts_verifier": "none",
        **common_metrics(row),
    }


def normalize_tts(row: Mapping[str, str]) -> dict[str, object]:
    budget = as_int(row.get("tts_budget"))
    return {
        "config_id": row["config_name"],
        "encoder": "efficientnet_b0",
        "scheduler": row.get("scheduler_kind", "unknown"),
        "num_steps": as_int(row.get("num_steps")),
        "cfg_weight": as_float(row.get("cfg_weight"), 0.0),
        "tts_budget": budget,
        "tts_topk": as_int(row.get("tts_topk")) if budget else 0,
        "tts_verifier": row.get("tts_verifier", "unknown") if budget else "none",
        **common_metrics(row),
    }


def normalize_encoder(row: Mapping[str, str]) -> dict[str, object]:
    model = row["model_name"]
    return {
        "config_id": f"{model}_ddpm10_cfg0_tts0",
        "encoder": model,
        "scheduler": "ddpm",
        "num_steps": 10,
        "cfg_weight": 0.0,
        "tts_budget": 0,
        "tts_topk": 0,
        "tts_verifier": "none",
        **common_metrics(row),
    }


_JOINT_CONFIG_RE = re.compile(
    r"(?P<scheduler>ddim|ddpm)(?P<steps>\d+)_cfg(?P<cfg>[\dmp.]+)_tts(?P<tts>\d+)"
)


def normalize_encoder_joint(row: Mapping[str, str]) -> dict[str, object]:
    raw_config = row["config_name"]
    match = _JOINT_CONFIG_RE.fullmatch(raw_config)
    if not match:
        raise ValueError(f"Unrecognized encoder-joint config: {raw_config}")
    cfg_text = match.group("cfg").replace("p", ".").replace("m", "-")
    budget = int(match.group("tts"))
    experiment = row.get("experiment", "encoder_joint")
    model = row["model_name"]
    return {
        "config_id": f"{experiment}__{model}__{raw_config}",
        "encoder": model,
        "scheduler": match.group("scheduler"),
        "num_steps": int(match.group("steps")),
        "cfg_weight": float(cfg_text),
        "tts_budget": budget,
        "tts_topk": 4 if budget else 0,
        "tts_verifier": "heuristic" if budget else "none",
        **common_metrics(row),
    }


def key_field(field: str) -> Callable[[Mapping[str, str]], str]:
    return lambda row: row[field]


def key_cfg(row: Mapping[str, str]) -> str:
    return fkey(row.get("guidance_scale"))


def key_encoder_joint(row: Mapping[str, str]) -> str:
    return "|".join((row.get("experiment", ""), row["model_name"], row["config_name"]))


SOURCES = (
    SourceSpec(
        "ddim_sweep",
        "DDIM",
        "day2/20260410_143136_ddim_stat_experiment/ddim_overall_summary.csv",
        "day2/20260410_143136_ddim_stat_experiment/ddim_case_records.csv",
        key_field("config_name"),
        normalize_ddim,
    ),
    SourceSpec(
        "cfg_sweep",
        "CFG",
        "day3/20260410_143225_cfg_stat_experiment/cfg_overall_summary.csv",
        "day3/20260410_143225_cfg_stat_experiment/cfg_case_records.csv",
        key_cfg,
        normalize_cfg,
    ),
    SourceSpec(
        "ddim_cfg_joint",
        "DDIM x CFG",
        "day4/20260410_143709_joint_ddim_cfg_experiment/overall_summary.csv",
        "day4/20260410_143709_joint_ddim_cfg_experiment/case_records.csv",
        key_field("config_name"),
        normalize_joint,
    ),
    SourceSpec(
        "tts_budget",
        "TTS",
        "day4/20260421_140601_tts_stat_experiment/tts_overall_summary.csv",
        "day4/20260421_140601_tts_stat_experiment/tts_case_records.csv",
        key_field("config_name"),
        normalize_tts,
    ),
    SourceSpec(
        "encoder_comparison",
        "Encoder",
        "day4/20260421_151228_encoder_comparison_experiment/encoder_overall_summary.csv",
        "day4/20260421_151228_encoder_comparison_experiment/encoder_case_records.csv",
        key_field("model_name"),
        normalize_encoder,
    ),
    SourceSpec(
        "encoder_joint",
        "Encoder joint",
        "day5/20260422_171021_encoder_joint_ablation_experiment/joint_overall_summary.csv",
        "day5/20260422_171021_encoder_joint_ablation_experiment/joint_case_records.csv",
        key_encoder_joint,
        normalize_encoder_joint,
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values: Sequence[float], probability: float) -> float | None:
    """Linear (R-7/NumPy default) percentile without NumPy."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def source_quality_scores(
    case_rows: Sequence[Mapping[str, str]], key: Callable[[Mapping[str, str]], str]
) -> tuple[dict[str, tuple[float, float]], list[str]]:
    """Return config -> (mean proxy, CI95) using source-level case z-scores."""
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    omitted: list[str] = []
    for metric in METRIC_FIELDS:
        values = [row_metric(row, metric) for row in case_rows]
        present = [value for value in values if value is not None]
        if len(present) < 2:
            omitted.append(metric)
            continue
        scale = statistics.pstdev(present)
        if scale <= 0:
            omitted.append(metric)
            continue
        centers[metric] = statistics.fmean(present)
        scales[metric] = scale

    by_config: dict[str, list[float]] = defaultdict(list)
    for case in case_rows:
        terms: list[float] = []
        for metric, weight in METRIC_WEIGHTS.items():
            raw = row_metric(case, metric)
            if metric in scales and raw is not None:
                terms.append(weight * (raw - centers[metric]) / scales[metric])
        if terms:
            by_config[key(case)].append(sum(terms))

    output: dict[str, tuple[float, float]] = {}
    for config_key, scores in by_config.items():
        mean = statistics.fmean(scores)
        ci95 = 0.0
        if len(scores) > 1:
            ci95 = 1.96 * statistics.stdev(scores) / math.sqrt(len(scores))
        output[config_key] = (mean, ci95)
    return output, omitted


def assign_pareto(rows: list[dict[str, object]]) -> None:
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_experiment"])].append(row)
    for cohort in by_source.values():
        for candidate in cohort:
            latency = float(candidate["latency_mean_ms"])
            quality = float(candidate["quality_proxy"])
            dominated = any(
                float(other["latency_mean_ms"]) <= latency
                and float(other["quality_proxy"]) >= quality
                and (
                    float(other["latency_mean_ms"]) < latency
                    or float(other["quality_proxy"]) > quality
                )
                for other in cohort
                if other is not candidate
            )
            candidate["pareto_frontier"] = not dominated


def assign_budget_modes(rows: list[dict[str, object]]) -> None:
    for row in rows:
        scheduler = row["scheduler"]
        steps = as_int(row["num_steps"])
        cfg = float(row["cfg_weight"])
        tts = as_int(row["tts_budget"])
        frontier = bool(row["pareto_frontier"])
        if scheduler == "ddpm" and steps == 10 and cfg == 0.0 and tts == 0:
            mode = "Baseline"
        elif not frontier:
            mode = "Dominated"
        elif steps <= 2 and cfg == 0.0 and tts == 0:
            mode = "Fast"
        elif steps <= 3 and cfg == 0.0 and 0 < tts <= 16:
            mode = "Balanced"
        elif steps <= 3 and 0.0 < cfg <= 2.0 and 0 < tts <= 16:
            mode = "Quality"
        else:
            latency = float(row["latency_mean_ms"])
            mode = "Fast" if latency <= 10.0 else "Balanced" if latency <= 25.0 else "Quality"
        row["budget_mode"] = mode


def risk_note(source: str, omitted: Sequence[str]) -> str:
    parts = ["proxy normalized within source; Pareto not cross-experiment"]
    if omitted:
        parts.append("proxy omitted " + ",".join(omitted))
    parts.append("invalid/saturation unavailable")
    if source in {"tts_budget", "encoder_joint"}:
        parts.append("12-case subset")
    if source == "tts_budget":
        parts.append("verifier and proxy share trajectory-statistic terms")
    if source in {"encoder_comparison", "encoder_joint"}:
        parts.append("different trained encoder weights; not fixed-policy evidence")
    return "; ".join(parts)


def build_rows(results: Path) -> tuple[list[dict[str, object]], list[str]]:
    unified: list[dict[str, object]] = []
    audit: list[str] = []
    for source in SOURCES:
        summary_path = results / source.summary_relpath
        cases_path = results / source.cases_relpath
        if not summary_path.is_file() or not cases_path.is_file():
            raise FileNotFoundError(f"Missing source files for {source.name}: {summary_path}, {cases_path}")
        summaries = read_csv(summary_path)
        cases = read_csv(cases_path)
        case_groups: dict[str, list[Mapping[str, str]]] = defaultdict(list)
        for case in cases:
            case_groups[source.key(case)].append(case)
        quality, omitted = source_quality_scores(cases, source.key)

        mismatch_count = 0
        for summary in summaries:
            key = source.key(summary)
            row = source.normalize(summary)
            if key not in quality:
                raise ValueError(f"No case-level quality scores for {source.name}:{key}")
            expected_cases = as_int(row["num_cases"])
            if len(case_groups[key]) != expected_cases:
                mismatch_count += 1
            latencies = [
                value
                for value in (as_float(case.get("latency_ms")) for case in case_groups[key])
                if value is not None
            ]
            proxy, proxy_ci = quality[key]
            row.update(
                {
                    "source_experiment": source.name,
                    "comparison_group": source.name,
                    "family": source.family,
                    "latency_p50_ms": percentile(latencies, 0.50),
                    "latency_p90_ms": percentile(latencies, 0.90),
                    "latency_p95_ms": percentile(latencies, 0.95),
                    "quality_proxy": proxy,
                    "quality_proxy_ci95": proxy_ci,
                    "quality_proxy_scope": "within_source_case_zscore",
                    "pareto_frontier": False,
                    "budget_mode": "",
                    "risk_note": risk_note(source.name, omitted),
                }
            )
            unified.append(row)
        audit.append(
            f"{source.name}: {len(summaries)} configs, {len(cases)} case records, "
            f"{mismatch_count} config count mismatches"
        )
    assign_pareto(unified)
    assign_budget_modes(unified)
    unified.sort(key=lambda row: (str(row["source_experiment"]), float(row["latency_mean_ms"])))
    return unified, audit


def format_csv_value(field: str, value: object) -> object:
    if value is None:
        return NA
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def write_unified_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_csv_value(field, row.get(field)) for field in OUTPUT_FIELDS})


def fmt(value: object, digits: int = 2) -> str:
    if value is None or value == NA:
        return NA
    return f"{float(value):.{digits}f}"


def display_config(row: Mapping[str, object]) -> str:
    config = str(row["config_id"])
    return config.replace("__", " / ").replace("_", "\\_")


def select_representatives(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    # Table II should compare like with like.  The main rows therefore come from
    # the single TTS experiment (same 12 sampled cases), instead of selecting the
    # largest source-relative z-score across incompatible experiments.  Encoder
    # variants are kept in the unified/supplementary table but excluded here
    # because they use separately trained weights and violate the fixed-policy
    # framing.  One CFG=4 row is included as a strong-guidance negative control.
    requested = (
        ("Baseline", "tts_budget", "ddpm10_cfg0.0_tts0"),
        ("Fast", "tts_budget", "ddim2_cfg0.0_tts0"),
        ("Fast", "tts_budget", "ddim3_cfg0.0_tts0"),
        ("Balanced", "tts_budget", "ddim2_cfg0.0_tts8"),
        ("Balanced", "tts_budget", "ddim2_cfg0.0_tts16"),
        ("Balanced", "tts_budget", "ddim3_cfg0.0_tts8"),
        ("Quality", "tts_budget", "ddim2_cfg1.0_tts8"),
        ("Quality", "tts_budget", "ddim2_cfg2.0_tts8"),
        ("Quality", "tts_budget", "ddim2_cfg2.0_tts16"),
        ("Dominated", "cfg_sweep", "ddpm10_cfg4_tts0"),
        ("Dominated", "tts_budget", "ddim10_cfg1.0_tts32"),
    )
    index = {(str(row["source_experiment"]), str(row["config_id"])): row for row in rows}
    selected: list[dict[str, object]] = []
    for table_mode, source, config in requested:
        try:
            row = dict(index[(source, config)])
        except KeyError as error:
            raise ValueError(f"Missing requested Table II row: {source}:{config}") from error
        row["budget_mode"] = table_mode
        selected.append(row)
    return selected


def markdown_table(rows: Sequence[dict[str, object]]) -> str:
    lines = [
        "# Table II — Representative Offline Latency–Quality Configurations",
        "",
        "Pareto flags and quality proxy values are valid within each source experiment only; "
        "the experiments used different sampled case sets.",
        "",
        "| Mode | Source | Configuration | Mean / case-p95 latency (ms) | Hz | Forward | Lateral | Smoothness | Diversity | Quality proxy ± CI95 | Pareto |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {source} | {config} | {mean} / {p95} | {hz} | {forward} | "
            "{lateral} | {smooth} | {diversity} | {quality} ± {quality_ci} | {pareto} |".format(
                mode=row["budget_mode"],
                source=row["source_experiment"],
                config=display_config(row),
                mean=fmt(row["latency_mean_ms"]),
                p95=fmt(row["latency_p95_ms"]),
                hz=fmt(row["hz_mean"], 1),
                forward=fmt(row["forward_progress"], 3),
                lateral=fmt(row["lateral_abs"], 3),
                smooth=fmt(row["smoothness"], 3),
                diversity=fmt(row["diversity"], 3),
                quality=fmt(row["quality_proxy"], 3),
                quality_ci=fmt(row["quality_proxy_ci95"], 3),
                pareto="yes" if row["pareto_frontier"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "`Baseline/Fast/Balanced/Quality` are descriptive latency-budget labels, not claims of global optimality. "
            "The two dominated rows are retained as negative controls.",
        ]
    )
    return "\n".join(lines) + "\n"


def budget_modes_table(rows: Sequence[dict[str, object]]) -> str:
    reps = select_representatives(rows)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in reps:
        grouped[str(row["budget_mode"])].append(row)
    lines = [
        "# Deployment Budget Modes",
        "",
        "These are reproducible candidates for paper discussion. Closed-loop validation is still required before a mode can be called a deployment recommendation.",
        "",
        "| Mode | Operational interpretation | Representative candidates | Evidence limitation |",
        "|---|---|---|---|",
    ]
    descriptions = {
        "Baseline": "Original DDPM-10 reference cost",
        "Fast": "Low-step DDIM, CFG=0, no candidate expansion",
        "Balanced": "Low-step DDIM with heuristic selection/modest candidate budget",
        "Quality": "Low-step DDIM with moderate CFG/candidate budget",
        "Dominated": "High-budget/strong-guidance negative control",
    }
    for mode in ("Baseline", "Fast", "Balanced", "Quality", "Dominated"):
        candidates = grouped.get(mode, [])
        labels = "; ".join(
            f"{row['source_experiment']}:{str(row['config_id']).replace('__', '/')} ({fmt(row['latency_mean_ms'], 1)} ms)"
            for row in candidates
        ) or "NA"
        limitation = (
            "reference only"
            if mode == "Baseline"
            else "offline proxy; source-specific normalization and case set"
        )
        lines.append(f"| {mode} | {descriptions[mode]} | {labels} | {limitation} |")
    return "\n".join(lines) + "\n"


class SimplePDF:
    """Tiny vector PDF writer sufficient for reproducible scientific scatter plots."""

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.commands: list[str] = []

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def text(self, x: float, y: float, text: str, size: float = 8.0) -> None:
        clean = text.encode("ascii", "replace").decode("ascii")
        self.commands.append(f"BT /F1 {size:.2f} Tf {x:.2f} {y:.2f} Td ({self._escape(clean)}) Tj ET")

    def text_rotated(self, x: float, y: float, text: str, size: float = 8.0) -> None:
        clean = text.encode("ascii", "replace").decode("ascii")
        self.commands.append(
            f"BT /F1 {size:.2f} Tf 0 1 -1 0 {x:.2f} {y:.2f} Tm ({self._escape(clean)}) Tj ET"
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, width: float = 0.6) -> None:
        self.commands.append(f"{width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def color(self, rgb: tuple[float, float, float]) -> None:
        self.commands.append(f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} RG {rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} rg")

    def circle(self, x: float, y: float, radius: float = 2.2, fill: bool = True) -> None:
        k = 0.5522847498 * radius
        op = "B" if fill else "S"
        self.commands.append(
            f"{x+radius:.2f} {y:.2f} m {x+radius:.2f} {y+k:.2f} {x+k:.2f} {y+radius:.2f} {x:.2f} {y+radius:.2f} c "
            f"{x-k:.2f} {y+radius:.2f} {x-radius:.2f} {y+k:.2f} {x-radius:.2f} {y:.2f} c "
            f"{x-radius:.2f} {y-k:.2f} {x-k:.2f} {y-radius:.2f} {x:.2f} {y-radius:.2f} c "
            f"{x+k:.2f} {y-radius:.2f} {x+radius:.2f} {y-k:.2f} {x+radius:.2f} {y:.2f} c {op}"
        )

    def save(self, path: Path) -> None:
        stream = ("\n".join(self.commands) + "\n").encode("ascii")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.2f} {self.height:.2f}] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode(),
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
        payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(payload))
            payload.extend(f"{index} 0 obj\n".encode())
            payload.extend(obj)
            payload.extend(b"\nendobj\n")
        xref = len(payload)
        payload.extend(f"xref\n0 {len(objects)+1}\n".encode())
        payload.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            payload.extend(f"{offset:010d} 00000 n \n".encode())
        payload.extend(
            f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        )
        path.write_bytes(payload)


FAMILY_COLORS = {
    "DDIM": (0.12, 0.47, 0.71),
    "CFG": (1.00, 0.50, 0.05),
    "DDIM x CFG": (0.17, 0.63, 0.17),
    "TTS": (0.84, 0.15, 0.16),
    "Encoder": (0.58, 0.40, 0.74),
    "Encoder joint": (0.55, 0.34, 0.29),
}


def nice_range(values: Sequence[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return low - 1.0, high + 1.0
    margin = 0.06 * (high - low)
    return low - margin, high + margin


def draw_panel(
    pdf: SimplePDF,
    rows: Sequence[dict[str, object]],
    box: tuple[float, float, float, float],
    xfield: str,
    yfield: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    x0, y0, width, height = box
    xs = [float(row[xfield]) for row in rows if row.get(xfield) is not None]
    ys = [float(row[yfield]) for row in rows if row.get(yfield) is not None]
    xmin, xmax = nice_range(xs)
    ymin, ymax = nice_range(ys)
    if "latency" in xfield:
        xmin = max(0.0, xmin)
    plot_x, plot_y = x0 + 43, y0 + 31
    plot_w, plot_h = width - 52, height - 52
    pdf.color((0.0, 0.0, 0.0))
    pdf.line(plot_x, plot_y, plot_x + plot_w, plot_y)
    pdf.line(plot_x, plot_y, plot_x, plot_y + plot_h)
    for index in range(5):
        fraction = index / 4
        x = plot_x + fraction * plot_w
        y = plot_y + fraction * plot_h
        pdf.line(x, plot_y - 2, x, plot_y + 2, 0.4)
        pdf.line(plot_x - 2, y, plot_x + 2, y, 0.4)
        pdf.text(x - 8, plot_y - 13, f"{xmin + fraction*(xmax-xmin):.1f}", 6.5)
        pdf.text(plot_x - 35, y - 2, f"{ymin + fraction*(ymax-ymin):.2f}", 6.5)
    pdf.text(x0 + width / 2 - 28, y0 + 8, xlabel, 7.5)
    pdf.text_rotated(x0 + 11, y0 + height / 2 - 28, ylabel, 7.5)
    pdf.text(x0 + 5, y0 + height - 10, title, 9.0)
    for row in rows:
        if row.get(xfield) is None or row.get(yfield) is None:
            continue
        x = plot_x + (float(row[xfield]) - xmin) / (xmax - xmin) * plot_w
        y = plot_y + (float(row[yfield]) - ymin) / (ymax - ymin) * plot_h
        pdf.color(FAMILY_COLORS[str(row["family"])])
        pdf.circle(x, y, 2.4, fill=True)
        if row["pareto_frontier"]:
            pdf.color((0.0, 0.0, 0.0))
            pdf.circle(x, y, 3.5, fill=False)


def write_pareto_figure(rows: Sequence[dict[str, object]], path: Path) -> None:
    pdf = SimplePDF(792, 310)
    pdf.text(18, 294, "Fig. 2. Offline inference trade-offs (rings: within-experiment Pareto points)", 11)
    boxes = ((10, 35, 255, 245), (268, 35, 255, 245), (526, 35, 255, 245))
    draw_panel(pdf, rows, boxes[0], "latency_mean_ms", "quality_proxy", "mean latency (ms)", "quality proxy", "(a) Mean latency")
    draw_panel(pdf, rows, boxes[1], "latency_p95_ms", "quality_proxy", "p95 case-mean latency (ms)", "quality proxy", "(b) Across-case p95")
    draw_panel(pdf, rows, boxes[2], "latency_mean_ms", "forward_progress", "mean latency (ms)", "forward progress", "(c) Raw component")
    legend_x = 18
    for family, color in FAMILY_COLORS.items():
        pdf.color(color)
        pdf.circle(legend_x, 20, 2.5, True)
        pdf.color((0.0, 0.0, 0.0))
        pdf.text(legend_x + 5, 17, family, 6.8)
        legend_x += 40 + 5.0 * len(family)
    pdf.save(path)


def write_tts_figure(rows: Sequence[dict[str, object]], path: Path) -> None:
    tts_rows = [
        row
        for row in rows
        if row["source_experiment"] == "tts_budget"
        and row["scheduler"] == "ddim"
        and as_int(row["num_steps"]) == 2
    ]
    pdf = SimplePDF(600, 310)
    pdf.text(18, 294, "TTS candidate-budget marginal effects (DDIM-2)", 11)
    boxes = ((15, 40, 280, 235), (305, 40, 280, 235))
    colors = {0.0: (0.12, 0.47, 0.71), 0.5: (1.0, 0.5, 0.05), 1.0: (0.17, 0.63, 0.17), 2.0: (0.84, 0.15, 0.16)}

    def panel(box: tuple[float, float, float, float], field: str, ylabel: str, title: str) -> None:
        x0, y0, width, height = box
        ymin, ymax = nice_range([float(row[field]) for row in tts_rows])
        plot_x, plot_y, plot_w, plot_h = x0 + 42, y0 + 30, width - 52, height - 50
        pdf.color((0, 0, 0))
        pdf.line(plot_x, plot_y, plot_x + plot_w, plot_y)
        pdf.line(plot_x, plot_y, plot_x, plot_y + plot_h)
        budgets = [0, 8, 16, 32]
        for index, budget in enumerate(budgets):
            x = plot_x + index / 3 * plot_w
            pdf.text(x - 4, plot_y - 13, str(budget), 7)
        for index in range(5):
            fraction = index / 4
            y = plot_y + fraction * plot_h
            pdf.text(plot_x - 35, y - 2, f"{ymin + fraction*(ymax-ymin):.2f}", 6.5)
        pdf.text(x0 + width / 2 - 48, y0 + 7, "TTS setting (0=unfiltered)", 7.5)
        pdf.text_rotated(x0 + 11, y0 + height / 2 - 28, ylabel, 7.5)
        pdf.text(x0 + 5, y0 + height - 10, title, 9)
        for cfg in (0.0, 0.5, 1.0, 2.0):
            series = sorted(
                [row for row in tts_rows if math.isclose(float(row["cfg_weight"]), cfg)],
                key=lambda row: as_int(row["tts_budget"]),
            )
            points: list[tuple[float, float]] = []
            for row in series:
                x = plot_x + budgets.index(as_int(row["tts_budget"])) / 3 * plot_w
                y = plot_y + (float(row[field]) - ymin) / (ymax - ymin) * plot_h
                points.append((x, y))
            pdf.color(colors[cfg])
            for first, second in zip(points, points[1:]):
                pdf.line(first[0], first[1], second[0], second[1], 1.0)
            for x, y in points:
                pdf.circle(x, y, 2.5, True)

    panel(boxes[0], "forward_progress", "forward progress", "(a) Candidate quality component")
    panel(boxes[1], "latency_mean_ms", "mean latency (ms)", "(b) Compute cost")
    x = 40
    for cfg, color in colors.items():
        pdf.color(color)
        pdf.circle(x, 20, 2.5, True)
        pdf.color((0, 0, 0))
        pdf.text(x + 5, 17, f"CFG={cfg:g}", 7)
        x += 95
    pdf.save(path)


def metric_notes(rows: Sequence[dict[str, object]], audit: Sequence[str], results: Path) -> str:
    frontier_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["pareto_frontier"]:
            frontier_counts[str(row["source_experiment"])] += 1
    sources = "\n".join(
        f"- `{source.name}`: `{results / source.summary_relpath}` + `{results / source.cases_relpath}`"
        for source in SOURCES
    )
    audit_lines = "\n".join(f"- {item}; Pareto points: {frontier_counts[item.split(':', 1)[0]]}" for item in audit)
    return f"""# Offline Metric Notes

## Scope and source data

This analysis unifies {len(rows)} configurations from six existing offline experiments. It does not run the navigation policy again.

{sources}

## Raw fields

- `num_cases`, mean latency and its CI95, forward progress, smoothness, absolute lateral motion, diversity, and endpoint norm come from each source summary CSV.
- p50/p90/p95 latency are recomputed from the matching case-record `latency_ms` values with linear (R-7) interpolation. Each case-record value is already a mean over 3 or 8 inference runs, so these are percentiles **across case-level means**, not per-call tail-latency percentiles.
- Encoder comparison is recorded as DDPM-10/CFG-0/TTS-0 because that experiment script defaults to DDPM and the observed run uses the default; CFG sweep uses DDPM-10 as explicitly defined by `cfg_stat_experiment.py`.

## Derived fields

- `hz_mean = 1000 / latency_mean_ms`; this is an inference-only upper bound, not full perception-control loop frequency.
- Within each source experiment, every case record is z-normalized using the mean and population standard deviation over all case/config records in that source.
- `quality_proxy = z(forward_progress) - z(smoothness) - z(lateral_abs) + 0.5*z(diversity)`.
- `quality_proxy_ci95` is `1.96 * sample_std(case_proxy) / sqrt(num_cases)` and therefore includes covariance between component metrics at case level.
- `pareto_frontier` minimizes mean latency and maximizes quality proxy **within the same source experiment**. Cross-experiment dominance is deliberately not claimed because the sampled case sets and collection dates differ.
- Budget modes are descriptive rather than globally optimized: Baseline is DDPM-10/CFG-0/TTS-0; Fast prioritizes low-step CFG-0 sampling without candidate expansion; Balanced adds a modest candidate budget; Quality adds moderate guidance/candidate budget. Non-frontier rows are marked Dominated.

## Missing fields and limitations

- `invalid_or_saturation` is unavailable in all six source datasets and is written as `NA`; its penalty is omitted from the proxy.
- The proxy is a transparent trajectory-statistics summary, **not navigation success rate**, collision rate, safety, or real-world utility. Higher diversity is not universally beneficial, and forward progress can reward unsafe straight motion.
- TTS uses a heuristic verifier containing forward progress, lateral deviation, smoothness, and path efficiency, while the reported proxy reuses three of those terms. TTS proxy improvements are therefore partly true **by construction** and are not independent evidence that navigation improved.
- `TTS=0` still samples the standard batch of 8 NoMaD trajectories without heuristic top-k selection. `TTS=8` ranks that same-size batch (top-4 here); only budgets 16/32 expand generation beyond the standard batch.
- The 12-case TTS and encoder-joint experiments are not directly interchangeable with the 24-case sweeps.
- All offline cases come from `go_stanford`; the results do not establish cross-dataset or outdoor generalization.
- Timings are historical GPU inference measurements. They do not include image transport, preprocessing, robot communication, low-level control, or current Lite3 on-board hardware latency.
- Normal-approximation CIs are descriptive for the sampled cases; they are not independent repeated-training confidence intervals.

## Integrity audit

{audit_lines}

## Reproduction

```powershell
python scripts/analysis/ral_offline_pareto.py --results results --out 投稿冲刺/workspace/offline_pareto
```

The script uses only the Python standard library and writes vector PDF directly, so it does not depend on the repository's currently inconsistent NumPy/pandas/matplotlib environment.
"""


def verify_pdf(path: Path) -> None:
    data = path.read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-64:]:
        raise ValueError(f"Invalid PDF structure: {path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"), help="Results root")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = args.results.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, audit = build_rows(results)
    representatives = select_representatives(rows)

    csv_path = output / "offline_pareto_unified.csv"
    table_path = output / "table_offline_pareto.md"
    modes_path = output / "table_budget_modes.md"
    figure_path = output / "fig_offline_pareto.pdf"
    tts_figure_path = output / "fig_tts_marginal_gain.pdf"
    notes_path = output / "offline_metric_notes.md"

    write_unified_csv(rows, csv_path)
    table_path.write_text(markdown_table(representatives), encoding="utf-8")
    modes_path.write_text(budget_modes_table(rows), encoding="utf-8")
    write_pareto_figure(rows, figure_path)
    write_tts_figure(rows, tts_figure_path)
    notes_path.write_text(metric_notes(rows, audit, args.results), encoding="utf-8")
    verify_pdf(figure_path)
    verify_pdf(tts_figure_path)

    print(f"Wrote {len(rows)} unified configurations to {output}")
    for item in audit:
        print(f"  {item}")
    print(f"Selected {len(representatives)} representative Table II rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
