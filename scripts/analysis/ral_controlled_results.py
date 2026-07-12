#!/usr/bin/env python3
"""Analyze one immutable controlled MuJoCo run without historical-result mixing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import beta


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "scripts" / "experiments"
for candidate in (Path(__file__).resolve().parent, EXPERIMENTS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import ral_mujoco_controlled as runner
import validate_ral_mujoco_trial as trial_validator


BASELINE_ID = "ddpm10_cfg0_standard_k8"
CONFIG_ORDER = [
    BASELINE_ID,
    "ddim2_cfg0_standard_k8",
    "ddim3_cfg0_standard_k8",
    "ddim2_cfg0_tts8",
    "ddim2_cfg2_tts8",
]
STRATUM_ORDER = ["policy_only_off", "system_stabilizer_on"]


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), probability * 100.0))


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> list[float | None]:
    if total <= 0:
        return [None, None]
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return [lower, upper]


def mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def rank_average(values: Sequence[float], higher_is_better: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order_values = -array if higher_is_better else array
    order = np.argsort(order_values, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and order_values[order[end]] == order_values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_with_exact_permutation(x: Sequence[float], y: Sequence[float], x_higher: bool, y_higher: bool) -> dict[str, Any]:
    if len(x) != len(y) or len(x) < 3:
        return {"rho": None, "p_exact": None, "degenerate": True, "reason": "insufficient_items"}
    rx = rank_average(x, x_higher)
    ry = rank_average(y, y_higher)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return {"rho": None, "p_exact": None, "degenerate": True, "reason": "tied_ranking"}
    rho = float(np.corrcoef(rx, ry)[0, 1])
    permuted = []
    for permutation in itertools.permutations(ry.tolist()):
        permuted.append(float(np.corrcoef(rx, np.asarray(permutation))[0, 1]))
    p_value = float(sum(abs(value) >= abs(rho) - 1e-12 for value in permuted) / len(permuted))
    return {"rho": rho, "p_exact": p_value, "degenerate": False, "reason": None, "permutation_n": len(permuted)}


def _trial_path(run_dir: Path, value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    allowed = (run_dir / "trials").resolve()
    if path.parent != allowed:
        raise ValueError(f"Trial record escapes controlled run: {path}")
    return path


def load_primary_trials(run_dir: Path, strict: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((run_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    manifest_report = trial_validator.validate_batch(manifest)
    if not manifest_report["passed"]:
        raise ValueError(f"Invalid batch manifest: {manifest_report['errors']}")
    records = []
    for relative in manifest["trial_records"]:
        path = _trial_path(run_dir, relative)
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = trial_validator.validate_trial(payload)
        if not report["passed"]:
            raise ValueError(f"Invalid trial {path}: {report['errors']}")
        payload["_source_path"] = str(path.relative_to(REPO_ROOT))
        records.append(payload)
    if strict and manifest["missing_trial_keys"]:
        raise ValueError(f"Controlled matrix has missing trials: {manifest['missing_trial_keys']}")
    if len(records) + len(manifest["missing_trial_keys"]) != manifest["planned_trial_count"]:
        raise ValueError("Primary trial count does not reconcile with the manifest.")
    return manifest, records


def episode_row(record: dict[str, Any]) -> dict[str, Any]:
    outcome = record["outcome"]
    timing = record["timing"]
    warmup = int(timing["warmup_calls"])
    sampler = [float(value) for value in timing["sampler_per_call_ms"]][warmup:]
    full_loop = [float(value) for value in timing["full_loop_per_call_ms"]][warmup:]
    path_length = float(outcome["path_length_m"])
    shortest = float(outcome["shortest_path_m"])
    scientific = record["status"] != "infrastructure_invalid"
    trajectory = record["trajectory"]
    return {
        "run_id": record["run_id"], "protocol_sha256": record["inputs"]["protocol_sha256"],
        "trial_key": record["trial_key"], "trial_id": record["trial_id"], "attempt": record["attempt"], "status": record["status"],
        "config_id": record["trial_key"].split("__", 1)[0], "scheduler": record["policy"]["scheduler"], "diffusion_steps": record["policy"]["diffusion_steps"],
        "cfg_weight": record["policy"]["cfg_weight"], "tts_enabled": record["policy"]["tts_enabled"], "selection": record["policy"]["selection"], "candidate_k": record["policy"]["candidate_k"],
        "scene_id": record["episode"]["map"], "goal_seed": record["episode"]["goal_seed"], "diffusion_seed": record["episode"]["diffusion_seed"], "stabilizer_mode": record["episode"]["stabilizer_mode"],
        "success_radius_m": record["episode"]["success_radius_m"], "max_steps": record["episode"]["max_steps"], "simulated_timeout_s": record["episode"]["simulated_timeout_s"],
        "success": bool(outcome["success"]), "final_distance_m": outcome["final_distance_m"], "path_length_m": path_length, "shortest_path_m": shortest,
        "spl": float(outcome["spl"]), "path_efficiency": min(shortest / max(path_length, 1e-12), 1.0),
        "collision": bool(outcome["collision"]), "fall": bool(outcome["fall"]), "stuck": bool(outcome["stuck"]), "timeout": bool(outcome["timeout"]), "intervention": bool(outcome["intervention"]),
        "recovery_count": int(outcome["recovery_count"]), "scientific_denominator": scientific, "infrastructure_invalid": not scientific, "crash": record["status"] == "crashed",
        "sampler_mean_ms": mean_or_none(sampler), "sampler_p95_ms": percentile(sampler, 0.95), "full_loop_mean_ms": mean_or_none(full_loop), "full_loop_p95_ms": percentile(full_loop, 0.95),
        "loop_hz_mean": None if not full_loop else 1000.0 / float(np.mean(full_loop)), "trajectory_n": len(trajectory), "source_trial_json": record["_source_path"], "git_commit": record["git"]["commit"],
    }


def reconcile_rows(rows: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    statuses = {name: sum(row["status"] == name for row in rows) for name in ("completed", "failed", "crashed", "infrastructure_invalid")}
    errors = []
    for name, count in statuses.items():
        if count != int(manifest[name]):
            errors.append(f"{name}: rows={count} manifest={manifest[name]}")
    if len(rows) + len(manifest["missing_trial_keys"]) != int(manifest["planned_trial_count"]):
        errors.append("planned count mismatch")
    return {
        "passed": not errors, "errors": errors, "planned_n": manifest["planned_trial_count"], "recorded_n": len(rows),
        "scientific_n": sum(row["scientific_denominator"] for row in rows), "missing_n": len(manifest["missing_trial_keys"]), **statuses,
    }


def paired_cluster_bootstrap(rows: Sequence[dict[str, Any]], metric: str, stratum: str, replicates: int, seed: int) -> dict[str, Any]:
    bucket = [row for row in rows if row["stabilizer_mode"] == stratum and row["scientific_denominator"]]
    cluster_keys = sorted({(row["scene_id"], row["goal_seed"], row["diffusion_seed"]) for row in bucket})
    by_key_config = {(row["scene_id"], row["goal_seed"], row["diffusion_seed"], row["config_id"]): row for row in bucket}
    rng = np.random.default_rng(seed)
    output = {}
    for config_id in CONFIG_ORDER:
        observed = [float(by_key_config[key + (config_id,)][metric]) for key in cluster_keys if key + (config_id,) in by_key_config and by_key_config[key + (config_id,)][metric] is not None]
        samples = []
        if cluster_keys and len(observed) == len(cluster_keys):
            for _ in range(replicates):
                selected = rng.integers(0, len(cluster_keys), size=len(cluster_keys))
                values = [float(by_key_config[cluster_keys[index] + (config_id,)][metric]) for index in selected]
                samples.append(float(np.mean(values)))
        output[config_id] = {"mean": mean_or_none(observed), "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)], "cluster_n": len(cluster_keys), "replicates": replicates}
    return output


def paired_delta_bootstrap(rows: Sequence[dict[str, Any]], metric: str, stratum: str, replicates: int, seed: int) -> dict[str, Any]:
    bucket = [row for row in rows if row["stabilizer_mode"] == stratum and row["scientific_denominator"]]
    cluster_keys = sorted({(row["scene_id"], row["goal_seed"], row["diffusion_seed"]) for row in bucket})
    by_key_config = {(row["scene_id"], row["goal_seed"], row["diffusion_seed"], row["config_id"]): row for row in bucket}
    rng = np.random.default_rng(seed)
    output = {}
    for config_id in CONFIG_ORDER:
        deltas = []
        valid_keys = []
        for key in cluster_keys:
            candidate = by_key_config.get(key + (config_id,))
            baseline = by_key_config.get(key + (BASELINE_ID,))
            if candidate is not None and baseline is not None and candidate[metric] is not None and baseline[metric] is not None:
                valid_keys.append(key)
                deltas.append(float(candidate[metric]) - float(baseline[metric]))
        samples = []
        if valid_keys:
            for _ in range(replicates):
                selected = rng.integers(0, len(deltas), size=len(deltas))
                samples.append(float(np.mean([deltas[index] for index in selected])))
        output[config_id] = {"mean_delta_vs_ddpm10": mean_or_none(deltas), "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)], "paired_cluster_n": len(valid_keys), "replicates": replicates, "degenerate_all_zero": bool(deltas) and all(abs(value) <= 1e-12 for value in deltas)}
    return output


def summarize(rows: Sequence[dict[str, Any]], replicates: int, seed: int) -> list[dict[str, Any]]:
    bootstraps = {stratum: {metric: paired_cluster_bootstrap(rows, metric, stratum, replicates, seed + index) for index, metric in enumerate(("success", "spl", "final_distance_m", "path_efficiency"))} for stratum in STRATUM_ORDER}
    summaries = []
    for stratum in STRATUM_ORDER:
        for config_id in CONFIG_ORDER:
            bucket = [row for row in rows if row["stabilizer_mode"] == stratum and row["config_id"] == config_id]
            scientific = [row for row in bucket if row["scientific_denominator"]]
            success_n = sum(row["success"] for row in scientific)
            scientific_n = len(scientific)
            item = {
                "config_id": config_id, "stabilizer_mode": stratum, "planned_n": len(bucket), "scientific_n": scientific_n,
                "infrastructure_invalid_n": sum(row["infrastructure_invalid"] for row in bucket), "crash_n": sum(row["crash"] for row in bucket),
                "success_n": success_n, "success_rate": success_n / scientific_n if scientific_n else None, "success_clopper_pearson_95": clopper_pearson(success_n, scientific_n),
                "success_bootstrap_95": bootstraps[stratum]["success"][config_id]["ci95"],
                "spl_bootstrap_95": bootstraps[stratum]["spl"][config_id]["ci95"],
                "final_distance_bootstrap_95": bootstraps[stratum]["final_distance_m"][config_id]["ci95"],
                "path_efficiency_bootstrap_95": bootstraps[stratum]["path_efficiency"][config_id]["ci95"],
            }
            for metric in ("spl", "final_distance_m", "path_efficiency", "sampler_p95_ms", "full_loop_p95_ms", "loop_hz_mean"):
                values = [float(row[metric]) for row in scientific if row[metric] is not None]
                item[f"{metric}_mean"] = mean_or_none(values)
            for event in ("collision", "fall", "stuck", "timeout", "intervention"):
                item[f"{event}_rate"] = sum(row[event] for row in scientific) / scientific_n if scientific_n else None
            summaries.append(item)
    return summaries


def offline_closed_loop_correlation(summary: Sequence[dict[str, Any]], offline_summary: Path) -> dict[str, Any]:
    offline_payload = json.loads(offline_summary.read_text(encoding="utf-8"))
    offline = {row["config_id"]: row for row in offline_payload["summary"]}
    output = {
        "source": {
            "summary_path": str(offline_summary.resolve().relative_to(REPO_ROOT)),
            "summary_sha256": hashlib.sha256(offline_summary.read_bytes()).hexdigest(),
            "run_id": offline_payload.get("run_id"),
            "metric_waypoint_spacing_m": offline_payload.get("metric_waypoint_spacing_m"),
            "data_config_sha256": offline_payload.get("data_config_sha256"),
        }
    }
    for stratum in STRATUM_ORDER:
        closed = {row["config_id"]: row for row in summary if row["stabilizer_mode"] == stratum}
        configs = [config for config in CONFIG_ORDER if config in offline and config in closed]
        output[stratum] = {
            "config_ids": configs,
            "ade_vs_success": spearman_with_exact_permutation([offline[c]["action_ade_m_mean"] for c in configs], [closed[c]["success_rate"] for c in configs], False, True),
            "ade_vs_spl": spearman_with_exact_permutation([offline[c]["action_ade_m_mean"] for c in configs], [closed[c]["spl_mean"] for c in configs], False, True),
            "sampler_latency_vs_full_loop": spearman_with_exact_permutation([offline[c]["sampler_latency_ms_mean"] for c in configs], [closed[c]["full_loop_p95_ms_mean"] for c in configs], False, False),
        }
    return output


def robustness(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    radii = (0.4, 0.5, 0.6, 0.75)
    timeout_factors = (0.75, 1.0, 1.25)
    output = []
    for stratum in STRATUM_ORDER:
        for config_id in CONFIG_ORDER:
            bucket = [row for row in rows if row["stabilizer_mode"] == stratum and row["config_id"] == config_id and row["scientific_denominator"]]
            source_records = [json.loads((REPO_ROOT / row["source_trial_json"]).read_text(encoding="utf-8")) for row in bucket]
            for radius in radii:
                for factor in timeout_factors:
                    successes = 0
                    for record in source_records:
                        if radius == float(record["episode"]["success_radius_m"]) and factor == 1.0:
                            reached = bool(record["outcome"]["success"])
                        else:
                            goal = np.asarray(record["episode"]["goal_xy_m"], dtype=float)
                            max_tick = int(math.floor(record["episode"]["max_steps"] * factor))
                            reached = any(point["tick"] < max_tick and float(np.linalg.norm(np.asarray([point["x_m"], point["y_m"]]) - goal)) < radius for point in record["trajectory"])
                            reached = bool(reached and not record["outcome"]["fall"])
                        successes += reached
                    output.append({"config_id": config_id, "stabilizer_mode": stratum, "radius_m": radius, "timeout_factor": factor, "secondary_analysis": True, "primary_condition_reuses_terminal_outcome": radius == 0.5 and factor == 1.0, "nonprimary_condition_is_trajectory_approximation": not (radius == 0.5 and factor == 1.0), "success_n": successes, "trial_n": len(source_records), "success_rate": successes / len(source_records) if source_records else None})
    return output


def trajectory_equivalence(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for stratum in STRATUM_ORDER:
        groups = []
        keys = sorted({(row["scene_id"], row["goal_seed"], row["diffusion_seed"]) for row in rows if row["stabilizer_mode"] == stratum})
        for key in keys:
            bucket = [row for row in rows if row["stabilizer_mode"] == stratum and (row["scene_id"], row["goal_seed"], row["diffusion_seed"]) == key]
            hashes = {}
            for row in bucket:
                record = json.loads((REPO_ROOT / row["source_trial_json"]).read_text(encoding="utf-8"))
                encoded = json.dumps(record["trajectory"], sort_keys=True, separators=(",", ":")).encode("utf-8")
                hashes[row["config_id"]] = hashlib.sha256(encoded).hexdigest()
            groups.append({"scene_id": key[0], "goal_seed": key[1], "diffusion_seed": key[2], "configuration_count": len(hashes), "unique_trajectory_count": len(set(hashes.values())), "hashes": hashes})
        output[stratum] = {"all_methods_trajectory_equivalent": bool(groups) and all(group["unique_trajectory_count"] == 1 for group in groups), "groups": groups}
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_analysis(run_dir: Path, offline_summary: Path, protocol_path: Path, out_dir: Path, replicates: int, seed: int, strict: bool) -> dict[str, Any]:
    protocol, protocol_sha = runner.load_protocol(protocol_path)
    manifest, trials = load_primary_trials(run_dir, strict=strict)
    if manifest["protocol_sha256"] != protocol_sha:
        raise ValueError("Manifest protocol hash does not match the supplied frozen protocol.")
    rows = [episode_row(record) for record in trials]
    reconciliation = reconcile_rows(rows, manifest)
    if strict and not reconciliation["passed"]:
        raise ValueError(f"Denominator reconciliation failed: {reconciliation['errors']}")
    summaries = summarize(rows, replicates, seed)
    equivalence = trajectory_equivalence(rows)
    for item in summaries:
        item["trajectory_equivalent_across_methods"] = equivalence[item["stabilizer_mode"]]["all_methods_trajectory_equivalent"]
    paired_effects = {
        stratum: {
            metric: paired_delta_bootstrap(rows, metric, stratum, replicates, seed + 100 + metric_index)
            for metric_index, metric in enumerate(("success", "spl", "final_distance_m", "path_efficiency", "sampler_p95_ms", "full_loop_p95_ms"))
        }
        for stratum in STRATUM_ORDER
    }
    correlations = offline_closed_loop_correlation(summaries, offline_summary)
    sensitivity = robustness(rows)
    write_csv(out_dir / "controlled_episodes.csv", rows)
    write_json(out_dir / "summary_by_config_stratum.json", summaries)
    write_json(out_dir / "paired_effects_vs_ddpm10.json", paired_effects)
    write_json(out_dir / "trajectory_equivalence.json", equivalence)
    write_json(out_dir / "offline_closed_loop_correlation.json", correlations)
    write_csv(out_dir / "robustness_sensitivity.csv", sensitivity)
    provenance = {row["trial_key"]: row["source_trial_json"] for row in rows}
    write_json(out_dir / "provenance_map.json", provenance)
    audit = {"run_id": manifest["run_id"], "protocol_sha256": protocol_sha, "policy_action_scale_m": protocol["navigation_contract"].get("policy_action_scale_m", 1.0), "reconciliation": reconciliation, "historical_records_ingested": 0, "stabilizer_pooled_rows": 0, "stabilizer_on_all_methods_trajectory_equivalent": equivalence["system_stabilizer_on"]["all_methods_trajectory_equivalent"], "bootstrap_replicates": replicates, "bootstrap_seed": seed, "descriptive_only_three_seeds": True, "passed": reconciliation["passed"] and not manifest["missing_trial_keys"]}
    write_json(out_dir / "audit.json", audit)
    return {"rows": rows, "summary": summaries, "paired_effects": paired_effects, "equivalence": equivalence, "correlations": correlations, "sensitivity": sensitivity, "audit": audit, "protocol": protocol}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--offline-summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_analysis(args.run_dir.resolve(), args.offline_summary.resolve(), args.protocol.resolve(), args.out.resolve(), args.bootstrap_replicates, args.seed, args.strict)
    print(json.dumps(result["audit"], indent=2))
    return 0 if result["audit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
