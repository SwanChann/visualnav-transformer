#!/usr/bin/env python3
"""Descriptive replay of candidate-diversity adaptive-compute routing.

This does not tune or validate a deployable router. It asks whether an online
observable from the cheap backend would have improved an existing paired offline
record set. Every predeclared quantile is reported to prevent best-threshold cherry
picking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def replay(records: list[dict], cheap_id: str, fallback_id: str) -> dict:
    paired: dict[int, dict[str, dict]] = {}
    for record in records:
        if record.get("status") == "valid":
            paired.setdefault(int(record["case_index"]), {})[str(record["config_id"])] = record
    cases = [bucket for _, bucket in sorted(paired.items()) if cheap_id in bucket and fallback_id in bucket]
    if not cases:
        raise ValueError("no paired valid cases for requested configurations")
    diversity = np.asarray([c[cheap_id]["candidate_diversity_m"] for c in cases], dtype=float)
    rows = []
    for quantile in (0.0, 0.25, 0.5, 0.75, 1.0):
        threshold = float(np.quantile(diversity, quantile))
        chosen = [c[fallback_id] if c[cheap_id]["candidate_diversity_m"] > threshold else c[cheap_id] for c in cases]
        rows.append({
            "quantile": quantile,
            "threshold_m": threshold,
            "fallback_cases": sum(r["config_id"] == fallback_id for r in chosen),
            "case_n": len(chosen),
            "action_ade_m_mean": float(np.mean([r["action_ade_m"] for r in chosen])),
            "sampler_latency_ms_mean": float(np.mean([r["sampler_latency_ms"] for r in chosen])),
        })
    return {
        "record_kind": "descriptive_posthoc_falsification_only",
        "cheap_id": cheap_id,
        "fallback_id": fallback_id,
        "routing_signal": "cheap candidate_diversity_m",
        "threshold_policy": "all predeclared empirical quantiles reported; none selected",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--cheap-id", default="ddim2_cfg0_standard_k8")
    parser.add_argument("--fallback-id", default="ddpm10_cfg0_standard_k8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = replay(json.loads(args.records.read_text(encoding="utf-8")), args.cheap_id, args.fallback_id)
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
