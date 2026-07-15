"""Auditable multi-dataset sampling weights without constructing a DataLoader."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence


class SamplerContractError(ValueError):
    pass


def build_sample_weights(dataset_ids: Sequence[str], mode: str) -> tuple[list[float], dict]:
    clean_ids = [str(item).strip() for item in dataset_ids]
    if not clean_ids or any(not item for item in clean_ids):
        raise SamplerContractError("dataset_ids must be non-empty strings")
    if mode not in {"proportional", "balanced"}:
        raise SamplerContractError("mode must be proportional or balanced")
    counts = Counter(clean_ids)
    dataset_count = len(counts)
    if mode == "proportional":
        weights = [1.0 / len(clean_ids)] * len(clean_ids)
    else:
        weights = [1.0 / (dataset_count * counts[item]) for item in clean_ids]
    mass = defaultdict(float)
    for dataset_id, weight in zip(clean_ids, weights):
        mass[dataset_id] += weight
    report = {
        "mode": mode,
        "sample_count": len(clean_ids),
        "dataset_counts": dict(sorted(counts.items())),
        "dataset_probability_mass": dict(sorted(mass.items())),
        "weights_sum": sum(weights),
    }
    return weights, report
