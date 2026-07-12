"""Training-free trajectory aggregation for diffusion-policy candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateDiagnostics:
    candidate_count: int
    pairwise_mean_m: float
    pairwise_p95_m: float
    medoid_index: int
    medoid_mean_distance_m: float


def _validated(candidates: np.ndarray) -> np.ndarray:
    values = np.asarray(candidates, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2 or values.shape[0] < 1:
        raise ValueError(f"expected candidates [N,T,2], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("candidate trajectories contain non-finite coordinates")
    return values


def _pairwise(candidates: np.ndarray) -> np.ndarray:
    delta = candidates[:, None, :, :] - candidates[None, :, :, :]
    return np.linalg.norm(delta, axis=-1).mean(axis=-1)


def trajectory_diagnostics(candidates: np.ndarray) -> CandidateDiagnostics:
    values = _validated(candidates)
    distances = _pairwise(values)
    medoid = int(np.argmin(distances.mean(axis=1)))
    upper = distances[np.triu_indices(len(values), k=1)]
    return CandidateDiagnostics(
        candidate_count=len(values),
        pairwise_mean_m=float(upper.mean()) if len(upper) else 0.0,
        pairwise_p95_m=float(np.percentile(upper, 95)) if len(upper) else 0.0,
        medoid_index=medoid,
        medoid_mean_distance_m=float(distances[medoid].mean()),
    )


def select_trajectory(candidates: np.ndarray, method: str, trim_fraction: float = 0.25) -> np.ndarray:
    """Select or aggregate candidates without a learned verifier.

    ``mean`` matches the current NoMaD deployment behavior. ``medoid`` returns an
    actually sampled trajectory. ``trimmed_mean`` removes trajectories farthest
    from the medoid before averaging; it is an exploration candidate, not a safety
    guarantee.
    """

    values = _validated(candidates)
    if method == "mean":
        return values.mean(axis=0)
    distances = _pairwise(values)
    medoid = int(np.argmin(distances.mean(axis=1)))
    if method == "medoid":
        return values[medoid].copy()
    if method == "trimmed_mean":
        if not 0.0 <= trim_fraction < 1.0:
            raise ValueError("trim_fraction must be in [0, 1)")
        keep_n = max(1, int(np.ceil(len(values) * (1.0 - trim_fraction))))
        keep = np.argsort(distances[medoid])[:keep_n]
        return values[keep].mean(axis=0)
    raise ValueError(f"unknown selection method: {method}")
