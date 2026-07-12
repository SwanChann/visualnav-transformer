"""Framework-neutral boundary between navigation policy and deployment runtime.

The contract deliberately contains no Torch, MuJoCo, ROS, or Lite3 types. A backend
may use those libraries internally, while deployment consumes a validated decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class PolicyMetadata:
    backend_id: str
    action_frame: str
    action_horizon: int
    context_frames: int
    metric_scale_m: float
    checkpoint_sha256: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyRequest:
    observation_frames: Sequence[Any]
    goal_frames: Sequence[Any] = ()
    goal_masked: bool = False
    deadline_ms: float | None = None
    seed: int | None = None


@dataclass(frozen=True)
class PolicyDecision:
    trajectory_xy: np.ndarray
    candidates_xy: np.ndarray
    selected_waypoint_index: int
    predicted_distance: float | None
    latency_ms: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, metadata: PolicyMetadata) -> None:
        trajectory = np.asarray(self.trajectory_xy, dtype=float)
        candidates = np.asarray(self.candidates_xy, dtype=float)
        expected = (metadata.action_horizon, 2)
        if trajectory.shape != expected:
            raise ValueError(f"trajectory shape {trajectory.shape} != {expected}")
        if candidates.ndim != 3 or candidates.shape[1:] != expected:
            raise ValueError(f"candidate shape {candidates.shape} is incompatible with {expected}")
        if not np.isfinite(trajectory).all() or not np.isfinite(candidates).all():
            raise ValueError("policy decision contains non-finite coordinates")
        if not 0 <= self.selected_waypoint_index < metadata.action_horizon:
            raise ValueError("selected waypoint index is outside the action horizon")
        if not np.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")


@runtime_checkable
class PolicyBackend(Protocol):
    """Minimal injectable high-level policy backend."""

    @property
    def metadata(self) -> PolicyMetadata:
        ...

    def infer(self, request: PolicyRequest) -> PolicyDecision:
        ...

    def close(self) -> None:
        ...
