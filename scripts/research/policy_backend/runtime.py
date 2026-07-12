"""Structural contract for the high-level navigation policy used by a runtime."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class NavigationPolicyBackend(Protocol):
    """The deployment state machine depends on this boundary, not on NoMaD internals."""

    context_size: int

    def preprocess_frame(self, frame: Any) -> Any:
        ...

    def build_topomap_tensor(self, topomap: list[Any]) -> Any:
        ...

    def predict_exploration(self, frame_buffer: Any) -> Any:
        ...

    def predict_navigation(
        self,
        frame_buffer: Any,
        topomap: list[Any],
        closest_node: int,
        goal_node: int,
        topomap_tensor: Any | None = None,
    ) -> Any:
        ...

    def close(self) -> None:
        ...
