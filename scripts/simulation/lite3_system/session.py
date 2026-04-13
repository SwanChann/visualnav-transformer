from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from project_paths import repo_path


@dataclass
class CaptureFrame:
    index: int
    label: str
    image: object
    position: np.ndarray | None
    yaw: float | None
    map_name: str | None
    saved_path: Path


class NavigationSession:
    def __init__(self, run_label: str = "navigation_session") -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = repo_path("results", "deployment", f"{timestamp}_{run_label}")
        self.capture_dir = self.run_dir / "captures"
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.capture_queue: list[CaptureFrame] = []
        self.selected_capture_index = -1
        self.estop_requested = False

    def clear_estop(self) -> None:
        self.estop_requested = False

    def request_estop(self) -> None:
        self.estop_requested = True

    def add_capture(self, image, position, yaw: float | None, map_name: str | None, label_prefix: str = "capture") -> CaptureFrame:
        capture_index = len(self.capture_queue)
        label = f"{label_prefix}_{capture_index:02d}"
        saved_path = self.capture_dir / f"{label}.png"
        image.save(saved_path)
        stored_position = None if position is None else np.asarray(position, dtype=float).copy()
        capture = CaptureFrame(
            index=capture_index,
            label=label,
            image=image.copy(),
            position=stored_position,
            yaw=None if yaw is None else float(yaw),
            map_name=map_name,
            saved_path=saved_path,
        )
        self.capture_queue.append(capture)
        self.selected_capture_index = capture_index
        return capture

    def get_selected_capture(self):
        if not self.capture_queue:
            return None
        if self.selected_capture_index < 0 or self.selected_capture_index >= len(self.capture_queue):
            self.selected_capture_index = len(self.capture_queue) - 1
        return self.capture_queue[self.selected_capture_index]

    def select_previous_capture(self):
        if not self.capture_queue:
            return None
        self.selected_capture_index = (self.selected_capture_index - 1) % len(self.capture_queue)
        return self.get_selected_capture()

    def select_next_capture(self):
        if not self.capture_queue:
            return None
        self.selected_capture_index = (self.selected_capture_index + 1) % len(self.capture_queue)
        return self.get_selected_capture()

    def get_capture_by_index(self, capture_index: int):
        if not self.capture_queue:
            return None
        if capture_index < 0:
            return self.get_selected_capture()
        if capture_index >= len(self.capture_queue):
            raise IndexError(
                f"Capture index {capture_index} out of range; current queue size is {len(self.capture_queue)}."
            )
        self.selected_capture_index = capture_index
        return self.capture_queue[capture_index]

    def get_recent_captures(self, count: int) -> list[CaptureFrame]:
        if count <= 0:
            return []
        return list(self.capture_queue[-count:])
