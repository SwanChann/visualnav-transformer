from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

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
        self.goal_dir = self.run_dir / "goal_views"
        self.images_root = self.run_dir / "images"
        self.videos_root = self.run_dir / "videos"
        self.logs_dir = self.run_dir / "logs"
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.goal_dir.mkdir(parents=True, exist_ok=True)
        self.images_root.mkdir(parents=True, exist_ok=True)
        self.videos_root.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.capture_queue: list[CaptureFrame] = []
        self.selected_capture_index = -1
        self.estop_requested = False
        self.exit_requested = False
        self._saved_goal_labels: set[str] = set()

    def clear_estop(self) -> None:
        self.estop_requested = False

    def request_estop(self) -> None:
        self.estop_requested = True

    def clear_exit(self) -> None:
        self.exit_requested = False

    def request_exit(self) -> None:
        self.exit_requested = True

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

    def save_goal_view(self, image, label: str) -> Path:
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_") or "goal"
        saved_path = self.goal_dir / f"{safe_label}.png"
        if safe_label not in self._saved_goal_labels:
            image.save(saved_path)
            self._saved_goal_labels.add(safe_label)
        return saved_path

    def start_image_recording(self, label: str = "navigation") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_") or "navigation"
        output_dir = self.images_root / f"{timestamp}_{safe_label}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def start_video_recording(self, label: str = "navigation") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_") or "navigation"
        output_dir = self.videos_root / f"{timestamp}_{safe_label}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def save_camera_image(self, camera_image, output_dir: Path, tick: int, label: str = "camera") -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_") or "camera"
        saved_path = output_dir / f"{tick:06d}_{safe_label}.png"
        camera_image.convert("RGB").save(saved_path)
        return saved_path

    def save_goal_pair(self, camera_image, goal_image, output_dir: Path, tick: int, label: str = "nav") -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        fpv = camera_image.convert("RGB")
        goal = goal_image.convert("RGB") if goal_image is not None else Image.new("RGB", fpv.size, (32, 32, 32))
        if goal.size != fpv.size:
            resampling = getattr(Image, "Resampling", Image)
            goal = goal.resize(fpv.size, resampling.BILINEAR)
        combined = Image.new("RGB", (fpv.width + goal.width, max(fpv.height, goal.height)), (0, 0, 0))
        combined.paste(fpv, (0, 0))
        combined.paste(goal, (fpv.width, 0))
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_") or "nav"
        saved_path = output_dir / f"{tick:06d}_{safe_label}.png"
        combined.save(saved_path)
        return saved_path
