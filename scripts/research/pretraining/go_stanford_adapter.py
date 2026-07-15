"""Read-only Go Stanford adapter for the TinyNavBrain canonical batch.

This module consumes the already-processed ViNT trajectory layout.  It never
creates an LMDB cache, index, model, optimizer, or checkpoint.  Pickle files
must therefore come from the trusted local processed dataset named by the
operator; do not use this loader on downloaded or otherwise untrusted pickle
payloads.
"""

from __future__ import annotations

import csv
import math
import pickle
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from batch_contract import build_past_action_history, validate_canonical_batch


GO_STANFORD_DATASET_ID = "go_stanford"
GO_STANFORD_DT_S = 1.0 / 3.0
GO_STANFORD_WAYPOINT_SPACING_M = 0.12
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


class GoStanfordDataError(ValueError):
    """Raised when a trajectory cannot satisfy the frozen data contract."""


@dataclass(frozen=True)
class GoStanfordTrajectory:
    trajectory_id: str
    trajectory_dir: Path
    positions_xy_m: np.ndarray
    yaw_rad: np.ndarray
    image_paths: tuple[Path, ...]

    @property
    def frame_count(self) -> int:
        return int(self.positions_xy_m.shape[0])


def _positions_xy(value: Any) -> np.ndarray:
    try:
        positions = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GoStanfordDataError(f"position cannot be converted to float64: {exc}") from exc
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise GoStanfordDataError(f"position must have shape [T,>=2], got {positions.shape}")
    positions = np.ascontiguousarray(positions[:, :2])
    if positions.shape[0] < 1 or not bool(np.isfinite(positions).all()):
        raise GoStanfordDataError("position must be non-empty and finite")
    return positions


def _yaw_rad(value: Any) -> np.ndarray:
    raw = np.asarray(value, dtype=object).reshape(-1)
    result: list[float] = []
    for index, item in enumerate(raw):
        try:
            scalar = np.asarray(item, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise GoStanfordDataError(f"yaw[{index}] is not numeric: {exc}") from exc
        if scalar.size != 1:
            raise GoStanfordDataError(f"yaw[{index}] must be scalar, got {scalar.shape}")
        result.append(float(scalar[0]))
    yaw = np.asarray(result, dtype=np.float64)
    if yaw.size < 1 or not bool(np.isfinite(yaw).all()):
        raise GoStanfordDataError("yaw must be non-empty and finite")
    return yaw


def _numbered_images(trajectory_dir: Path) -> tuple[Path, ...]:
    indexed: list[tuple[int, Path]] = []
    for path in trajectory_dir.glob("*.jpg"):
        try:
            index = int(path.stem)
        except ValueError as exc:
            raise GoStanfordDataError(f"non-numeric image name: {path.name}") from exc
        indexed.append((index, path))
    indexed.sort(key=lambda item: item[0])
    indices = [item[0] for item in indexed]
    expected = list(range(len(indexed)))
    if indices != expected:
        raise GoStanfordDataError(
            f"image indices must be contiguous from zero; got {indices[:5]}...{indices[-5:]}"
        )
    return tuple(path for _, path in indexed)


def load_go_stanford_trajectory(trajectory_dir: Path) -> GoStanfordTrajectory:
    """Load one trusted, processed trajectory without writing caches."""

    trajectory_dir = Path(trajectory_dir)
    metadata_path = trajectory_dir / "traj_data.pkl"
    if not trajectory_dir.is_dir():
        raise GoStanfordDataError(f"trajectory directory does not exist: {trajectory_dir}")
    if not metadata_path.is_file():
        raise GoStanfordDataError(f"missing traj_data.pkl: {trajectory_dir}")
    try:
        with metadata_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise GoStanfordDataError(f"cannot load trusted metadata {metadata_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GoStanfordDataError(f"traj_data.pkl must contain a dict, got {type(payload).__name__}")
    missing = sorted({"position", "yaw"} - set(payload))
    if missing:
        raise GoStanfordDataError(f"traj_data.pkl missing keys: {missing}")

    positions = _positions_xy(payload["position"])
    yaw = _yaw_rad(payload["yaw"])
    images = _numbered_images(trajectory_dir)
    if yaw.shape[0] != positions.shape[0]:
        raise GoStanfordDataError(
            f"position/yaw length mismatch: {positions.shape[0]} vs {yaw.shape[0]}"
        )
    if len(images) != positions.shape[0]:
        raise GoStanfordDataError(
            f"image/pose length mismatch: {len(images)} vs {positions.shape[0]}"
        )
    return GoStanfordTrajectory(
        trajectory_id=trajectory_dir.name,
        trajectory_dir=trajectory_dir,
        positions_xy_m=positions,
        yaw_rad=yaw,
        image_paths=images,
    )


def decode_rgb_image(path: Path, image_size: tuple[int, int]) -> Tensor:
    """Decode, 4:3 center-crop, resize, and ImageNet-normalize an RGB tensor."""

    if len(image_size) != 2 or min(image_size) < 1:
        raise GoStanfordDataError(f"image_size must contain two positive values, got {image_size}")
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            target_ratio = 4.0 / 3.0
            if width / height > target_ratio:
                crop_width = int(round(height * target_ratio))
                left = (width - crop_width) // 2
                image = image.crop((left, 0, left + crop_width, height))
            else:
                crop_height = int(round(width / target_ratio))
                top = (height - crop_height) // 2
                image = image.crop((0, top, width, top + crop_height))
            # PIL takes (width, height), while the contract uses (height, width).
            image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:
        raise GoStanfordDataError(f"cannot decode image {path}: {exc}") from exc
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    if tuple(tensor.shape) != (3, image_size[0], image_size[1]):
        raise GoStanfordDataError(f"decoded image has unexpected shape: {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise GoStanfordDataError(f"decoded image is non-finite: {path}")
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _local_waypoints(
    positions_xy_m: np.ndarray,
    yaw_rad: np.ndarray,
    current_index: int,
    action_horizon: int,
) -> Tensor:
    future = positions_xy_m[current_index + 1 : current_index + action_horizon + 1]
    delta = future - positions_xy_m[current_index]
    yaw = float(yaw_rad[current_index])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    local = delta.dot(rotation)
    return torch.as_tensor(local, dtype=torch.float32)


def build_go_stanford_canonical_sample(
    trajectory: GoStanfordTrajectory,
    *,
    current_index: int,
    observation_frames: int = 6,
    action_history_steps: int = 4,
    action_horizon: int = 8,
    frame_stride: int = 1,
    image_size: tuple[int, int] = (96, 96),
    dt_s: float = GO_STANFORD_DT_S,
    waypoint_spacing_m: float = GO_STANFORD_WAYPOINT_SPACING_M,
) -> dict[str, Any]:
    """Build one deterministic canonical sample in meters without model execution."""

    if observation_frames < 1 or action_history_steps < 1 or action_horizon < 1:
        raise GoStanfordDataError("frame and horizon settings must be positive")
    first_observation = current_index - (observation_frames - 1) * frame_stride
    final_target = current_index + action_horizon * frame_stride
    if first_observation < 0 or final_target >= trajectory.frame_count:
        raise GoStanfordDataError(
            f"sample indices [{first_observation},{final_target}] exceed trajectory length "
            f"{trajectory.frame_count}"
        )
    observation_indices = list(range(first_observation, current_index + 1, frame_stride))
    target_indices = list(range(current_index + frame_stride, final_target + 1, frame_stride))
    if len(observation_indices) != observation_frames or len(target_indices) != action_horizon:
        raise GoStanfordDataError("frame_stride does not produce the frozen frame/horizon counts")

    observations = torch.stack(
        [decode_rgb_image(trajectory.image_paths[index], image_size) for index in observation_indices]
    )
    goal = decode_rgb_image(trajectory.image_paths[final_target], image_size)
    positions = torch.as_tensor(trajectory.positions_xy_m, dtype=torch.float32)
    yaw = torch.as_tensor(trajectory.yaw_rad, dtype=torch.float32)
    history, history_mask = build_past_action_history(
        positions,
        yaw,
        current_index=current_index,
        history_steps=action_history_steps,
        frame_stride=frame_stride,
    )
    # _local_waypoints assumes adjacent targets.  The current frozen pilot uses
    # frame_stride=1; reject other values rather than silently changing labels.
    if frame_stride != 1:
        raise GoStanfordDataError("canonical pilot currently requires frame_stride=1")
    targets = _local_waypoints(
        trajectory.positions_xy_m,
        trajectory.yaw_rad,
        current_index,
        action_horizon,
    )
    batch = {
        "obs_images": observations.unsqueeze(0),
        "goal_image": goal.unsqueeze(0),
        "goal_mask": torch.zeros(1, dtype=torch.bool),
        "action_history": history.unsqueeze(0),
        "action_history_mask": history_mask.unsqueeze(0),
        "dt_s": torch.tensor([[dt_s]], dtype=torch.float32),
        "waypoint_spacing_m": torch.tensor([[waypoint_spacing_m]], dtype=torch.float32),
        "target_waypoints": targets.unsqueeze(0),
        "target_mask": torch.ones(1, action_horizon, dtype=torch.bool),
        "dataset_id": [GO_STANFORD_DATASET_ID],
        "trajectory_id": [trajectory.trajectory_id],
    }
    validate_canonical_batch(
        batch,
        observation_frames=observation_frames,
        action_history_steps=action_history_steps,
        action_horizon=action_horizon,
    )
    return batch


@dataclass(frozen=True)
class GoStanfordSampleIndex:
    trajectory_id: str
    current_index: int


class GoStanfordCanonicalDataset(Dataset):
    """Map-style dataset over leakage-safe manifest rows.

    Every item is unbatched.  Use :func:`collate_go_stanford_canonical` so the
    resulting dict is revalidated against the frozen canonical batch contract.
    The deterministic ``data_fraction`` selection is intended for the frozen
    B0 preflight/smoke and does not mutate split files.
    """

    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest_path: Path,
        split: str,
        observation_frames: int = 6,
        action_history_steps: int = 4,
        action_horizon: int = 8,
        image_size: tuple[int, int] = (96, 96),
        data_fraction: float = 1.0,
        subset_seed: int = 0,
        trajectory_cache_size: int = 8,
    ) -> None:
        if not 0 < data_fraction <= 1:
            raise GoStanfordDataError("data_fraction must be in (0,1]")
        if trajectory_cache_size < 1:
            raise GoStanfordDataError("trajectory_cache_size must be positive")
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        self.split = split.strip().lower()
        self.observation_frames = observation_frames
        self.action_history_steps = action_history_steps
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.trajectory_cache_size = trajectory_cache_size
        self._trajectory_cache: OrderedDict[str, GoStanfordTrajectory] = OrderedDict()

        with self.manifest_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        matching = sorted(
            (
                row
                for row in rows
                if row.get("dataset_id") == GO_STANFORD_DATASET_ID
                and row.get("split", "").strip().lower() == self.split
            ),
            key=lambda row: row["trajectory_id"],
        )
        if not matching:
            raise GoStanfordDataError(
                f"manifest has no {GO_STANFORD_DATASET_ID} rows for split {self.split!r}"
            )
        identities = [row["trajectory_id"] for row in matching]
        if len(identities) != len(set(identities)):
            raise GoStanfordDataError(f"manifest contains duplicate trajectories in {self.split}")

        sample_index: list[GoStanfordSampleIndex] = []
        for row in matching:
            try:
                frame_count = int(row["num_frames"])
            except (KeyError, ValueError) as exc:
                raise GoStanfordDataError(
                    f"invalid num_frames for {row.get('trajectory_id', '<unknown>')}"
                ) from exc
            for current_index in range(observation_frames - 1, frame_count - action_horizon):
                sample_index.append(GoStanfordSampleIndex(row["trajectory_id"], current_index))
        if not sample_index:
            raise GoStanfordDataError(f"split {self.split!r} has no canonical samples")
        self.full_sample_count = len(sample_index)
        if data_fraction < 1:
            subset_size = max(1, int(round(len(sample_index) * data_fraction)))
            chosen = sorted(random.Random(subset_seed).sample(range(len(sample_index)), subset_size))
            sample_index = [sample_index[index] for index in chosen]
        self.sample_index = tuple(sample_index)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_trajectory_cache"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return len(self.sample_index)

    def _trajectory(self, trajectory_id: str) -> GoStanfordTrajectory:
        cached = self._trajectory_cache.pop(trajectory_id, None)
        if cached is None:
            cached = load_go_stanford_trajectory(self.dataset_root / trajectory_id)
        self._trajectory_cache[trajectory_id] = cached
        while len(self._trajectory_cache) > self.trajectory_cache_size:
            self._trajectory_cache.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.sample_index[index]
        batch = build_go_stanford_canonical_sample(
            self._trajectory(sample.trajectory_id),
            current_index=sample.current_index,
            observation_frames=self.observation_frames,
            action_history_steps=self.action_history_steps,
            action_horizon=self.action_horizon,
            image_size=self.image_size,
        )
        item: dict[str, Any] = {}
        for name, value in batch.items():
            if isinstance(value, Tensor):
                item[name] = value[0]
            elif name in {"dataset_id", "trajectory_id"}:
                item[name] = value[0]
            else:
                raise GoStanfordDataError(f"unsupported canonical item field {name}")
        item["current_index"] = sample.current_index
        return item


def collate_go_stanford_canonical(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise GoStanfordDataError("cannot collate an empty item list")
    metadata_fields = {"dataset_id", "trajectory_id"}
    batch: dict[str, Any] = {}
    for name in items[0]:
        if name == "current_index":
            continue
        if name in metadata_fields:
            batch[name] = [str(item[name]) for item in items]
        else:
            batch[name] = torch.stack([item[name] for item in items])
    validate_canonical_batch(batch)
    batch["current_index"] = torch.tensor(
        [int(item["current_index"]) for item in items], dtype=torch.int64
    )
    return batch


def make_go_stanford_dataloader(
    dataset: GoStanfordCanonicalDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if batch_size < 1 or num_workers < 0:
        raise GoStanfordDataError("batch_size must be positive and num_workers non-negative")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_go_stanford_canonical,
        drop_last=False,
    )
