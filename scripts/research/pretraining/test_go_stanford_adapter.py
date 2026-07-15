from __future__ import annotations

import pickle
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from go_stanford_adapter import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    GoStanfordCanonicalDataset,
    GoStanfordDataError,
    build_go_stanford_canonical_sample,
    collate_go_stanford_canonical,
    load_go_stanford_trajectory,
)


class GoStanfordAdapterTests(unittest.TestCase):
    def make_trajectory(self, root: Path, frames: int = 14) -> Path:
        trajectory = root / "demo_0"
        trajectory.mkdir(parents=True)
        positions = np.asarray([[index * 0.12, 0.0] for index in range(frames)], dtype=object)
        yaw = np.asarray([np.asarray([0.0]) for _ in range(frames)], dtype=object)
        with (trajectory / "traj_data.pkl").open("wb") as handle:
            pickle.dump({"position": positions, "yaw": yaw}, handle)
        for index in range(frames):
            Image.new("RGB", (16, 12), color=(index, 10, 20)).save(trajectory / f"{index}.jpg")
        return trajectory

    def test_real_layout_normalizes_object_yaw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory = load_go_stanford_trajectory(self.make_trajectory(Path(tmp)))
            self.assertEqual(trajectory.positions_xy_m.shape, (14, 2))
            self.assertEqual(trajectory.yaw_rad.shape, (14,))
            self.assertEqual(trajectory.frame_count, 14)

    def test_canonical_sample_is_metric_and_past_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory = load_go_stanford_trajectory(self.make_trajectory(Path(tmp)))
            batch = build_go_stanford_canonical_sample(trajectory, current_index=5)
            self.assertEqual(tuple(batch["obs_images"].shape), (1, 6, 3, 96, 96))
            self.assertEqual(tuple(batch["target_waypoints"].shape), (1, 8, 2))
            expected_x = torch.arange(1, 9, dtype=torch.float32) * 0.12
            self.assertTrue(torch.allclose(batch["target_waypoints"][0, :, 0], expected_x))
            self.assertTrue(torch.equal(batch["target_waypoints"][0, :, 1], torch.zeros(8)))
            self.assertTrue(bool(batch["action_history_mask"].all()))
            self.assertAlmostEqual(float(batch["waypoint_spacing_m"][0, 0]), 0.12, places=6)
            expected_pixel = (torch.tensor([0.0, 10.0, 20.0]) / 255.0 - IMAGENET_MEAN[:, 0, 0]) / IMAGENET_STD[:, 0, 0]
            self.assertTrue(torch.allclose(batch["obs_images"][0, 0, :, 0, 0], expected_pixel, atol=1e-5))

    def test_missing_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_trajectory(Path(tmp))
            (path / "7.jpg").unlink()
            with self.assertRaisesRegex(GoStanfordDataError, "image indices"):
                load_go_stanford_trajectory(path)

    def test_short_trajectory_cannot_form_frozen_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory = load_go_stanford_trajectory(self.make_trajectory(Path(tmp), frames=13))
            with self.assertRaisesRegex(GoStanfordDataError, "exceed trajectory length"):
                build_go_stanford_canonical_sample(trajectory, current_index=5)

    def test_map_dataset_and_collate_use_deterministic_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = self.make_trajectory(root, frames=17)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["dataset_id", "trajectory_id", "split", "num_frames"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset_id": "go_stanford",
                        "trajectory_id": trajectory.name,
                        "split": "train",
                        "num_frames": 17,
                    }
                )
            first = GoStanfordCanonicalDataset(
                dataset_root=root,
                manifest_path=manifest,
                split="train",
                data_fraction=0.5,
                subset_seed=7,
            )
            second = GoStanfordCanonicalDataset(
                dataset_root=root,
                manifest_path=manifest,
                split="train",
                data_fraction=0.5,
                subset_seed=7,
            )
            self.assertEqual(first.full_sample_count, 4)
            self.assertEqual(first.sample_index, second.sample_index)
            self.assertEqual(len(first), 2)
            batch = collate_go_stanford_canonical([first[0], first[1]])
            self.assertEqual(tuple(batch["obs_images"].shape), (2, 6, 3, 96, 96))
            self.assertEqual(tuple(batch["current_index"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
