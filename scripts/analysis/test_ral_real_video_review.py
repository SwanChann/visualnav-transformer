#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import ral_real_video_review as review


class RealVideoReviewTest(unittest.TestCase):
    def test_contact_sheet_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "trial"
            video_dir = trial / "videos" / "run"
            video_dir.mkdir(parents=True)
            video = video_dir / "navigation_record.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
            self.assertTrue(writer.isOpened())
            for value in range(20):
                writer.write(np.full((48, 64, 3), value * 10, dtype=np.uint8))
            writer.release()
            annotations = root / "annotations.csv"
            with annotations.open("w", encoding="utf-8", newline="") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=["trial_id", "trial_path", "map"])
                csv_writer.writeheader()
                csv_writer.writerow({"trial_id": "trial", "trial_path": str(trial), "map": "demo"})
            rows = review.run(root, annotations, root / "out", samples=4)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["video_role"], "robot_camera")
            self.assertTrue(Path(rows[0]["contact_sheet"]).is_file())
            self.assertGreater(rows[0]["duration_s"], 0)


if __name__ == "__main__":
    unittest.main()
