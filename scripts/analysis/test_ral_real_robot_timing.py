#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import ral_real_robot_timing as timing


class RealRobotTimingTest(unittest.TestCase):
    def test_package_and_untrusted_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trials"
            trial = root / "ddim2cfg0tts8"
            logs = trial / "logs"
            logs.mkdir(parents=True)
            fields = list(timing.TIMING_FIELDS)
            with (logs / "timing_profile.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for tick, total in ((5, 100.0), (10, 200.0)):
                    writer.writerow({
                        "timestamp": "1970-01-01T00:00:00.000", "label": "navigate", "tick": tick,
                        "camera_ms": 1, "ui_ms": 1, "preprocess_ms": 2, "infer_ms": total - 5,
                        "command_ms": 1, "total_ms": total, "infer_fps": 1000 / (total - 5),
                        "loop_fps": 1000 / total, "camera_status": "ok"
                    })
            (trial / "interactive_summary.txt").write_text(
                "Backend: real, Map: demo\nTask 1: status=success\n", encoding="utf-8"
            )
            out = Path(tmp) / "out"
            audit = timing.run(root, out)
            self.assertEqual(audit["timing_files"], 1)
            self.assertEqual(audit["profiled_loop_samples"], 2)
            with (out / "real_trial_annotation_template.csv").open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["auto_status_untrusted"], "success")
            self.assertEqual(row["manual_outcome"], "")
            self.assertEqual(row["evidence_usable"], "")

    def test_r7_percentile(self) -> None:
        self.assertAlmostEqual(timing.percentile([0, 10], 0.95), 9.5)


if __name__ == "__main__":
    unittest.main()
