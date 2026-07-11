#!/usr/bin/env python3
"""Synthetic tests for ral_mujoco_summary.py (no simulator required)."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import ral_mujoco_summary as summary


def record(cfg: float, run_index: int, final_distance: float) -> dict:
    return {
        "encoder": "baseline",
        "map_name": "synthetic",
        "scheduler": "ddim",
        "ddim_steps": 2,
        "cfg_weight": cfg,
        "run_index": run_index,
        "success": True,
        "steps": 10,
        "path_distance": 1.25,
        "final_goal_dist": final_distance,
        "wall_time": 0.5,
        "exit_code": 0,
    }


class MujocoSummaryTest(unittest.TestCase):
    def test_audit_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench = root / "bench" / "source_a"
            bench.mkdir(parents=True)
            payload = [record(0.0, 0, 0.4), record(1.0, 0, 1.2)]
            (bench / "raw_results.json").write_text(json.dumps(payload), encoding="utf-8")

            out = root / "out"
            audit = summary.run(root / "bench", out, threshold=0.6)

            self.assertEqual(audit["record_count"], 2)
            self.assertEqual(audit["success_final_distance_inconsistency_count"], 1)
            self.assertEqual(audit["repeated_trajectory_signature_count"], 0)
            self.assertEqual(audit["missing_optional_field_counts"]["stabilizer_mode"], 2)
            self.assertFalse((out / "fig_mujoco_paths.pdf").exists())

            with (out / "mujoco_closed_loop_unified.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("success_final_distance_inconsistent", rows[1]["record_flags"])

    def test_missing_required_field_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench = root / "bench" / "bad"
            bench.mkdir(parents=True)
            bad = record(0.0, 0, 0.4)
            del bad["success"]
            (bench / "raw_results.json").write_text(json.dumps([bad]), encoding="utf-8")
            with self.assertRaises(summary.DataError):
                summary.load_records(root / "bench", 0.6)


if __name__ == "__main__":
    unittest.main()
