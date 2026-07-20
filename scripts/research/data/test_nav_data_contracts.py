#!/usr/bin/env python3
"""Synthetic contract tests; no training data or model dependencies."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import audit_nav_dataset as audit
import build_nav_manifest as build
import build_grouped_splits as grouped


class NavDataContractsTest(unittest.TestCase):
    def registry(self) -> dict:
        return {
            "demo": {
                "dataset_id": "demo",
                "license_spdx_or_name": "MIT",
                "license_status": "official",
                "metric_waypoint_spacing_m": 0.2,
                "nominal_dt_s": 0.1,
                "platform": "robot",
                "environment": ["synthetic"],
            }
        }

    def row(self, trajectory: str, split: str, session: str) -> dict[str, str]:
        return {
            "manifest_version": build.MANIFEST_VERSION,
            "dataset_id": "demo",
            "trajectory_id": trajectory,
            "source_session": session,
            "source_session_method": "fixture",
            "source_path": ".",
            "split": split,
            "leakage_group": f"demo:{session}",
            "robot_id": "robot",
            "camera_id": "default",
            "environment": "synthetic",
            "num_frames": "2",
            "has_traj_data": "True",
            "nominal_dt_s": "0.1",
            "metric_waypoint_spacing_m": "0.2",
            "position_unit": "meter",
            "yaw_unit": "radian",
            "license_id": "MIT",
            "license_status": "official",
            "processor_version": "test-v1",
            "trajectory_meta_sha256": "abc",
        }

    def test_clean_manifest_passes(self) -> None:
        report = audit.audit_rows(
            [self.row("t1", "train", "s1"), self.row("t2", "test", "s2")],
            self.registry(),
            check_paths=False,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["error_count"], 0)

    def test_session_leakage_fails(self) -> None:
        report = audit.audit_rows(
            [self.row("t1", "train", "same"), self.row("t2", "test", "same")],
            self.registry(),
            check_paths=False,
        )
        codes = {item["code"] for item in report["errors"]}
        self.assertIn("leakage_group_cross_split", codes)
        self.assertIn("source_session_cross_split", codes)

    def test_whitespace_is_not_a_leakage_escape(self) -> None:
        first = self.row("t1", "train", "same")
        second = self.row("t2", "test", " same ")
        second["leakage_group"] = " demo:same "
        report = audit.audit_rows([first, second], self.registry(), check_paths=False)
        codes = {item["code"] for item in report["errors"]}
        self.assertIn("leakage_group_cross_split", codes)
        self.assertIn("source_session_cross_split", codes)

    def test_validation_alias_is_canonical(self) -> None:
        rows = [self.row("t1", "val", "same"), self.row("t2", "validation", "same")]
        report = audit.audit_rows(rows, self.registry(), check_paths=False)
        self.assertTrue(report["passed"])

    def test_builder_does_not_unpickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            split = root / "splits" / "train"
            trajectory = dataset / "run_0"
            trajectory.mkdir(parents=True)
            split.mkdir(parents=True)
            (trajectory / "0.jpg").write_bytes(b"not-an-image-but-countable")
            (trajectory / "traj_data.pkl").write_bytes(b"not-a-valid-pickle")
            (split / "traj_names.txt").write_text("run_0\n", encoding="utf-8")
            rows = build.build_rows("demo", dataset, root / "splits", self.registry()["demo"], "test-v1")
            self.assertEqual(rows[0]["num_frames"], 1)
            self.assertTrue(rows[0]["has_traj_data"])
            self.assertEqual(len(rows[0]["trajectory_meta_sha256"]), 64)

    def test_builder_can_create_initial_unassigned_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = root / "dataset" / "session_0"
            trajectory.mkdir(parents=True)
            (trajectory / "0.jpg").write_bytes(b"frame-placeholder")
            (trajectory / "traj_data.pkl").write_bytes(b"metadata-placeholder")
            rows = build.build_rows(
                "recon",
                root / "dataset",
                None,
                self.registry()["demo"],
                "test-v1",
                nominal_dt_s=0.3,
                metric_waypoint_spacing_m=0.25,
            )
            self.assertEqual(rows[0]["split"], "unassigned")
            self.assertEqual(rows[0]["source_session"], "session_0")
            self.assertEqual(rows[0]["nominal_dt_s"], 0.3)
            self.assertEqual(rows[0]["metric_waypoint_spacing_m"], 0.25)
            report = audit.audit_rows(rows, {"recon": self.registry()["demo"]}, False)
            codes = {item["code"] for item in report["errors"]}
            self.assertIn("invalid_or_unassigned_split", codes)

    def test_recon_and_huron_session_identity_matches_legacy_processors(self) -> None:
        self.assertEqual(
            build.infer_session("recon", "recon_session_01"),
            ("recon_session_01", "recon_hdf5_stem_identity"),
        )
        self.assertEqual(
            build.infer_session("huron_sacson", "building_a_day_01_0"),
            ("building_a_day_01", "sacson_legacy_bag_remove_segment_index"),
        )
        self.assertEqual(
            build.infer_session("huron_sacson", "building_a_day_01_7"),
            ("building_a_day_01", "sacson_legacy_bag_remove_segment_index"),
        )

    def test_huron_fallback_session_is_rejected(self) -> None:
        row = self.row("bag_0", "train", "bag_0")
        row["dataset_id"] = "huron_sacson"
        row["leakage_group"] = "huron_sacson:bag_0"
        row["source_session_method"] = "trajectory_id_fallback"
        registry = {
            "huron_sacson": {
                "dataset_id": "huron_sacson",
                "license_spdx_or_name": "MIT",
            }
        }
        report = audit.audit_rows([row], registry, check_paths=False)
        codes = {item["code"] for item in report["errors"]}
        self.assertIn("unsafe_session_method", codes)

    def test_grouped_split_keeps_session_together(self) -> None:
        rows = [
            self.row("t1", "train", "same"),
            self.row("t2", "test", "same"),
            self.row("t3", "train", "other"),
        ]
        updated, report = grouped.assign_groups(
            rows, test_fraction=0.34, val_fraction=0.0, seed=7, max_fraction_deviation=None
        )
        same_splits = {row["split"] for row in updated if row["source_session"] == "same"}
        self.assertEqual(len(same_splits), 1)
        self.assertEqual(report["group_count"], 2)
        audited = audit.audit_rows(updated, self.registry(), check_paths=False)
        self.assertTrue(audited["passed"])


if __name__ == "__main__":
    unittest.main()
