import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from preflight_recon_huron_raw import build_report


class ReconHuronRawPreflightTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        (self.repo / "train/vint_train/process_data").mkdir(parents=True)
        (self.repo / "train/process_recon.py").write_text("# recon\n", encoding="utf-8")
        (self.repo / "train/process_bags.py").write_text("# bags\n", encoding="utf-8")
        (self.repo / "train/vint_train/process_data/process_bags_config.yaml").write_text(
            "sacson: {}\n", encoding="utf-8"
        )
        self.registry = self.repo / "registry.json"
        self.registry.write_text(
            json.dumps(
                {
                    "datasets": [
                        {
                            "dataset_id": "recon",
                            "expected_local_root": "nomad_dataset/recon",
                            "raw_format": "HDF5",
                            "metric_waypoint_spacing_m": 0.25,
                            "license_spdx_or_name": "MIT",
                        },
                        {
                            "dataset_id": "huron_sacson",
                            "expected_local_root": "nomad_dataset/sacson",
                            "raw_format": "ROS bags",
                            "metric_waypoint_spacing_m": 0.255,
                            "license_spdx_or_name": "MIT",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _build(self, dataset_id, input_root, checksum_mode="metadata", **kwargs):
        return build_report(
            repo_root=self.repo,
            registry_path=self.registry,
            dataset_id=dataset_id,
            input_root=input_root,
            checksum_mode=checksum_mode,
            **kwargs,
        )

    def test_missing_root_is_blocked_without_creating_output(self):
        missing = self.root / "missing"
        report = self._build("recon", missing)
        self.assertFalse(report["passed"])
        self.assertEqual(report["input"]["candidate_count"], 0)
        self.assertEqual(report["execution_counts"]["conversions"], 0)
        self.assertFalse((self.repo / "nomad_dataset/recon").exists())

    def test_recon_inventory_requires_recon_release_and_can_hash(self):
        raw = self.root / "recon_raw"
        release = raw / "recon_release"
        release.mkdir(parents=True)
        artifact = release / "session_01.hdf5"
        artifact.write_bytes(b"synthetic-hdf5-placeholder")
        report = self._build("recon", raw, checksum_mode="sha256")
        candidate = report["input"]["candidates"][0]
        self.assertEqual(candidate["planned_source_id"], "session_01")
        self.assertEqual(candidate["sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest())
        self.assertIn(
            "raw_artifact_receipt_schema_binding_valid",
            report["blocking_gates"],
        )
        self.assertFalse(report["processor"]["executed"])

    def test_huron_inventory_renders_sacson_command(self):
        raw = self.root / "huron_raw"
        bag = raw / "building_a" / "day_01.bag"
        bag.parent.mkdir(parents=True)
        bag.write_bytes(b"synthetic-rosbag-placeholder")
        report = self._build("huron_sacson", raw)
        self.assertEqual(
            report["input"]["candidates"][0]["planned_source_id"],
            "building_a_day_01",
        )
        self.assertIn("--dataset-name", report["processor"]["planned_command"])
        self.assertIn("sacson", report["processor"]["planned_command"])
        self.assertEqual(report["execution_counts"]["model_forward"], 0)

    def test_valid_receipt_and_license_snapshot_close_governance_fields(self):
        raw = self.root / "recon_raw"
        release = raw / "recon_release"
        release.mkdir(parents=True)
        artifact = release / "session.h5"
        artifact.write_bytes(b"raw")
        license_snapshot = self.root / "license.md"
        license_snapshot.write_text("official snapshot", encoding="utf-8")
        receipt = self.root / "receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "dataset_id": "recon",
                    "registry_sha256": hashlib.sha256(self.registry.read_bytes()).hexdigest(),
                    "artifact": {"bytes": 3, "sha256": "a" * 64},
                    "license": {
                        "name": "MIT",
                        "snapshot_sha256": hashlib.sha256(
                            license_snapshot.read_bytes()
                        ).hexdigest(),
                    },
                    "conversion_status": "not_started",
                }
            ),
            encoding="utf-8",
        )
        report = self._build(
            "recon",
            raw,
            checksum_mode="sha256",
            receipt_path=receipt,
            license_snapshot_path=license_snapshot,
        )
        self.assertTrue(report["receipt"]["schema_and_registry_binding_valid"])
        self.assertTrue(report["receipt"]["license_snapshot_hash_matches"])
        self.assertTrue(report["passed"])
        self.assertFalse(report["output"]["created_or_modified"])


if __name__ == "__main__":
    unittest.main()
