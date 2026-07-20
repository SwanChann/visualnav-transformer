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
                            "official_page": "https://example.test/recon",
                            "official_download": "https://example.test/recon-download",
                            "expected_local_root": "nomad_dataset/recon",
                            "raw_format": "HDF5",
                            "metric_waypoint_spacing_m": 0.25,
                            "license_spdx_or_name": "MIT",
                            "license_status": "official_dataset_page_explicit",
                        },
                        {
                            "dataset_id": "huron_sacson",
                            "official_page": "https://example.test/huron",
                            "official_download": "https://example.test/huron-download",
                            "expected_local_root": "nomad_dataset/sacson",
                            "raw_format": "ROS bags",
                            "metric_waypoint_spacing_m": 0.255,
                            "license_spdx_or_name": "MIT",
                            "license_status": "official_dataset_page_explicit",
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
        registry_payload = json.loads(self.registry.read_text(encoding="utf-8"))
        registry_entry = next(
            item for item in registry_payload["datasets"]
            if item["dataset_id"] == "recon"
        )
        registry_entry_sha = hashlib.sha256(
            json.dumps(
                registry_entry,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        receipt = self.root / "receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema_version": "0.1.0",
                    "dataset_id": "recon",
                    "registry_sha256": hashlib.sha256(self.registry.read_bytes()).hexdigest(),
                    "registry_entry_sha256": registry_entry_sha,
                    "artifact": {
                        "path": str(artifact),
                        "bytes": 3,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    },
                    "source_url": "https://example.test/recon.h5",
                    "license": {
                        "name": "MIT",
                        "status": "official_dataset_page_explicit",
                        "snapshot_sha256": hashlib.sha256(
                            license_snapshot.read_bytes()
                        ).hexdigest(),
                    },
                    "operator": "tester",
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
        self.assertTrue(report["receipt"]["artifact_checksum_reverified"])
        self.assertTrue(report["passed"])
        self.assertFalse(report["output"]["created_or_modified"])

    def test_template_or_nonofficial_receipt_is_rejected(self):
        raw = self.root / "recon_raw"
        release = raw / "recon_release"
        release.mkdir(parents=True)
        (release / "session.h5").write_bytes(b"raw")
        license_snapshot = self.root / "license.md"
        license_snapshot.write_text("snapshot", encoding="utf-8")
        receipt = self.root / "receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "_template_only": True,
                    "schema_version": "0.1.0",
                    "dataset_id": "recon",
                    "source_url": "https://evil.test/recon.h5",
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
        self.assertFalse(report["receipt"]["schema_and_registry_binding_valid"])
        errors = "\n".join(report["receipt"]["errors"])
        self.assertIn("not_a_template", errors)
        self.assertIn("source_url_https_official_host", errors)
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
