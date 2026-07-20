import json
import tempfile
import unittest
from pathlib import Path

from register_raw_artifact import register


class RegisterRawArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.artifact = self.root / "pilot.bin"
        self.artifact.write_bytes(b"pilot-data")
        self.license = self.root / "LICENSE.txt"
        self.license.write_text("MIT", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _registry(self, blocked=False):
        path = self.root / ("blocked.json" if blocked else "ok.json")
        dataset = {
            "dataset_id": "demo",
            "display_name": "Demo",
            "official_page": "https://example.test/data",
            "official_download": "https://example.test/download",
            "license_spdx_or_name": "UNKNOWN" if blocked else "MIT",
            "license_status": "unknown" if blocked else "official_dataset_page_explicit",
            "redistribution": "blocked_until_license_verified" if blocked else "permitted",
        }
        path.write_text(json.dumps({"datasets": [dataset]}), encoding="utf-8")
        return path

    def test_writes_checksum_bound_receipt(self):
        output = self.root / "receipt.json"
        receipt = register(
            self._registry(), "demo", self.artifact, "https://example.test/pilot.bin", self.license,
            "MIT", "tester", output
        )
        self.assertEqual(receipt["artifact"]["bytes"], 10)
        self.assertEqual(len(receipt["artifact"]["sha256"]), 64)
        self.assertEqual(receipt["conversion_status"], "not_started")
        self.assertTrue(output.is_file())

    def test_blocks_unresolved_license(self):
        with self.assertRaisesRegex(ValueError, "blocked"):
            register(
                self._registry(blocked=True),
                "demo",
                self.artifact,
                "https://example.test/pilot.bin",
                self.license,
                "UNKNOWN",
                "tester",
                self.root / "receipt.json",
            )

    def test_rejects_nonofficial_or_insecure_url(self):
        for url in ("http://example.test/pilot.bin", "https://evil.test/pilot.bin"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "HTTPS|official"):
                register(
                    self._registry(), "demo", self.artifact, url, self.license,
                    "MIT", "tester", self.root / "receipt.json"
                )

    def test_rejects_license_mismatch_and_empty_snapshot(self):
        with self.assertRaisesRegex(ValueError, "license name"):
            register(
                self._registry(), "demo", self.artifact, "https://example.test/pilot.bin",
                self.license, "Apache-2.0", "tester", self.root / "receipt.json"
            )
        self.license.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            register(
                self._registry(), "demo", self.artifact, "https://example.test/pilot.bin",
                self.license, "MIT", "tester", self.root / "receipt.json"
            )

    def test_refuses_overwrite(self):
        output = self.root / "receipt.json"
        args = (
            self._registry(), "demo", self.artifact, "https://example.test/pilot.bin",
            self.license, "MIT", "tester", output
        )
        register(*args)
        with self.assertRaises(FileExistsError):
            register(*args)

    def test_registry_revision_creates_linked_receipt_without_overwrite(self):
        first_registry = self._registry()
        first_output = self.root / "receipt-v1.json"
        register(
            first_registry, "demo", self.artifact,
            "https://example.test/pilot.bin", self.license,
            "MIT", "tester", first_output,
        )
        payload = json.loads(first_registry.read_text(encoding="utf-8"))
        payload["datasets"][0]["local_presence_status"] = "raw_pilot_present"
        second_registry = self.root / "registry-v2.json"
        second_registry.write_text(json.dumps(payload), encoding="utf-8")
        second_output = self.root / "receipt-v2.json"
        receipt = register(
            second_registry, "demo", self.artifact,
            "https://example.test/pilot.bin", self.license,
            "MIT", "tester", second_output,
        )
        self.assertTrue(first_output.is_file())
        self.assertTrue(second_output.is_file())
        self.assertEqual(len(receipt["prior_receipts_same_artifact"]), 1)
        self.assertEqual(
            receipt["prior_receipts_same_artifact"][0]["path"],
            first_output.as_posix(),
        )


if __name__ == "__main__":
    unittest.main()
