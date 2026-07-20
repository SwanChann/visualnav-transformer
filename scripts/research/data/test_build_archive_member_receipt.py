import json
import tempfile
import unittest
from pathlib import Path

from build_archive_member_receipt import build_receipt


class BuildArchiveMemberReceiptTest(unittest.TestCase):
    def test_binds_member_to_parent_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent.json"
            parent.write_text(
                json.dumps({
                    "dataset_id": "recon",
                    "conversion_status": "not_started",
                    "artifact": {"path": "raw.tar.gz", "bytes": 10, "sha256": "a" * 64},
                }),
                encoding="utf-8",
            )
            inventory = root / "members.txt"
            inventory.write_text(
                "-rw-r--r-- owner/group 4 2026-01-01 00:00 recon_release/pilot.hdf5\n",
                encoding="utf-8",
            )
            artifact = root / "pilot.hdf5"
            artifact.write_bytes(b"test")
            receipt = build_receipt(
                "recon", parent, inventory, "recon_release/pilot.hdf5", artifact, "tester"
            )
            self.assertEqual(receipt["artifact"]["bytes"], 4)
            self.assertEqual(receipt["conversion_status"], "not_started")
            self.assertEqual(receipt["execution_counts"]["semantic_conversions"], 0)

    def test_rejects_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent.json"
            parent.write_text(
                json.dumps({
                    "dataset_id": "recon",
                    "conversion_status": "not_started",
                    "artifact": {},
                }),
                encoding="utf-8",
            )
            inventory = root / "members.txt"
            inventory.write_text(
                "-rw-r--r-- owner/group 5 2026-01-01 00:00 member.hdf5\n",
                encoding="utf-8",
            )
            artifact = root / "member.hdf5"
            artifact.write_bytes(b"test")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                build_receipt("recon", parent, inventory, "member.hdf5", artifact, "tester")


if __name__ == "__main__":
    unittest.main()
