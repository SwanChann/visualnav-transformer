import io
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from audit_recon_huron_raw_pilot import audit_recon


class ReconRawPilotAuditTest(unittest.TestCase):
    def make_hdf5(self, path: Path, finite: bool = True) -> None:
        image = Image.new("RGB", (8, 6), color=(20, 40, 60))
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG")
        payload = np.frombuffer(encoded.getvalue(), dtype=np.uint8)
        with h5py.File(path, "w") as handle:
            jackal = handle.create_group("jackal")
            position = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
            if not finite:
                position[1, 0] = np.nan
            jackal.create_dataset("position", data=position)
            jackal.create_dataset("yaw", data=np.array([0.0, 0.1, 0.2]))
            images = handle.create_group("images")
            rgb = images.create_dataset(
                "rgb_left", (3,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
            )
            for index in range(3):
                rgb[index] = payload

    def test_valid_recon_hdf5_passes_without_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pilot.hdf5"
            self.make_hdf5(path)
            report = audit_recon(path)
            self.assertTrue(report["passed"])
            self.assertEqual(report["counts"]["rgb_left"], 3)
            self.assertEqual(len(report["decoded_images"]), 3)

    def test_nonfinite_pose_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pilot.hdf5"
            self.make_hdf5(path, finite=False)
            report = audit_recon(path)
            self.assertFalse(report["passed"])
            self.assertIn(
                "sampled position/yaw contains non-finite values",
                report["errors"],
            )


if __name__ == "__main__":
    unittest.main()
