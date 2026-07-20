import io
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from convert_recon_huron_pilot import convert_recon, forward_segment_indices


class ConvertReconHuronPilotTest(unittest.TestCase):
    def test_forward_segments_keep_endpoints_and_apply_minimum(self) -> None:
        positions = np.array([[0, 0], [1, 0], [2, 0], [1, 0], [2, 0], [3, 0]])
        yaws = np.zeros(6)
        self.assertEqual(
            forward_segment_indices(positions, yaws, min_frames=2),
            [[0, 1, 2], [3, 4, 5]],
        )
        self.assertEqual(forward_segment_indices(positions, yaws, min_frames=4), [])

    def test_recon_conversion_is_single_artifact_and_canonical_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "pilot.hdf5"
            encoded = io.BytesIO()
            Image.new("RGB", (16, 12), (10, 20, 30)).save(encoded, format="JPEG")
            payload = np.frombuffer(encoded.getvalue(), dtype=np.uint8)
            with h5py.File(artifact, "w") as handle:
                jackal = handle.create_group("jackal")
                jackal.create_dataset(
                    "position", data=np.column_stack((np.arange(15) * 0.25, np.zeros(15)))
                )
                jackal.create_dataset("yaw", data=np.zeros(15))
                jackal.create_dataset("linear_velocity", data=np.ones(15))
                images = handle.create_group("images")
                rgb = images.create_dataset(
                    "rgb_left", (15,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
                )
                for index in range(15):
                    rgb[index] = payload
            output = root / "processed"
            result = convert_recon(artifact, output, 14, 14)
            self.assertEqual(result["frame_count"], 15)
            self.assertEqual(result["canonical_window_count"], 2)
            trajectory = output / "pilot"
            self.assertEqual(len(list(trajectory.glob("*.jpg"))), 15)
            self.assertTrue((trajectory / "traj_data.pkl").is_file())
            self.assertTrue((trajectory / "source_metadata.json").is_file())

    def test_recon_rejects_decodable_but_too_short_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "short.hdf5"
            encoded = io.BytesIO()
            Image.new("RGB", (4, 4)).save(encoded, format="JPEG")
            payload = np.frombuffer(encoded.getvalue(), dtype=np.uint8)
            with h5py.File(artifact, "w") as handle:
                jackal = handle.create_group("jackal")
                jackal.create_dataset("position", data=np.zeros((2, 2)))
                jackal.create_dataset("yaw", data=np.zeros(2))
                images = handle.create_group("images")
                rgb = images.create_dataset(
                    "rgb_left", (2,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
                )
                rgb[0] = payload
                rgb[1] = payload
            with self.assertRaisesRegex(ValueError, "requires 14"):
                convert_recon(artifact, root / "processed", 14, 14)


if __name__ == "__main__":
    unittest.main()
