#!/usr/bin/env python3

from pathlib import Path
import unittest


class LegacyPdSourceTest(unittest.TestCase):
    def test_controller_preserves_heading_quadrant(self) -> None:
        source = (Path(__file__).resolve().parents[2] / "deployment" / "src" / "pd_controller.py").read_text(encoding="utf-8")
        self.assertIn("np.arctan2(dy, dx)", source)
        self.assertNotIn("np.arctan(dy/dx)", source)


if __name__ == "__main__":
    unittest.main()
