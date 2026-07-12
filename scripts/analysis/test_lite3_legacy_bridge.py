#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest


SIMULATION_ROOT = Path(__file__).resolve().parents[1] / "simulation"
if str(SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATION_ROOT))

from lite3_system.legacy_bridge import load_legacy


class LegacyBridgePathTest(unittest.TestCase):
    def test_loads_legacy_script_from_simulation_directory(self) -> None:
        module = load_legacy()
        self.assertEqual(Path(module.__file__).resolve(), SIMULATION_ROOT / "nomad_mujoco_lite3_nav.py")
        self.assertIn("easy", module.SCENE_MAPS)


if __name__ == "__main__":
    unittest.main()
