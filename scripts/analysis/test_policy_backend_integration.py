#!/usr/bin/env python3

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1]
for candidate in (SCRIPTS, SCRIPTS / "simulation"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research.policy_backend.runtime import NavigationPolicyBackend
from lite3_system.interfaces import scale_policy_actions
from lite3_system.system import Lite3System


class FakeBackend:
    context_size = 3
    def preprocess_frame(self, frame): return frame
    def build_topomap_tensor(self, topomap): return topomap
    def predict_exploration(self, frame_buffer): return frame_buffer
    def predict_navigation(self, frame_buffer, topomap, closest_node, goal_node, topomap_tensor=None): return topomap
    def close(self): return None


class PolicyBackendIntegrationTest(unittest.TestCase):
    def test_backend_contract_is_structural(self) -> None:
        self.assertIsInstance(FakeBackend(), NavigationPolicyBackend)

    def test_system_exposes_constructor_injection(self) -> None:
        parameters = inspect.signature(Lite3System.__init__).parameters
        self.assertIn("high_level_backend", parameters)
        self.assertIsNone(parameters["high_level_backend"].default)

    def test_policy_native_actions_are_scaled_to_target_meters(self) -> None:
        native = np.asarray([[[1.0, -2.0], [0.5, 0.25]]])
        np.testing.assert_allclose(scale_policy_actions(native, 0.1), native * 0.1)
        with self.assertRaises(ValueError):
            scale_policy_actions(native, 0.0)


if __name__ == "__main__":
    unittest.main()
