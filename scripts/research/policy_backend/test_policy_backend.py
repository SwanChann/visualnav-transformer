from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contracts import PolicyDecision, PolicyMetadata
from selection import select_trajectory, trajectory_diagnostics
from adaptive_compute_replay import replay
from runtime import NavigationPolicyBackend


class SelectionTests(unittest.TestCase):
    def test_runtime_contract_is_structural(self):
        class FakeBackend:
            context_size = 3
            def preprocess_frame(self, frame): return frame
            def build_topomap_tensor(self, topomap): return topomap
            def predict_exploration(self, frame_buffer): return frame_buffer
            def predict_navigation(self, frame_buffer, topomap, closest_node, goal_node, topomap_tensor=None): return topomap
            def close(self): return None
        self.assertIsInstance(FakeBackend(), NavigationPolicyBackend)

    def test_medoid_rejects_single_outlier(self):
        candidates = np.asarray([[[0, 0], [1, 0]], [[0, 0], [1.1, 0]], [[0, 0], [8, 0]]], dtype=float)
        selected = select_trajectory(candidates, "medoid")
        np.testing.assert_allclose(selected, candidates[1])
        self.assertEqual(trajectory_diagnostics(candidates).medoid_index, 1)

    def test_trimmed_mean_is_deterministic(self):
        candidates = np.asarray([[[0, 0]], [[1, 0]], [[100, 0]], [[2, 0]]], dtype=float)
        np.testing.assert_allclose(select_trajectory(candidates, "trimmed_mean", 0.25), [[[1, 0]]][0])

    def test_decision_validation(self):
        metadata = PolicyMetadata("fake", "base_link", 2, 4, 0.25)
        decision = PolicyDecision(np.zeros((2, 2)), np.zeros((3, 2, 2)), 1, None, 1.0)
        decision.validate(metadata)
        with self.assertRaises(ValueError):
            PolicyDecision(np.zeros((1, 2)), np.zeros((3, 2, 2)), 0, None, 1.0).validate(metadata)

    def test_replay_reports_all_thresholds(self):
        records = []
        for index, diversity in enumerate((0.1, 0.2)):
            records.extend([
                {"case_index": index, "config_id": "cheap", "status": "valid", "candidate_diversity_m": diversity, "action_ade_m": 1.0, "sampler_latency_ms": 2.0},
                {"case_index": index, "config_id": "fallback", "status": "valid", "candidate_diversity_m": diversity, "action_ade_m": 0.5, "sampler_latency_ms": 8.0},
            ])
        result = replay(records, "cheap", "fallback")
        self.assertEqual(len(result["rows"]), 5)
        self.assertEqual(result["record_kind"], "descriptive_posthoc_falsification_only")


if __name__ == "__main__":
    unittest.main()
