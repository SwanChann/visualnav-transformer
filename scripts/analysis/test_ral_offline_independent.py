#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from ral_offline_independent import compute_independent_metrics, load_metric_waypoint_spacing, wrap_angle


class IndependentOfflineMetricsTest(unittest.TestCase):
    def test_dataset_scale_comes_from_training_config(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        spacing = load_metric_waypoint_spacing(
            repo_root / "nomad_dataset" / "go_stanford",
            repo_root / "train" / "vint_train" / "data" / "data_config.yaml",
        )
        self.assertEqual(spacing, 0.12)

    def test_metrics_are_physical_values(self) -> None:
        ground_truth = np.array([[1.0, 0.0], [2.0, 0.0]])
        chosen = np.array([[0.5, 0.0], [1.0, 0.0]])
        candidates = np.stack([chosen, ground_truth])
        metrics = compute_independent_metrics(chosen, candidates, ground_truth)
        self.assertAlmostEqual(metrics["action_ade_m"], 0.75)
        self.assertAlmostEqual(metrics["action_fde_m"], 1.0)
        self.assertAlmostEqual(metrics["progress_error_m"], 1.0)
        self.assertAlmostEqual(metrics["minade_k_m"], 0.0)

    def test_minade_honors_all_candidates(self) -> None:
        ground_truth = np.array([[1.0, 0.0], [2.0, 0.0]])
        bad = ground_truth + 5.0
        exact = ground_truth.copy()
        metrics = compute_independent_metrics(bad, np.stack([bad, exact]), ground_truth)
        self.assertEqual(metrics["minade_k_m"], 0.0)
        self.assertGreater(metrics["action_ade_m"], 0.0)

    def test_candidate_diversity_is_pairwise_waypoint_distance(self) -> None:
        gt = np.zeros((2, 2))
        a = np.zeros((2, 2))
        b = np.ones((2, 2))
        metrics = compute_independent_metrics(a, np.stack([a, b]), gt)
        self.assertAlmostEqual(metrics["candidate_diversity_m"], np.sqrt(2.0))

    def test_heading_error_wraps_at_pi(self) -> None:
        self.assertAlmostEqual(abs(wrap_angle(2 * np.pi - 0.1)), 0.1)


if __name__ == "__main__":
    unittest.main()
