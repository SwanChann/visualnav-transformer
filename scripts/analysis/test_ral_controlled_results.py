#!/usr/bin/env python3

from __future__ import annotations

import unittest
import inspect

import ral_controlled_results as analysis


class ControlledResultsAnalysisTest(unittest.TestCase):
    def test_clopper_pearson_bounds(self) -> None:
        self.assertEqual(analysis.clopper_pearson(0, 3)[0], 0.0)
        self.assertEqual(analysis.clopper_pearson(3, 3)[1], 1.0)
        lower, upper = analysis.clopper_pearson(1, 3)
        self.assertLess(lower, 1 / 3)
        self.assertGreater(upper, 1 / 3)

    def test_spearman_degenerate_ties_are_explicit(self) -> None:
        result = analysis.spearman_with_exact_permutation([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], False, True)
        self.assertTrue(result["degenerate"])
        self.assertIsNone(result["rho"])

    def test_spearman_exact_order(self) -> None:
        result = analysis.spearman_with_exact_permutation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], False, True)
        self.assertAlmostEqual(result["rho"], 1.0)
        self.assertEqual(result["permutation_n"], 120)

    def test_reconciliation_keeps_crash_and_excludes_infrastructure(self) -> None:
        rows = [
            {"status": "completed", "scientific_denominator": True},
            {"status": "failed", "scientific_denominator": True},
            {"status": "crashed", "scientific_denominator": True},
            {"status": "infrastructure_invalid", "scientific_denominator": False},
        ]
        manifest = {"completed": 1, "failed": 1, "crashed": 1, "infrastructure_invalid": 1, "planned_trial_count": 4, "missing_trial_keys": []}
        report = analysis.reconcile_rows(rows, manifest)
        self.assertTrue(report["passed"])
        self.assertEqual(report["scientific_n"], 3)

    def test_paired_delta_uses_matching_clusters(self) -> None:
        rows = []
        for seed, baseline, candidate in ((11, 0.0, 1.0), (23, 1.0, 1.0), (47, 0.0, 1.0)):
            common = {"scene_id": "easy", "goal_seed": 0, "diffusion_seed": seed, "stabilizer_mode": "policy_only_off", "scientific_denominator": True}
            rows.append({**common, "config_id": analysis.BASELINE_ID, "success": baseline})
            rows.append({**common, "config_id": "ddim2_cfg0_standard_k8", "success": candidate})
        result = analysis.paired_delta_bootstrap(rows, "success", "policy_only_off", 100, 7)
        self.assertAlmostEqual(result["ddim2_cfg0_standard_k8"]["mean_delta_vs_ddpm10"], 2 / 3)
        self.assertEqual(result["ddim2_cfg0_standard_k8"]["paired_cluster_n"], 3)

    def test_primary_robustness_condition_reuses_terminal_outcome(self) -> None:
        source = inspect.getsource(analysis.robustness)
        self.assertIn('reached = bool(record["outcome"]["success"])', source)


if __name__ == "__main__":
    unittest.main()
