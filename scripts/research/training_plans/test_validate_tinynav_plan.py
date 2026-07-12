from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from validate_tinynav_plan import analytic_budget, validate


PLAN = Path(__file__).resolve().parent / "tinynavbrain_4090_plan.yaml"


class PlanValidationTests(unittest.TestCase):
    def setUp(self):
        self.payload = yaml.safe_load(PLAN.read_text(encoding="utf-8"))

    def test_frozen_plan_passes(self):
        self.assertEqual(validate(self.payload), [])

    def test_rejects_non_meter_actions(self):
        broken = copy.deepcopy(self.payload)
        broken["model"]["action_units"] = "normalized"
        self.assertIn("model.action_units must be meters", validate(broken))

    def test_rejects_training_in_static_phase(self):
        broken = copy.deepcopy(self.payload)
        broken["phases"][0]["optimizer_steps"] = 1
        self.assertIn("static phase must use zero optimizer steps", validate(broken))

    def test_budget_is_explicitly_unmeasured(self):
        self.assertEqual(analytic_budget(self.payload)["kind"], "analytic_partial-accounting_not_measured")


if __name__ == "__main__":
    unittest.main()
