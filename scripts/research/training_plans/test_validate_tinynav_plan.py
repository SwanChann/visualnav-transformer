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

    def test_rejects_identity_token_and_wrong_frame_count(self):
        broken = copy.deepcopy(self.payload)
        broken["model"]["physical_tokens"].append("embodiment_id")
        broken["model"]["observation_frames"] = 4
        errors = validate(broken)
        self.assertIn("model.observation_frames must equal the frozen contract value 6", errors)
        self.assertIn("model.physical_tokens must exclude dataset or embodiment identity", errors)

    def test_rejects_obsolete_residual_flow_name(self):
        broken = copy.deepcopy(self.payload)
        del broken["model"]["rectified_flow_head"]
        broken["model"]["residual_flow_head"] = True
        self.assertIn("model must use the frozen rectified_flow_head name", validate(broken))

    def test_rejects_wrong_execution_environment(self):
        broken = copy.deepcopy(self.payload)
        broken["execution"]["environment"] = "windows"
        broken["execution"]["data_residency"] = "local_copy"
        errors = validate(broken)
        self.assertIn("execution.environment must be rtx4090_server", errors)
        self.assertIn("execution.data_residency must be rtx4090_server_only", errors)

    def test_budget_is_explicitly_unmeasured(self):
        self.assertEqual(analytic_budget(self.payload)["kind"], "analytic_partial-accounting_not_measured")


if __name__ == "__main__":
    unittest.main()
