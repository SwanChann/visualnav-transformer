from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from validate_b0_execution_config import validate_b0_config


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("b0_execution_config_v0.1.yaml")


class B0ExecutionConfigTests(unittest.TestCase):
    def payload(self):
        return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_config_and_parent_hashes_pass(self):
        self.assertEqual(validate_b0_config(self.payload(), ROOT), [])

    def test_rejects_execution_result_or_authorization(self):
        payload = self.payload()
        payload["execution_performed"] = True
        payload["execution_guard"]["authorization_granted_in_this_config"] = True
        errors = validate_b0_config(payload)
        self.assertIn("static B0 config must not contain execution results", errors)
        self.assertIn("static config cannot grant B0 execution", errors)

    def test_rejects_step_or_accumulation_drift(self):
        payload = self.payload()
        payload["runs"][0]["optimizer_steps"] = 199
        payload["training"]["gradient_accumulation_steps"] = 1
        errors = validate_b0_config(payload)
        self.assertIn("B0 runs must be H0 200 then H1 200", errors)
        self.assertIn("B0 batch/accumulation contract must remain 4x2=8", errors)

    def test_rejects_pretrained_or_precision_drift(self):
        payload = copy.deepcopy(self.payload())
        payload["model"]["encoder_weights"] = "imagenet"
        payload["training"]["precision"]["grad_scaler"] = False
        errors = validate_b0_config(payload)
        self.assertIn("B0 must use local random initialization", errors)
        self.assertIn("B0 FP16/GradScaler contract changed", errors)

    def test_rejects_incomplete_resume_state(self):
        payload = self.payload()
        payload["checkpoint"]["required_state"].remove("ema")
        self.assertIn("B0 checkpoint state is incomplete", validate_b0_config(payload))

    def test_rejects_prerequisite_drift(self):
        payload = self.payload()
        payload["prerequisites"]["train_step_readiness"]["sha256"] = "0" * 64
        self.assertIn(
            "prerequisite report hash mismatch: train_step_readiness",
            validate_b0_config(payload, ROOT),
        )


if __name__ == "__main__":
    unittest.main()
