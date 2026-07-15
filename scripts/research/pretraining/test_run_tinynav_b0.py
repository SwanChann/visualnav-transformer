from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from run_tinynav_b0 import (
    B0RunnerError,
    dry_run_report,
    load_frozen_config,
    validate_execute_guard,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("b0_execution_config_v0.1.yaml")
RUNNER = Path(__file__).with_name("run_tinynav_b0.py")


class B0RunnerStaticTests(unittest.TestCase):
    def test_module_has_no_top_level_torch_or_model_import(self):
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        forbidden = {"torch", "torchvision", "tinynavbrain_image_policy", "go_stanford_adapter"}
        top_level = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.append(node.module.split(".")[0])
        self.assertTrue(forbidden.isdisjoint(top_level))

    def test_dry_run_records_zero_execution(self):
        payload = load_frozen_config(CONFIG, ROOT)
        report = dry_run_report(payload, CONFIG, ROOT)
        boundary = report["execution_boundary"]
        self.assertFalse(boundary["model_instantiated"])
        self.assertFalse(boundary["optimizer_created"])
        self.assertEqual(boundary["forward_calls"], 0)
        self.assertEqual(boundary["backward_calls"], 0)
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertFalse(report["execution_authorized"])

    def test_execute_guard_rejects_missing_token_before_execution(self):
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(B0RunnerError, "authorization token"):
            validate_execute_guard(payload, ROOT, None)


if __name__ == "__main__":
    unittest.main()
