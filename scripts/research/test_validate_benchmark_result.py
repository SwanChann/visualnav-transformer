#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import validate_benchmark_result as validator


EXAMPLE = Path(__file__).parents[2] / "后续研究内容" / "benchmark" / "examples" / "iid_result_example.json"


class BenchmarkResultValidatorTest(unittest.TestCase):
    def payload(self) -> dict:
        return json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_example_passes(self) -> None:
        self.assertTrue(validator.validate(self.payload())["passed"])

    def test_lodo_leakage_fails(self) -> None:
        payload = self.payload()
        payload["track"] = "lodo"
        payload["data"]["heldout_dataset"] = "go_stanford"
        payload["data"]["eval_dataset"] = "go_stanford"
        report = validator.validate(payload)
        self.assertFalse(report["passed"])
        self.assertTrue(any("appears in training" in error for error in report["errors"]))

    def test_completed_requires_raw_results(self) -> None:
        payload = self.payload()
        payload["artifacts"]["raw_results"] = None
        self.assertFalse(validator.validate(payload)["passed"])

    def test_example_count_must_reconcile(self) -> None:
        payload = self.payload()
        payload["training"]["examples_seen_total"] += 1
        self.assertFalse(validator.validate(payload)["passed"])

    def test_heldout_dataset_cannot_hide_in_exposure_keys(self) -> None:
        payload = self.payload()
        payload["training"]["examples_seen_by_dataset"] = {"go_stanford": 1600, "heldout": 1600}
        report = validator.validate(payload)
        self.assertFalse(report["passed"])
        self.assertTrue(any("keys must exactly equal" in error for error in report["errors"]))

    def test_head_and_nfe_contract(self) -> None:
        payload = self.payload()
        payload["model"]["head"] = "deterministic"
        payload["inference"]["nfe"] = 8
        self.assertFalse(validator.validate(payload)["passed"])

    def test_bad_corruption_vocabulary_fails(self) -> None:
        payload = self.payload()
        payload["track"] = "corruption"
        payload["data"]["corruption"] = {
            "type": "make_it_bad", "severity": "high", "seed": "x", "clean_reference_run_id": ""
        }
        self.assertFalse(validator.validate(payload)["passed"])


if __name__ == "__main__":
    unittest.main()
