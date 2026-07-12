#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import ral_mujoco_controlled as runner
import validate_ral_mujoco_trial as validator


PROTOCOL_PATH = REPO_ROOT / "后续研究内容" / "benchmark" / "ral_mujoco_protocol_v0.1.json"


class ControlledMuJoCoContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol, cls.protocol_sha = runner.load_protocol(PROTOCOL_PATH)
        cls.plans = runner.expand_trial_plans(cls.protocol)

    def valid_record(self) -> dict:
        plan = self.plans[0]
        path = REPO_ROOT / "results" / "research" / "ral_mujoco_controlled" / "unit" / "trials" / "trial.json"
        runtime = path.parent / "runtime"
        record = runner._base_record(plan, self.protocol, PROTOCOL_PATH, self.protocol_sha, "unit", 1, path, runtime)
        record["status"] = "failed"
        record["outcome"].update({"final_distance_m": 4.0, "path_length_m": 1.0, "shortest_path_m": 5.0, "exception": None, "exit_code": 1})
        record["trajectory"] = [{"tick": 0, "x_m": 0.0, "y_m": 0.0}]
        record["timing"]["full_loop_per_call_ms"] = [10.0]
        return record

    def test_protocol_expands_to_unique_frozen_matrix(self) -> None:
        self.assertEqual(len(self.plans), 90)
        self.assertEqual(len({plan["trial_key"] for plan in self.plans}), 90)

    def test_protocol_overlay_is_hash_bound_and_applies_only_allowed_fields(self) -> None:
        overlay = REPO_ROOT / "后续研究内容" / "benchmark" / "ral_mujoco_calibration_v0.2.json"
        protocol, overlay_hash = runner.load_protocol(overlay)
        self.assertEqual(protocol["navigation_contract"]["timeout"]["max_steps"], 62)
        self.assertEqual(protocol["navigation_contract"]["timeout"]["simulated_navigation_seconds"], 15.5)
        self.assertEqual(len(runner.expand_trial_plans(protocol)), 90)
        self.assertEqual(len(overlay_hash), 64)

    def test_duplicate_trial_key_is_rejected(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["configurations"].append(copy.deepcopy(protocol["configurations"][0]))
        protocol["planned_matrix"]["planned_trial_count_before_gate"] += 18
        with self.assertRaisesRegex(ValueError, "duplicate"):
            runner.expand_trial_plans(protocol)

    def test_valid_trial_passes(self) -> None:
        report = validator.validate(self.valid_record())
        self.assertTrue(report["passed"], report["errors"])

    def test_every_required_provenance_field_is_enforced(self) -> None:
        for location, fields in validator.TRIAL_REQUIRED.items():
            for field in fields:
                record = self.valid_record()
                target = record if location == "$" else record[location]
                del target[field]
                with self.subTest(location=location, field=field):
                    self.assertFalse(validator.validate(record)["passed"])

    def test_atomic_write_leaves_no_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            runner.atomic_write_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": True})
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

    def test_crash_is_retained_as_trial_record(self) -> None:
        plan = self.plans[0]
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            run_dir = Path(temporary)

            def crash(*_args, **_kwargs):
                raise RuntimeError("synthetic crash")

            path = runner.run_trial_attempt(plan, self.protocol, PROTOCOL_PATH, self.protocol_sha, "unit", run_dir, executor=crash)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "crashed")
            self.assertIn("synthetic crash", payload["outcome"]["exception"])
            self.assertTrue(validator.validate(payload)["passed"])

    def test_manifest_reconciles_and_prefers_scientific_retry(self) -> None:
        plan = self.plans[0]
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            run_dir = Path(temporary)
            trials = run_dir / "trials"
            trials.mkdir()
            crashed = self.valid_record(); crashed["trial_key"] = plan["trial_key"]; crashed["status"] = "crashed"; crashed["outcome"]["exception"] = "boom"
            completed = self.valid_record(); completed["trial_key"] = plan["trial_key"]; completed["status"] = "failed"
            runner.atomic_write_json(trials / f"{plan['trial_key']}__a01.json", crashed)
            runner.atomic_write_json(trials / f"{plan['trial_key']}__a02.json", completed)
            manifest = runner.build_manifest("unit", self.protocol_sha, [plan], run_dir)
            self.assertEqual(manifest["failed"], 1)
            self.assertEqual(manifest["crashed"], 0)
            self.assertEqual(len(manifest["trial_records"]), 1)
            self.assertTrue(validator.validate(manifest)["passed"])


if __name__ == "__main__":
    unittest.main()
