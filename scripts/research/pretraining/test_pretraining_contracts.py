from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from batch_contract import (
    BatchContractError,
    adapt_legacy_vint_batch,
    build_past_action_history,
)
from checkpoint_contract import validate_checkpoint_metadata
from sampler_contract import build_sample_weights
from training_contract import deterministic_h0_loss, rectified_flow_state
from validate_pretraining_backlog import validate_backlog, validate_smoke_config


ROOT = Path(__file__).resolve().parent


class BatchContractTests(unittest.TestCase):
    def legacy_fixture(self):
        batch = 2
        legacy = (
            torch.randn(batch, 18, 12, 16),
            torch.randn(batch, 3, 12, 16),
            torch.ones(batch, 8, 2),
            torch.tensor([3, 9]),
            torch.randn(batch, 2),
            torch.tensor([0, 1]),
            torch.tensor([1.0, 1.0]),
        )
        metadata = {
            "dataset_id": ["small", "large"],
            "trajectory_id": ["a", "b"],
            "dt_s": [0.1, 0.2],
            "waypoint_spacing_m": [0.12, 0.25],
            "target_scale_m": [0.12, 0.25],
            "goal_mask": torch.tensor([False, True]),
            "action_history": torch.zeros(batch, 4, 3),
            "action_history_mask": torch.zeros(batch, 4, dtype=torch.bool),
        }
        return legacy, metadata

    def test_adapter_uses_explicit_scale_and_goal_mask(self):
        legacy, metadata = self.legacy_fixture()
        batch = adapt_legacy_vint_batch(legacy, metadata)
        self.assertEqual(tuple(batch["obs_images"].shape), (2, 6, 3, 12, 16))
        self.assertTrue(torch.equal(batch["target_waypoints"][0], torch.full((8, 2), 0.12)))
        self.assertFalse(bool(batch["target_mask"][1].any()))

    def test_adapter_refuses_missing_scale(self):
        legacy, metadata = self.legacy_fixture()
        del metadata["target_scale_m"]
        with self.assertRaises(BatchContractError):
            adapt_legacy_vint_batch(legacy, metadata)

    def test_action_history_is_future_independent(self):
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [99.0, 99.0]])
        yaw = torch.zeros(5)
        first, first_mask = build_past_action_history(
            positions, yaw, current_index=3, history_steps=4
        )
        positions[4] = torch.tensor([-500.0, 800.0])
        yaw[4] = 2.7
        second, second_mask = build_past_action_history(
            positions, yaw, current_index=3, history_steps=4
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first_mask, second_mask))
        self.assertFalse(first_mask[0])
        self.assertTrue(torch.equal(first[0], torch.zeros(3)))


class SamplerAndLossTests(unittest.TestCase):
    def test_balanced_sampler_equalizes_dataset_mass(self):
        _, report = build_sample_weights(["small", "large", "large", "large"], "balanced")
        self.assertAlmostEqual(report["weights_sum"], 1.0)
        self.assertAlmostEqual(report["dataset_probability_mass"]["small"], 0.5)
        self.assertAlmostEqual(report["dataset_probability_mass"]["large"], 0.5)

    def test_dataset_macro_loss_does_not_follow_sample_count(self):
        prediction = torch.tensor([[[0.0]], [[10.0]], [[10.0]], [[10.0]]])
        target = torch.zeros_like(prediction)
        mask = torch.ones(4, 1, dtype=torch.bool)
        loss, per_dataset = deterministic_h0_loss(
            prediction, target, mask, ["small", "large", "large", "large"]
        )
        self.assertAlmostEqual(float(per_dataset["small"]), 0.0)
        self.assertAlmostEqual(float(per_dataset["large"]), 9.5)
        self.assertAlmostEqual(float(loss), 4.75)

    def test_rectified_flow_contract(self):
        target = torch.ones(2, 8, 2)
        noise = torch.zeros_like(target)
        time = torch.tensor([[0.0], [1.0]])
        state, velocity = rectified_flow_state(target, time, noise)
        self.assertTrue(torch.equal(state[0], target[0]))
        self.assertTrue(torch.equal(state[1], noise[1]))
        self.assertTrue(torch.equal(velocity, -target))


class ProvenanceAndBacklogTests(unittest.TestCase):
    def checkpoint_fixture(self):
        return {
            "schema_version": "0.1.0",
            "run_id": "fixture",
            "phase": "smoke",
            "head": "h0_deterministic",
            "seed": 0,
            "optimizer_step": 17,
            "git_commit": "a" * 40,
            "manifest_sha256": "b" * 64,
            "split_sha256": "c" * 64,
            "model_contract_sha256": "d" * 64,
            "config_sha256": "e" * 64,
            "rng_state_present": True,
            "sampler_state_present": True,
        }

    def test_checkpoint_requires_resolved_hashes_and_resume_identity(self):
        payload = self.checkpoint_fixture()
        self.assertEqual(validate_checkpoint_metadata(payload), [])
        broken = copy.deepcopy(payload)
        broken["manifest_sha256"] = "REQUIRED_AT_EXECUTION"
        self.assertIn(
            "manifest_sha256 must be a resolved lowercase SHA-256",
            validate_checkpoint_metadata(broken),
        )
        changed = copy.deepcopy(payload)
        changed["split_sha256"] = "f" * 64
        self.assertIn(
            "resume_mismatch:split_sha256",
            validate_checkpoint_metadata(changed, expected_resume=payload),
        )

    def test_backlog_and_smoke_config_are_pre_execution_only(self):
        backlog = yaml.safe_load((ROOT / "pretraining_backlog_v0.1.yaml").read_text(encoding="utf-8"))
        smoke = yaml.safe_load((ROOT / "smoke_config_v0.1.yaml").read_text(encoding="utf-8"))
        self.assertEqual(validate_backlog(backlog), [])
        self.assertEqual(validate_smoke_config(smoke), [])
        backlog["training_performed"] = True
        self.assertIn(
            "training_performed must be false in the pre-execution backlog",
            validate_backlog(backlog),
        )

    def test_environment_contract_keeps_data_work_on_4090(self):
        backlog = yaml.safe_load((ROOT / "pretraining_backlog_v0.1.yaml").read_text(encoding="utf-8"))
        data_pilot = next(task for task in backlog["tasks"] if task["id"] == "DATA-PILOT")
        data_pilot["execution_environment"] = "ubuntu"
        self.assertIn(
            "DATA-PILOT:stage DATA-PILOT must run on rtx4090_server",
            validate_backlog(backlog),
        )

    def test_smoke_cannot_move_to_windows(self):
        smoke = yaml.safe_load((ROOT / "smoke_config_v0.1.yaml").read_text(encoding="utf-8"))
        smoke["execution_environment"] = "windows"
        self.assertIn(
            "smoke execution_environment must be rtx4090_server",
            validate_smoke_config(smoke),
        )


if __name__ == "__main__":
    unittest.main()
