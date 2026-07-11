#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from tinynavbrain_scaffold import TinyNavBrainScaffold, TinyNavConfig


class TinyNavBrainScaffoldTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = TinyNavConfig(encoder_dim=32, d_model=64, transformer_layers=2, transformer_heads=4, transformer_ffn_dim=128, dropout=0.0)
        self.model = TinyNavBrainScaffold(self.config).eval()
        batch = 3
        self.inputs = {
            "obs_features": torch.randn(batch, self.config.observation_frames, self.config.encoder_dim),
            "goal_features": torch.randn(batch, self.config.encoder_dim),
            "goal_mask": torch.tensor([False, True, False]),
            "action_history": torch.randn(batch, self.config.action_history_length, 3),
            "action_history_mask": torch.tensor([[True, True, True, True], [False, False, False, False], [True, True, False, False]]),
            "dt_s": torch.full((batch, 1), 1.0 / 3.0),
            "waypoint_spacing_m": torch.full((batch, 1), 0.12),
        }

    def test_encode_and_heads_shapes(self) -> None:
        context = self.model.encode(**self.inputs)
        self.assertEqual(tuple(context.shape), (3, self.config.d_model))
        self.assertEqual(tuple(self.model.predict_deterministic(context).shape), (3, 8, 2))
        self.assertEqual(tuple(self.model.predict_progress(context).shape), (3, 1))
        velocity = self.model.flow_velocity(torch.randn(3, 8, 2), torch.rand(3, 1), context)
        self.assertEqual(tuple(velocity.shape), (3, 8, 2))

    def test_sampling_shape_and_seed_reproducibility(self) -> None:
        context = self.model.encode(**self.inputs)
        first = self.model.sample_flow(context, nfe=2, candidate_count=4, seed=123)
        second = self.model.sample_flow(context, nfe=2, candidate_count=4, seed=123)
        self.assertEqual(tuple(first.shape), (3, 4, 8, 2))
        self.assertTrue(torch.equal(first, second))

    def test_invalid_nfe_fails(self) -> None:
        context = self.model.encode(**self.inputs)
        with self.assertRaises(ValueError):
            self.model.sample_flow(context, nfe=3)

    def test_bad_physical_scale_fails(self) -> None:
        inputs = dict(self.inputs)
        inputs["dt_s"] = torch.zeros_like(inputs["dt_s"])
        with self.assertRaises(ValueError):
            self.model.encode(**inputs)

    def test_default_parameter_budget(self) -> None:
        model = TinyNavBrainScaffold(TinyNavConfig())
        self.assertLess(model.trainable_parameter_count(), model.config.max_trainable_parameters)


if __name__ == "__main__":
    unittest.main()
