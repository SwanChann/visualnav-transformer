#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from tinynavbrain_image_policy import (
    TinyNavBrainImagePolicy,
    TinyNavImagePolicyError,
)
from tinynavbrain_scaffold import TinyNavConfig


class TinyNavBrainImagePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.config = TinyNavConfig(
            d_model=64,
            transformer_layers=1,
            transformer_heads=4,
            transformer_ffn_dim=128,
            dropout=0.0,
        )
        cls.model = TinyNavBrainImagePolicy(cls.config).eval()
        batch_size = 1
        cls.batch = {
            "obs_images": torch.randn(
                batch_size, cls.config.observation_frames, 3, 96, 96
            ),
            "goal_image": torch.randn(batch_size, 3, 96, 96),
            "goal_mask": torch.tensor([False]),
            "action_history": torch.randn(
                batch_size, cls.config.action_history_length, 3
            ),
            "action_history_mask": torch.ones(
                batch_size, cls.config.action_history_length, dtype=torch.bool
            ),
            "dt_s": torch.full((batch_size, 1), 1.0 / 3.0),
            "waypoint_spacing_m": torch.full((batch_size, 1), 0.12),
        }
        with torch.inference_mode():
            cls.h0 = cls.model(cls.batch, head="h0_deterministic")

    def test_image_to_h0_shapes_and_finiteness(self) -> None:
        self.assertEqual(tuple(self.h0["context"].shape), (1, self.config.d_model))
        self.assertEqual(tuple(self.h0["waypoints_m"].shape), (1, 8, 2))
        self.assertEqual(tuple(self.h0["progress_m"].shape), (1, 1))
        self.assertTrue(bool(torch.isfinite(self.h0["waypoints_m"]).all()))

    def test_h1_velocity_and_sampling_interfaces(self) -> None:
        with torch.inference_mode():
            velocity = self.model(
                self.batch,
                head="h1_velocity",
                x_t=torch.zeros(1, 8, 2),
                t=torch.full((1, 1), 0.5),
            )
            first = self.model(
                self.batch,
                head="h1_rectified_flow",
                nfe=2,
                candidate_count=2,
                seed=123,
            )["waypoint_candidates_m"]
            second = self.model(
                self.batch,
                head="h1_rectified_flow",
                nfe=2,
                candidate_count=2,
                seed=123,
            )["waypoint_candidates_m"]
        self.assertEqual(tuple(velocity["velocity"].shape), (1, 8, 2))
        self.assertEqual(tuple(first.shape), (1, 2, 8, 2))
        self.assertTrue(torch.equal(first, second))

    def test_single_shared_encoder_and_parameter_budget(self) -> None:
        self.assertFalse(hasattr(self.model, "goal_encoder"))
        self.assertGreater(self.model.encoder_parameter_count(), 0)
        self.assertLess(
            self.model.trainable_parameter_count(), self.config.max_trainable_parameters
        )

    def test_pretrained_weights_are_blocked(self) -> None:
        with self.assertRaisesRegex(TinyNavImagePolicyError, "pretrained weights are disabled"):
            TinyNavBrainImagePolicy(self.config, encoder_weights="download")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
