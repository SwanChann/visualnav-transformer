from __future__ import annotations

import random
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from train_step_runtime import (
    DeterministicBatchCursor,
    TrainStepRuntimeError,
    capture_rng_state,
    compute_action_loss,
    restore_rng_state,
)


class FixturePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h0 = nn.Parameter(torch.zeros(2, 3, 2))
        self.h1 = nn.Parameter(torch.zeros(2, 3, 2))

    def forward(self, batch, *, head, x_t=None, t=None):
        if head == "h0_deterministic":
            return {"waypoints_m": self.h0}
        if head == "h1_velocity":
            return {"velocity": self.h1 + 0.0 * x_t + 0.0 * t[:, :, None]}
        raise ValueError(head)


class TrainStepRuntimeTests(unittest.TestCase):
    def batch(self):
        return {
            "target_waypoints": torch.ones(2, 3, 2),
            "target_mask": torch.ones(2, 3, dtype=torch.bool),
            "dataset_id": ["a", "b"],
        }

    def test_h0_h1_losses_are_finite_without_backward(self):
        model = FixturePolicy()
        h0, h0_by_dataset = compute_action_loss(
            model, self.batch(), head="h0_deterministic"
        )
        generator = torch.Generator().manual_seed(7)
        h1, h1_by_dataset = compute_action_loss(
            model, self.batch(), head="h1_rectified_flow", generator=generator
        )
        self.assertTrue(torch.isfinite(h0))
        self.assertTrue(torch.isfinite(h1))
        self.assertEqual(set(h0_by_dataset), {"a", "b"})
        self.assertEqual(set(h1_by_dataset), {"a", "b"})
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_batch_cursor_round_trip_and_tamper_rejection(self):
        source = DeterministicBatchCursor(20, 4, 3)
        source.next_indices()
        state = source.state_dict()
        expected = source.next_indices()
        resumed = DeterministicBatchCursor(20, 4, 3)
        resumed.load_state_dict(state)
        self.assertEqual(resumed.next_indices(), expected)
        state["order_sha256"] = "0" * 64
        with self.assertRaisesRegex(TrainStepRuntimeError, "checksum"):
            resumed.load_state_dict(state)

    def test_rng_round_trip_without_backward(self):
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), torch.rand(4))
        restore_rng_state(state)
        actual = (random.random(), float(np.random.rand()), torch.rand(4))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for mapped RNG regression")
    def test_rng_restore_accepts_cuda_mapped_byte_tensors(self):
        torch.manual_seed(13)
        state = capture_rng_state()
        state["torch_cpu"] = state["torch_cpu"].to("cuda:0")
        state["torch_cuda"] = [item.to("cuda:0") for item in state["torch_cuda"]]
        restore_rng_state(state)


if __name__ == "__main__":
    unittest.main()
