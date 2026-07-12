#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


SIMULATION_ROOT = Path(__file__).resolve().parents[1] / "simulation"
if str(SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATION_ROOT))

from lite3_system.interfaces import MotionCommand
from lite3_system.system import Lite3System


class FakePlatform:
    def __init__(self, *, fallen=False, collision=False):
        self.fallen = fallen
        self.collision = collision

    def is_fallen(self): return self.fallen
    def has_navigation_collision(self): return self.collision
    def viewer_alive(self): return True
    def stop_motion(self): return None
    def get_height(self): return 0.05 if self.fallen else 0.3


class FakeStuckDetector:
    def update(self, *_args): return None
    def is_stuck(self): return True


class Lite3FailureCaptureTest(unittest.TestCase):
    def bare_system(self) -> Lite3System:
        system = Lite3System.__new__(Lite3System)
        system.platform = FakePlatform()
        system.session = SimpleNamespace(exit_requested=False)
        system.args = SimpleNamespace(max_steps=10)
        system.tick = 0
        system.trajectory = []
        system.velocity_log = []
        system.collision_detected = False
        system.fall_detected = False
        system.stuck_detected = False
        system.recovery_count_total = 0
        return system

    def test_collision_is_latched_by_record_step(self) -> None:
        system = self.bare_system()
        system.platform.collision = True
        system.record_step([0.0, 0.0], MotionCommand(0.1, 0.0, 0.0))
        self.assertTrue(system.collision_detected)

    def test_fall_is_latched_and_fails(self) -> None:
        system = self.bare_system()
        system.platform.fallen = True
        self.assertEqual(system._common_failure_check(), "failed")
        self.assertTrue(system.fall_detected)

    def test_stuck_is_latched_and_enters_recovery(self) -> None:
        system = self.bare_system()
        system.stuck_detector = FakeStuckDetector()
        self.assertEqual(system._handle_stuck(0.0, MotionCommand(0.2, 0.0, 0.0)), "recovery")
        self.assertTrue(system.stuck_detected)
        self.assertEqual(system.recovery_count_total, 1)

    def test_timeout_completes_without_success(self) -> None:
        system = self.bare_system()
        system.tick = system.args.max_steps
        self.assertEqual(system._common_failure_check(), "completed")


if __name__ == "__main__":
    unittest.main()
