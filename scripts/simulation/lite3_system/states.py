from __future__ import annotations

import numpy as np

from lite3_system.interfaces import MotionCommand


class BaseState:
    name = "base"

    def on_enter(self, system) -> None:
        return None

    def step(self, system):
        raise NotImplementedError


class IdleState(BaseState):
    name = "idle"

    def step(self, system):
        return "standup"


class StandupState(BaseState):
    name = "standup"

    def on_enter(self, system) -> None:
        system.platform.standup(system.args.standup_time)

    def step(self, system):
        if system.args.mode in {"navigate", "mission"}:
            return "navigate"
        if system.args.mode == "explore":
            return "explore"
        return "walk"


class NavigateState(BaseState):
    name = "navigate"

    def step(self, system):
        return system.step_navigation()


class ExploreState(BaseState):
    name = "explore"

    def step(self, system):
        return system.step_exploration()


class WalkState(BaseState):
    name = "walk"

    def step(self, system):
        failure_state = system._common_failure_check()
        if failure_state is not None:
            return failure_state
        command = MotionCommand(0.5, 0.0, 0.0)
        system.platform.apply_command(command)
        position, _ = system.platform.get_pose()
        system.record_step(position, command)
        system.tick += 1
        if system.platform.is_fallen() or not system.platform.viewer_alive():
            return "failed"
        if system.tick >= system.args.max_steps:
            return "completed"
        return "walk"


class RecoveryBackState(BaseState):
    name = "recovery_back"

    def on_enter(self, system) -> None:
        system.recovery_counter = 0

    def step(self, system):
        failure_state = system._common_failure_check()
        if failure_state is not None:
            return failure_state
        command = MotionCommand(system.legacy.RECOVERY_BACK_VEL, 0.0, 0.0)
        system.platform.apply_command(command)
        system.recovery_counter += 1
        position, _ = system.platform.get_pose()
        system.record_step(position, command)
        system.tick += 1
        if system.recovery_counter >= system.legacy.RECOVERY_BACK_STEPS:
            system.recovery_turn_direction = 1.0 if np.random.random() > 0.5 else -1.0
            return "recovery_turn"
        return "recovery_back"


class RecoveryTurnState(BaseState):
    name = "recovery_turn"

    def on_enter(self, system) -> None:
        system.recovery_counter = 0

    def step(self, system):
        failure_state = system._common_failure_check()
        if failure_state is not None:
            return failure_state
        command = MotionCommand(0.05, 0.0, system.recovery_turn_direction * system.legacy.RECOVERY_TURN_VEL)
        system.platform.apply_command(command)
        system.recovery_counter += 1
        position, _ = system.platform.get_pose()
        system.record_step(position, command)
        system.tick += 1
        if system.recovery_counter >= system.legacy.RECOVERY_TURN_STEPS:
            system.stuck_detector.reset()
            return system.resume_state
        return "recovery_turn"


class CompletedState(BaseState):
    name = "completed"

    def on_enter(self, system) -> None:
        system.safe_stop()

    def step(self, system):
        return "completed"


class FailedState(BaseState):
    name = "failed"

    def on_enter(self, system) -> None:
        system.safe_stop()

    def step(self, system):
        return "failed"
