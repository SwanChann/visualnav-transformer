from __future__ import annotations

import numpy as np

from lite3_system.interfaces import MotionCommand


class BaseState:
    name = "base"

    def on_enter(self, system) -> None:
        return None

    def on_exit(self, system) -> None:
        return None

    def step(self, system):
        raise NotImplementedError


class IdleState(BaseState):
    name = "idle"

    def step(self, system):
        if getattr(system, "keyboard_exit_requested", False):
            return "completed"
        return "stand"


class StandState(BaseState):
    name = "stand"

    def on_enter(self, system) -> None:
        if not system.is_standing:
            system.platform.standup(system.args.standup_time)
            system.is_standing = True
            setattr(system.platform, "_nomad_is_standing", True)
        system.stand_counter = 0

    def step(self, system):
        if system.session.estop_requested:
            return "estop"
        if system.args.mode == "stand":
            command = MotionCommand(0.0, 0.0, 0.0)
            system.platform.apply_command(command)
            position, _ = system.platform.get_pose()
            system.record_step(position, command)
            system.tick += 1
            system.show_task_camera("STAND")
            system.stand_counter += 1
            if system.stand_counter >= max(1, int(system.args.stand_steps)):
                return "completed"
            failure_state = system._common_failure_check()
            return failure_state or "stand"
        if system.args.mode in {"navigate", "mission"}:
            return "navigate"
        if system.args.mode == "explore":
            return "explore"
        if system.args.mode == "keyboard":
            return "keyboard"
        if system.args.mode == "estop":
            return "estop"
        return "walk"


class NavigateState(BaseState):
    name = "navigate"

    def on_enter(self, system) -> None:
        system.refresh_goal_visualization()

    def step(self, system):
        if system.session.estop_requested:
            return "estop"
        return system.step_navigation()


class ExploreState(BaseState):
    name = "explore"

    def step(self, system):
        if system.session.estop_requested:
            return "estop"
        return system.step_exploration()


class WalkState(BaseState):
    name = "walk"

    def step(self, system):
        if system.session.estop_requested:
            return "estop"
        failure_state = system._common_failure_check()
        if failure_state is not None:
            return failure_state
        command = MotionCommand(0.5, 0.0, 0.0)
        system.platform.apply_command(command)
        position, _ = system.platform.get_pose()
        system.record_step(position, command)
        system.tick += 1
        system.show_task_camera("WALK")
        if system.session.exit_requested:
            system.platform.stop_motion()
            return "completed"
        if system.tick >= system.args.max_steps:
            return "completed"
        return "walk"


class KeyboardState(BaseState):
    name = "keyboard"

    def on_enter(self, system) -> None:
        system.enter_keyboard_mode()

    def on_exit(self, system) -> None:
        system.exit_keyboard_mode()

    def step(self, system):
        return system.step_keyboard()


class RecoveryState(BaseState):
    name = "recovery"

    def on_enter(self, system) -> None:
        system.recovery_counter = 0
        system.recovery_phase = "back"

    def step(self, system):
        if system.session.estop_requested:
            return "estop"
        failure_state = system._common_failure_check()
        if failure_state is not None:
            return failure_state

        if system.recovery_phase == "back":
            command = MotionCommand(system.legacy.RECOVERY_BACK_VEL, 0.0, 0.0)
            phase_limit = system.legacy.RECOVERY_BACK_STEPS
            phase_label = "RECOVERY-BACK"
        else:
            command = MotionCommand(0.05, 0.0, system.recovery_turn_direction * system.legacy.RECOVERY_TURN_VEL)
            phase_limit = system.legacy.RECOVERY_TURN_STEPS
            phase_label = "RECOVERY-TURN"

        system.platform.apply_command(command)
        position, _ = system.platform.get_pose()
        system.record_step(position, command)
        system.tick += 1
        system.show_task_camera(phase_label)
        if system.session.exit_requested:
            system.platform.stop_motion()
            return "completed"
        system.recovery_counter += 1

        if system.recovery_counter >= phase_limit:
            if system.recovery_phase == "back":
                system.recovery_phase = "turn"
                system.recovery_counter = 0
                system.recovery_turn_direction = 1.0 if np.random.random() > 0.5 else -1.0
                return "recovery"
            system.stuck_detector.reset()
            return system.resume_state
        return "recovery"


class EstopState(BaseState):
    name = "estop"

    def on_enter(self, system) -> None:
        system.safe_stop()

    def step(self, system):
        command = MotionCommand(0.0, 0.0, 0.0)
        position, _ = system.platform.get_pose()
        system.record_step(position, command)
        system.tick += 1
        system.show_task_camera("ESTOP")
        return "completed"


class CompletedState(BaseState):
    name = "completed"

    def on_enter(self, system) -> None:
        if system.args.mode == "estop" or system.session.estop_requested:
            return
        system.controlled_stop()

    def step(self, system):
        return "completed"


class FailedState(BaseState):
    name = "failed"

    def on_enter(self, system) -> None:
        system.safe_stop()

    def step(self, system):
        return "failed"
