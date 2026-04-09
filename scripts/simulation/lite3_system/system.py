from __future__ import annotations

from pathlib import Path

import numpy as np

from lite3_system.interfaces import ContextBuffer, Lite3HighLevelNoMaD, Lite3LowLevelPlatform, Lite3MiddleLayerPD, MotionCommand
from lite3_system.legacy_bridge import load_legacy
from lite3_system.states import (
    CompletedState,
    ExploreState,
    FailedState,
    IdleState,
    NavigateState,
    RecoveryBackState,
    RecoveryTurnState,
    StandupState,
    WalkState,
)
from lite3_system.topomap import MissionQueue, build_mission_queue


class Lite3System:
    def __init__(self, args) -> None:
        self.args = args
        self.legacy = load_legacy()
        self.legacy.SCENE_CONFIG = self.legacy.SCENE_MAPS[args.map]
        self.platform = Lite3LowLevelPlatform(gui=not args.no_gui, scene_name=args.map)
        self.high_level = Lite3HighLevelNoMaD(
            scheduler_kind=args.scheduler,
            ddim_steps=args.ddim_steps,
            cfg_weight=getattr(args, "cfg_weight", 0.0),
            waypoint_index=args.waypoint,
            radius=args.radius,
            close_threshold=args.close_threshold,
        )
        self.middle_layer = Lite3MiddleLayerPD()
        self.context = ContextBuffer(max_frames=self.legacy.CONTEXT_SIZE + 1)
        self.stuck_detector = self.legacy.StuckDetector()
        self.missions: MissionQueue | None = build_mission_queue(args, self.platform)
        self.trajectory = []
        self.velocity_log = []
        self.tick = 0
        self.recovery_counter = 0
        self.recovery_turn_direction = 1.0
        self.recovery_count_total = 0
        self.resume_state = "navigate" if args.mode in {"navigate", "mission"} else "explore"
        self.goal_reached = False
        self.fp_dir = None
        self.states = {
            "idle": IdleState(),
            "standup": StandupState(),
            "navigate": NavigateState(),
            "explore": ExploreState(),
            "walk": WalkState(),
            "recovery_back": RecoveryBackState(),
            "recovery_turn": RecoveryTurnState(),
            "completed": CompletedState(),
            "failed": FailedState(),
        }
        self.state_name = "idle"
        self.state = self.states[self.state_name]
        self.state.on_enter(self)
        if args.save_fpv:
            run_tag = self.legacy.RUN_TAG
            mode_name = "lite3_state_machine"
            self.fp_dir = Path(self.legacy.PROJECT_ROOT) / "results" / "nomad_mujoco" / f"{run_tag}_{mode_name}" / "fpv"
            self.fp_dir.mkdir(parents=True, exist_ok=True)

    def transition_to(self, state_name: str) -> None:
        if state_name == self.state_name:
            return
        self.state_name = state_name
        self.state = self.states[state_name]
        self.state.on_enter(self)

    def record_step(self, position, command: MotionCommand) -> None:
        self.trajectory.append(np.asarray(position, dtype=float).copy())
        self.velocity_log.append((command.linear_x, command.yaw_rate))

    def _show_camera(self, camera_image, extra_text: str, goal_view=None) -> None:
        self.legacy.show_fpv_realtime(camera_image, self.tick, extra_text, goal_view=goal_view)
        if self.fp_dir and self.tick % 5 == 0:
            camera_image.save(self.fp_dir / f"{self.tick:04d}.png")

    def _common_failure_check(self) -> str | None:
        if self.platform.is_fallen():
            return "failed"
        if not self.platform.viewer_alive():
            return "failed"
        if self.tick >= self.args.max_steps:
            return "completed"
        return None

    def _handle_stuck(self, actual_velocity: float, command: MotionCommand) -> str | None:
        self.stuck_detector.update(actual_velocity, command.linear_x)
        if self.stuck_detector.is_stuck():
            self.recovery_count_total += 1
            return "recovery_back"
        return None

    def _current_mission(self):
        if self.missions is None:
            return None
        return self.missions.current

    def step_navigation(self) -> str:
        mission = self._current_mission()
        if mission is None:
            return "failed"

        camera_image = self.platform.render_camera()
        forward_speed = self.platform.get_forward_speed()
        self._show_camera(
            camera_image,
            extra_text=f"NAVIGATE | mission={mission.label} | node={self.missions.closest_node}/{self.missions.goal_node} | v_body={forward_speed:.3f}",
            goal_view=mission.goal_view,
        )
        self.context.push(camera_image)
        self.tick += 1

        if not self.context.ready():
            self.platform.apply_command(MotionCommand(0.0, 0.0, 0.0))
            return self._common_failure_check() or "navigate"

        result = self.high_level.predict_navigation(
            self.context.frames,
            mission.topomap,
            self.missions.closest_node,
            self.missions.goal_node,
        )
        self.missions.closest_node = result.closest_node
        command = self.middle_layer.waypoint_to_command(result.chosen_waypoint)
        self.platform.apply_command(command)
        position, _ = self.platform.get_pose()
        self.record_step(position, command)
        actual_velocity = self.platform.get_forward_speed()

        failure_state = self._common_failure_check()
        if failure_state is not None:
            return failure_state
        stuck_state = self._handle_stuck(actual_velocity, command)
        if stuck_state is not None:
            self.resume_state = "navigate"
            return stuck_state

        reached_by_node = result.selected_node >= self.missions.goal_node and result.predicted_distance < self.args.close_threshold
        reached_by_physics = False
        if mission.goal_position is not None:
            reached_by_physics = np.linalg.norm(np.asarray(position) - mission.goal_position) < self.legacy.GOAL_REACH_DIST

        if reached_by_node or reached_by_physics:
            if self.missions.advance():
                return "navigate"
            self.goal_reached = True
            return "completed"
        return "navigate"

    def step_exploration(self) -> str:
        camera_image = self.platform.render_camera()
        forward_speed = self.platform.get_forward_speed()
        self._show_camera(camera_image, extra_text=f"EXPLORE | v_body={forward_speed:.3f}")
        self.context.push(camera_image)
        self.tick += 1

        if not self.context.ready():
            self.platform.apply_command(MotionCommand(0.0, 0.0, 0.0))
            return self._common_failure_check() or "explore"

        result = self.high_level.predict_exploration(self.context.frames)
        command = self.middle_layer.waypoint_to_command(result.chosen_waypoint)
        self.platform.apply_command(command)
        position, _ = self.platform.get_pose()
        self.record_step(position, command)
        actual_velocity = self.platform.get_forward_speed()

        failure_state = self._common_failure_check()
        if failure_state is not None:
            return failure_state
        stuck_state = self._handle_stuck(actual_velocity, command)
        if stuck_state is not None:
            self.resume_state = "explore"
            return stuck_state
        return "explore"

    def finalize(self) -> None:
        mode_name = f"lite3_state_machine_{self.args.mode}"
        self.legacy._save_results(
            self.trajectory,
            self.velocity_log,
            mode_name,
            self.args,
            reached_goal=self.goal_reached if self.args.mode in {"navigate", "mission"} else None,
        )
        self.platform.close()

    def run(self) -> int:
        if self.args.mode == "generate-topomap":
            self.legacy.run_generate_topomap(self.args)
            self.platform.close()
            return 0

        if self.args.mode in {"navigate", "mission"} and self.missions is None:
            raise ValueError("Navigation mode requires at least one topomap mission.")

        current_name = self.state_name
        while current_name not in {"completed", "failed"}:
            next_name = self.state.step(self)
            self.transition_to(next_name)
            current_name = self.state_name

        self.finalize()
        if current_name == "completed" and self.goal_reached:
            return 0
        if current_name == "completed" and self.args.mode in {"explore", "walk-test"}:
            return 0
        return 1 if current_name == "failed" else 0
