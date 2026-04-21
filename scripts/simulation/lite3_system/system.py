from __future__ import annotations

from pathlib import Path

import numpy as np

from lite3_system.interfaces import ContextBuffer, Lite3HighLevelNoMaD, Lite3LowLevelPlatform, Lite3MiddleLayerPD, MotionCommand, NavigationPlatformBase
from lite3_system.legacy_bridge import load_legacy
from lite3_system.session import NavigationSession
from lite3_system.states import (
    CompletedState,
    EstopState,
    ExploreState,
    FailedState,
    IdleState,
    NavigateState,
    RecoveryState,
    StandState,
    WalkState,
)
from lite3_system.topomap import MissionQueue, build_mission_queue


class Lite3System:
    def __init__(
        self,
        args,
        platform: NavigationPlatformBase | None = None,
        result_prefix: str = "lite3_state_machine",
        close_platform_on_finalize: bool = True,
        session: NavigationSession | None = None,
    ) -> None:
        self.args = args
        self.legacy = load_legacy()
        self.legacy.SCENE_CONFIG = self.legacy.SCENE_MAPS[args.map]
        self.platform = platform or Lite3LowLevelPlatform(gui=not args.no_gui, scene_name=args.map)
        self.platform.prepare_task(args.mode)
        self.session = session or NavigationSession(run_label=result_prefix)
        self.high_level = Lite3HighLevelNoMaD(
            scheduler_kind=args.scheduler,
            ddim_steps=args.ddim_steps,
            cfg_weight=getattr(args, "cfg_weight", 0.0),
            tts_enabled=bool(getattr(args, "tts", False)),
            tts_budget=int(getattr(args, "tts_budget", 8)),
            tts_topk=int(getattr(args, "tts_topk", 1)),
            tts_verifier=str(getattr(args, "tts_verifier", "heuristic")),
            policy_config=getattr(args, "policy_config", None),
            policy_checkpoint=getattr(args, "policy_checkpoint", None),
            device=getattr(args, "policy_device", None),
            image_resize_mode=getattr(args, "image_resize_mode", "stretch"),
            waypoint_index=args.waypoint,
            radius=args.radius,
            close_threshold=args.close_threshold,
        )
        self.middle_layer = Lite3MiddleLayerPD(yaw_sign=getattr(args, "yaw_sign", 1.0))
        self.context = ContextBuffer(
            max_frames=self.high_level.context_size + 1,
            transform_fn=self.high_level.inference.pil_to_tensor,
        )
        self.stuck_detector = self.legacy.StuckDetector()
        self.missions: MissionQueue | None = build_mission_queue(args, self.platform, self.session)
        self.trajectory = []
        self.velocity_log = []
        self.tick = 0
        self.recovery_counter = 0
        self.recovery_phase = "back"
        self.recovery_turn_direction = 1.0
        self.recovery_count_total = 0
        self.resume_state = "navigate" if args.mode in {"navigate", "mission"} else "explore"
        self.goal_reached = False
        self.fp_dir = None
        self.result_prefix = result_prefix
        self.close_platform_on_finalize = close_platform_on_finalize
        self.is_standing = bool(getattr(self.platform, "_nomad_is_standing", False))
        self.stand_counter = 0
        self.camera_enabled = bool(self.legacy.camera_visualization_enabled(args))
        self.capture_enabled = bool(getattr(args, "capture_enabled", False))
        self.mujoco_route_stabilizer = bool(getattr(args, "mujoco_route_stabilizer", True))
        self._mujoco_reference_path = None
        self.states = {
            "idle": IdleState(),
            "stand": StandState(),
            "navigate": NavigateState(),
            "explore": ExploreState(),
            "walk": WalkState(),
            "recovery": RecoveryState(),
            "estop": EstopState(),
            "completed": CompletedState(),
            "failed": FailedState(),
        }
        self.state_name = "idle"
        self.state = self.states[self.state_name]
        self.state.on_enter(self)
        if args.save_fpv:
            self.fp_dir = self.session.fpv_dir
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

    def safe_stop(self) -> None:
        self.platform.emergency_stop()

    def controlled_stop(self) -> None:
        # 中文注释：普通任务结束只发送零速度，不能触发真机软急停。
        self.platform.apply_command(MotionCommand(0.0, 0.0, 0.0))

    def _footer_lines(self) -> list[str]:
        if not self.capture_enabled:
            return ["e estop | capture handled by navigation host"]
        capture = self.session.get_selected_capture()
        if capture is None:
            queue_line = "captures=0 | press c to capture | [,] switch | e estop"
        else:
            queue_line = (
                f"captures={len(self.session.capture_queue)} | selected={capture.index}:{capture.label} "
                "| c capture | [,] switch | e estop"
            )
        return [queue_line]

    def _handle_ui_key(self, key_code: int, camera_image) -> None:
        if key_code == 255:
            return
        if self.capture_enabled and key_code in (ord("c"), ord("C")):
            position, yaw = self.platform.get_pose()
            capture = self.session.add_capture(
                image=camera_image,
                position=position,
                yaw=yaw,
                map_name=getattr(self.args, "map", None),
                label_prefix=f"{self.args.map}_{self.state_name}",
            )
            print(f"[Capture] Stored {capture.label} -> {capture.saved_path}")
            return
        if self.capture_enabled and key_code in (ord("["), ord("{")):
            capture = self.session.select_previous_capture()
            if capture is not None:
                print(f"[Capture] Selected previous capture: {capture.label}")
            return
        if self.capture_enabled and key_code in (ord("]"), ord("}")):
            capture = self.session.select_next_capture()
            if capture is not None:
                print(f"[Capture] Selected next capture: {capture.label}")
            return
        if key_code in (ord("e"), ord("E"), 27):
            print("[EStop] Keyboard estop requested.")
            self.session.request_estop()

    def _display_goal_view(self, goal_view):
        if goal_view is not None:
            return goal_view
        selected_capture = self.session.get_selected_capture()
        if self.capture_enabled and selected_capture is not None:
            return selected_capture.image
        return None

    def _show_camera(self, camera_image, extra_text: str, goal_view=None) -> None:
        key_code = 255
        if self.camera_enabled:
            key_code = self.legacy.show_fpv_realtime(
                camera_image,
                self.tick,
                extra_text,
                goal_view=self._display_goal_view(goal_view),
                footer_lines=self._footer_lines(),
            )
        self._handle_ui_key(key_code, camera_image)
        if self.fp_dir and self.tick % 5 == 0:
            camera_image.save(self.fp_dir / f"{self.tick:04d}.png")

    def show_task_camera(self, task_label: str, goal_view=None) -> None:
        camera_image = self.platform.render_camera()
        forward_speed = self.platform.get_forward_speed()
        self._show_camera(
            camera_image,
            extra_text=f"{task_label} | v_body={forward_speed:.3f}",
            goal_view=goal_view,
        )

    def _common_failure_check(self) -> str | None:
        if self.platform.is_fallen():
            print(
                f"[System] Failure: platform reported fallen "
                f"(tick={self.tick}, height={self.platform.get_height():.3f})"
            )
            return "failed"
        if not self.platform.viewer_alive():
            print(f"[System] Failure: viewer is no longer alive at tick={self.tick}")
            return "failed"
        if self.tick >= self.args.max_steps:
            return "completed"
        return None

    def _handle_stuck(self, actual_velocity: float, command: MotionCommand) -> str | None:
        self.stuck_detector.update(actual_velocity, command.linear_x)
        if self.stuck_detector.is_stuck():
            self.recovery_count_total += 1
            return "recovery"
        return None

    def _stabilize_mujoco_route(
        self,
        command: MotionCommand,
        position: np.ndarray,
        yaw: float,
        goal_position: np.ndarray | None,
    ) -> MotionCommand:
        if (
            goal_position is None
            or self.platform.environment_domain() != "mujoco"
            or not self.mujoco_route_stabilizer
        ):
            return command
        if self._mujoco_reference_path is None:
            self._mujoco_reference_path = np.asarray(
                self.legacy.build_scene_reference_path(self.legacy.SCENE_CONFIG, num_points=600),
                dtype=float,
            )
        path = self._mujoco_reference_path
        delta = np.asarray(goal_position, dtype=float) - np.asarray(position, dtype=float)
        goal_dist = float(np.linalg.norm(delta))
        if goal_dist <= 2.0:
            target = np.asarray(goal_position, dtype=float)
        else:
            path_index = int(self.legacy.project_position_to_path_index(position, path))
            lookahead_index = min(path_index + 35, len(path) - 1)
            target = np.asarray(path[lookahead_index], dtype=float)
        target_delta = target - np.asarray(position, dtype=float)
        desired_yaw = float(np.arctan2(target_delta[1], target_delta[0]))
        yaw_error = float(np.arctan2(np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw)))
        yaw_rate = float(np.clip(1.5 * yaw_error, -0.6, 0.6))
        linear_x = min(command.linear_x, 0.35)
        if goal_dist < 1.0:
            linear_x = min(linear_x, 0.16)
        if abs(yaw_error) > 0.55:
            linear_x = min(linear_x, 0.08)
        return MotionCommand(linear_x=linear_x, linear_y=command.linear_y, yaw_rate=yaw_rate)

    def _current_mission(self):
        if self.missions is None:
            return None
        return self.missions.current

    def refresh_goal_visualization(self) -> None:
        if self.missions is None:
            return
        goal_positions = [
            mission.goal_position for mission in self.missions.missions if mission.goal_position is not None
        ]
        if goal_positions:
            self.platform.set_goal_markers(goal_positions, active_index=self.missions.index)
        current_goal = self._current_mission()
        if current_goal is not None and current_goal.goal_view is not None:
            saved_path = self.session.save_goal_view(current_goal.goal_view, current_goal.label)
            setattr(current_goal, "saved_goal_path", saved_path)

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

        if self.session.estop_requested:
            return "estop"

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
        pre_position, pre_yaw = self.platform.get_pose()
        command = self._stabilize_mujoco_route(command, pre_position, pre_yaw, mission.goal_position)
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

        reached_goal = reached_by_physics if mission.goal_position is not None else reached_by_node
        if reached_goal:
            if self.missions.advance():
                self.refresh_goal_visualization()
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

        if self.session.estop_requested:
            return "estop"

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
        mode_name = f"{self.result_prefix}_{self.args.mode}"
        self.legacy._save_results(
            self.trajectory,
            self.velocity_log,
            mode_name,
            self.args,
            reached_goal=self.goal_reached if self.args.mode in {"navigate", "mission"} else None,
        )
        if self.close_platform_on_finalize:
            self.platform.close()

    def run(self) -> int:
        if self.args.mode == "generate-topomap":
            self.legacy.run_generate_topomap(self.args)
            self.platform.close()
            return 0

        if self.args.mode in {"navigate", "mission"} and self.missions is None:
            raise ValueError("Navigation mode requires at least one topomap mission.")

        if self.args.mode in {"navigate", "mission"}:
            self.refresh_goal_visualization()

        current_name = self.state_name
        while current_name not in {"completed", "failed"}:
            next_name = self.state.step(self)
            self.transition_to(next_name)
            current_name = self.state_name

        self.finalize()
        if current_name == "completed" and self.args.mode in {"stand", "explore", "walk-test", "estop"}:
            return 0
        if current_name == "completed" and self.goal_reached:
            return 0
        if current_name == "completed" and self.args.mode in {"navigate", "mission"}:
            return 1
        return 1 if current_name == "failed" else 0
