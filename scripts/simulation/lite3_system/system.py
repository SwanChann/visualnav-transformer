from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import select
import sys
import time

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
    KeyboardState,
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
        self.platform = platform or Lite3LowLevelPlatform(gui=not args.no_gui, scene_name=args.map)
        if args.map in self.legacy.SCENE_MAPS:
            self.legacy.SCENE_CONFIG = self.legacy.SCENE_MAPS[args.map]
        elif self.platform.environment_domain() == "mujoco":
            raise ValueError(f"Unknown MuJoCo map: {args.map}")
        else:
            # 中文注释：真机 backend 没有 MuJoCo 场景，保留默认仿真配置只给少数复用工具兜底。
            self.legacy.SCENE_CONFIG = self.legacy.SCENE_MAPS["easy"]
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
        self.middle_layer = Lite3MiddleLayerPD(
            yaw_sign=getattr(args, "yaw_sign", 1.0),
            linear_scale=getattr(args, "pd_linear_scale", 1.0),
            yaw_scale=getattr(args, "pd_yaw_scale", 1.0),
            max_linear=getattr(args, "pd_max_v", None),
            max_yaw=getattr(args, "pd_max_w", None),
        )
        self.context = ContextBuffer(
            max_frames=self.high_level.context_size + 1,
            transform_fn=self.high_level.inference.pil_to_tensor,
        )
        self.stuck_detector = self.legacy.StuckDetector()
        self.missions: MissionQueue | None = build_mission_queue(args, self.platform, self.session)
        self._prepare_mission_topomap_tensors()
        self.trajectory = []
        self.velocity_log = []
        self.tick = 0
        self.recovery_counter = 0
        self.recovery_phase = "back"
        self.recovery_turn_direction = 1.0
        self.recovery_count_total = 0
        self.resume_state = "navigate" if args.mode in {"navigate", "mission"} else "explore"
        self.goal_reached = False
        self.image_save_dir = None
        self.result_prefix = result_prefix
        self.close_platform_on_finalize = close_platform_on_finalize
        self.is_standing = bool(getattr(self.platform, "_nomad_is_standing", False))
        self.stand_counter = 0
        self.keyboard_exit_requested = False
        self.keyboard_vx = 0.0
        self.keyboard_vy = 0.0
        self.keyboard_wz = 0.0
        self._keyboard_old_terminal_settings = None
        self.keyboard_heartbeat_path = self._resolve_keyboard_heartbeat_path()
        self._keyboard_heartbeat_started_at = None
        self.camera_enabled = bool(self.legacy.camera_visualization_enabled(args))
        self.capture_enabled = bool(getattr(args, "capture_enabled", False))
        self.mujoco_route_stabilizer = bool(getattr(args, "mujoco_route_stabilizer", True))
        self._mujoco_reference_path = None
        self.states = {
            "idle": IdleState(),
            "stand": StandState(),
            "navigate": NavigateState(),
            "explore": ExploreState(),
            "keyboard": KeyboardState(),
            "walk": WalkState(),
            "recovery": RecoveryState(),
            "estop": EstopState(),
            "completed": CompletedState(),
            "failed": FailedState(),
        }
        self.state_name = "idle"
        self.state = self.states[self.state_name]
        self.state.on_enter(self)
        if bool(getattr(args, "save_images", False)):
            self.image_save_dir = self.session.start_image_recording(label=f"{result_prefix}_{args.mode}")
            print(f"[Recorder] Saving run images to: {self.image_save_dir}")

    def _prepare_mission_topomap_tensors(self) -> None:
        if self.missions is None:
            return
        for mission in self.missions.missions:
            if getattr(mission, "topomap_tensor", None) is None:
                mission.topomap_tensor = self.high_level.build_topomap_tensor(mission.topomap)

    def transition_to(self, state_name: str) -> None:
        if state_name == self.state_name:
            return
        self.state.on_exit(self)
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
        self.platform.stop_motion()
        self.platform.release_manual_control()

    def _resolve_keyboard_heartbeat_path(self) -> Path:
        raw_path = getattr(self.args, "keyboard_heartbeat_file", None)
        path = Path(raw_path or "results/deployment/navigation_host_keyboard_active.json").expanduser()
        if not path.is_absolute():
            path = Path(self.legacy.PROJECT_ROOT) / path
        return path.resolve()

    def _write_keyboard_heartbeat(self) -> None:
        self.keyboard_heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": "keyboard",
            "state": self.state_name,
            "pid": os.getpid(),
            "domain": self.platform.environment_domain(),
            "map": getattr(self.args, "map", None),
            "camera": getattr(self.args, "camera", None),
            "session_run_dir": str(self.session.run_dir),
            "started_at": self._keyboard_heartbeat_started_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at_epoch": time.time(),
            "tick": int(self.tick),
        }
        tmp_path = self.keyboard_heartbeat_path.with_suffix(self.keyboard_heartbeat_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(self.keyboard_heartbeat_path)

    def _clear_keyboard_heartbeat(self) -> None:
        try:
            self.keyboard_heartbeat_path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"[Keyboard] Unable to remove heartbeat file: {exc}")

    def enter_keyboard_mode(self) -> None:
        self.keyboard_exit_requested = False
        self.keyboard_vx = 0.0
        self.keyboard_vy = 0.0
        self.keyboard_wz = 0.0
        self._keyboard_heartbeat_started_at = datetime.now().isoformat(timespec="seconds")
        self._enable_terminal_keyboard_mode()
        self.platform.prepare_for_twist_control()
        self._write_keyboard_heartbeat()
        print(
            "\n[Keyboard] Enter keyboard mode: W/S forward/back, A/D strafe, "
            "Q/E turn, 0 stop, Space estop, U stand, P prepare, 1/2/3 gait, I state, ESC idle.\n"
        )
        print(f"[Keyboard] Heartbeat: {self.keyboard_heartbeat_path}")

    def exit_keyboard_mode(self) -> None:
        self._restore_terminal_keyboard_mode()
        self.platform.stop_motion()
        self.platform.release_manual_control()
        self._clear_keyboard_heartbeat()

    def _enable_terminal_keyboard_mode(self) -> None:
        if sys.platform == "win32" or not sys.stdin.isatty():
            return
        try:
            import termios
            import tty

            self._keyboard_old_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception as exc:
            print(f"[Keyboard] Terminal raw mode unavailable: {type(exc).__name__}: {exc}")
            self._keyboard_old_terminal_settings = None

    def _restore_terminal_keyboard_mode(self) -> None:
        if sys.platform == "win32" or self._keyboard_old_terminal_settings is None:
            return
        try:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._keyboard_old_terminal_settings)
        finally:
            self._keyboard_old_terminal_settings = None

    def _read_console_key(self) -> str | None:
        if sys.platform == "win32":
            try:
                import msvcrt

                if not msvcrt.kbhit():
                    return None
                key = msvcrt.getch()
                if key in (b"\xe0", b"\x00"):
                    if msvcrt.kbhit():
                        msvcrt.getch()
                    return None
                return key.decode("utf-8", errors="ignore").lower()
            except Exception:
                return None
        if not sys.stdin.isatty():
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                return None
            return sys.stdin.read(1).lower()
        except Exception:
            return None

    def _keyboard_help_line(self) -> str:
        return "KEYBOARD | W/S A/D Q/E | 0 stop | Space estop | U stand | P prepare | 1/2/3 gait | ESC idle"

    def _show_keyboard_camera(self, camera_image) -> int:
        if not self.camera_enabled:
            return 255
        key_code = self.legacy.show_fpv_realtime(
            camera_image,
            self.tick,
            extra_text=(
                f"KEYBOARD | vx={self.keyboard_vx:.2f} "
                f"vy={self.keyboard_vy:.2f} wz={self.keyboard_wz:.2f}"
            ),
            goal_view=None,
            footer_lines=[self._keyboard_help_line()],
        )
        self._save_run_image(camera_image, label="keyboard", every_n=5)
        return key_code

    def _handle_keyboard_key(self, key: str | None) -> None:
        if not key:
            return
        vx_step = float(getattr(self.args, "keyboard_vx_step", 0.2))
        vy_step = float(getattr(self.args, "keyboard_vy_step", 0.15))
        wz_step = float(getattr(self.args, "keyboard_wz_step", 0.3))
        max_vx = float(getattr(self.args, "keyboard_max_vx", 1.0))
        max_vy = float(getattr(self.args, "keyboard_max_vy", 0.5))
        max_wz = float(getattr(self.args, "keyboard_max_wz", 1.5))

        if key == "\x1b":
            print("[Keyboard] ESC pressed; leaving keyboard mode and returning to idle.")
            self.keyboard_exit_requested = True
            return
        if key == "\x03":
            print("[Keyboard] Ctrl+C ignored in keyboard mode; press ESC to return to idle.")
            return
        if key == " ":
            self.keyboard_vx = self.keyboard_vy = self.keyboard_wz = 0.0
            self.platform.emergency_stop()
            print("[Keyboard] Soft estop requested.")
            return
        if key == "u":
            self.platform.standup(self.args.standup_time)
            self.is_standing = True
            setattr(self.platform, "_nomad_is_standing", True)
            print("[Keyboard] Stand command sent.")
            return
        if key == "p":
            self.platform.prepare_for_twist_control()
            print("[Keyboard] Prepared for twist control.")
            return
        if key == "0":
            self.keyboard_vx = self.keyboard_vy = self.keyboard_wz = 0.0
            self.platform.stop_motion()
            print("[Keyboard] Stop motion.")
            return
        if key == "1":
            self.platform.set_gait("low")
            print("[Keyboard] Gait: low")
            return
        if key == "2":
            self.platform.set_gait("mid")
            print("[Keyboard] Gait: mid")
            return
        if key == "3":
            self.platform.set_gait("high")
            print("[Keyboard] Gait: high")
            return
        if key == "i":
            position, yaw = self.platform.get_pose()
            print(
                f"[Keyboard] pos=({position[0]:.3f}, {position[1]:.3f}) "
                f"yaw={yaw:.3f} height={self.platform.get_height():.3f} "
                f"vx={self.keyboard_vx:.2f} vy={self.keyboard_vy:.2f} wz={self.keyboard_wz:.2f}"
            )
            return
        if key == "w":
            self.keyboard_vx = min(max_vx, self.keyboard_vx + vx_step)
        elif key == "s":
            self.keyboard_vx = max(-max_vx, self.keyboard_vx - vx_step)
        elif key == "a":
            self.keyboard_vy = min(max_vy, self.keyboard_vy + vy_step)
        elif key == "d":
            self.keyboard_vy = max(-max_vy, self.keyboard_vy - vy_step)
        elif key == "q":
            self.keyboard_wz = min(max_wz, self.keyboard_wz + wz_step)
        elif key == "e":
            self.keyboard_wz = max(-max_wz, self.keyboard_wz - wz_step)
        else:
            return
        print(
            f"[Keyboard] vx={self.keyboard_vx:.2f}, "
            f"vy={self.keyboard_vy:.2f}, wz={self.keyboard_wz:.2f}"
        )

    def step_keyboard(self) -> str:
        camera_key = 255
        if self.camera_enabled:
            camera_image = self.platform.render_camera()
            camera_key = self._show_keyboard_camera(camera_image)
        console_key = self._read_console_key()

        key = console_key
        if camera_key != 255:
            key = chr(camera_key).lower() if 0 <= camera_key < 256 else key
        self._handle_keyboard_key(key)

        if self.keyboard_exit_requested:
            self.platform.stop_motion()
            return "idle"

        if self.platform.is_fallen():
            self.keyboard_vx = self.keyboard_vy = self.keyboard_wz = 0.0
            self.platform.stop_motion()
            print("[Keyboard] Platform reported fallen; motion stopped, staying in keyboard mode until ESC.")
            time.sleep(float(getattr(self.args, "keyboard_dt", 0.05)))
            return "keyboard"

        command = MotionCommand(self.keyboard_vx, self.keyboard_vy, self.keyboard_wz)
        self.platform.apply_command(command)
        self._write_keyboard_heartbeat()
        position, _ = self.platform.get_pose()
        self.record_step(position, command)
        self.tick += 1

        time.sleep(float(getattr(self.args, "keyboard_dt", 0.05)))
        return "keyboard"

    def _footer_lines(self) -> list[str]:
        base_exit = "ESC exit | e estop"
        if self.state_name == "navigate" and self.missions is not None:
            goal_line = (
                f"{base_exit} | [/] goal node {self.missions.goal_node}/"
                f"{len(self.missions.current.topomap) - 1}"
            )
            capture_line = "c capture | ,/. switch captured display"
            return [goal_line, capture_line]
        if not self.capture_enabled:
            return [base_exit]
        capture = self.session.get_selected_capture()
        if capture is None:
            queue_line = f"captures=0 | c capture | {base_exit}"
        else:
            queue_line = (
                f"captures={len(self.session.capture_queue)} | selected={capture.index}:{capture.label} "
                f"| c capture | ,/. switch | {base_exit}"
            )
        return [queue_line]

    def _switch_navigation_goal_node(self, delta: int) -> bool:
        if self.state_name != "navigate" or self.missions is None:
            return False
        node = self.missions.shift_goal_node(delta)
        current = self.missions.current
        saved_path = self.session.save_goal_view(current.goal_view, f"{current.label}_node_{node:03d}")
        setattr(current, "saved_goal_path", saved_path)
        print(
            f"[Goal] Switched topomap goal to node {node}/"
            f"{len(current.topomap) - 1}: {saved_path}"
        )
        return True

    def _handle_ui_key(self, key_code: int, camera_image) -> None:
        if key_code == 255:
            return
        if key_code == 27:
            print("[Exit] ESC pressed; leaving current mode.")
            self.session.request_exit()
            return
        if key_code in (ord("e"), ord("E")):
            print("[EStop] Keyboard estop requested.")
            self.session.request_estop()
            return
        if key_code in (ord("["), ord("{")) and self._switch_navigation_goal_node(-1):
            return
        if key_code in (ord("]"), ord("}")) and self._switch_navigation_goal_node(1):
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
        if self.capture_enabled and key_code in (ord(","), ord("<"), ord("["), ord("{")):
            capture = self.session.select_previous_capture()
            if capture is not None:
                print(f"[Capture] Selected previous capture: {capture.label}")
            return
        if self.capture_enabled and key_code in (ord("."), ord(">"), ord("]"), ord("}")):
            capture = self.session.select_next_capture()
            if capture is not None:
                print(f"[Capture] Selected next capture: {capture.label}")
            return

    def _display_goal_view(self, goal_view):
        if goal_view is not None:
            return goal_view
        selected_capture = self.session.get_selected_capture()
        if self.capture_enabled and selected_capture is not None:
            return selected_capture.image
        return None

    def _save_run_image(self, camera_image, display_goal=None, label: str | None = None, every_n: int = 1) -> None:
        if self.image_save_dir is None:
            return
        if every_n > 1 and self.tick % every_n != 0:
            return
        safe_label = label or self.state_name
        if display_goal is not None:
            self.session.save_goal_pair(camera_image, display_goal, self.image_save_dir, self.tick, label=safe_label)
        else:
            self.session.save_camera_image(camera_image, self.image_save_dir, self.tick, label=safe_label)

    def _profile_timing(self, label: str, timings: dict[str, float]) -> None:
        if not bool(getattr(self.args, "profile_timing", False)):
            return
        interval = max(1, int(getattr(self.args, "profile_interval", 10)))
        if self.tick % interval != 0:
            return
        timing_text = " ".join(f"{name}={value * 1000.0:.1f}ms" for name, value in timings.items())
        camera_status = self.platform.camera_status()
        suffix = f" | {camera_status}" if camera_status else ""
        print(f"[Timing] {label} tick={self.tick} {timing_text}{suffix}")

    def _show_camera(self, camera_image, extra_text: str, goal_view=None) -> None:
        key_code = 255
        display_goal = self._display_goal_view(goal_view)
        if self.camera_enabled:
            key_code = self.legacy.show_fpv_realtime(
                camera_image,
                self.tick,
                extra_text,
                goal_view=display_goal,
                footer_lines=self._footer_lines(),
            )
        self._handle_ui_key(key_code, camera_image)
        mission = self._current_mission()
        label = mission.label if self.state_name == "navigate" and mission is not None else self.state_name
        every_n = 1 if self.state_name == "navigate" and display_goal is not None else 5
        goal_for_save = display_goal if self.state_name == "navigate" else None
        self._save_run_image(camera_image, goal_for_save, label=label, every_n=every_n)

    def show_task_camera(self, task_label: str, goal_view=None) -> None:
        camera_image = self.platform.render_camera()
        forward_speed = self.platform.get_forward_speed()
        self._show_camera(
            camera_image,
            extra_text=f"{task_label} | v_body={forward_speed:.3f}",
            goal_view=goal_view,
        )

    def _common_failure_check(self) -> str | None:
        if self.session.exit_requested:
            self.platform.stop_motion()
            return "completed"
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

        tick_start = time.perf_counter()
        camera_image = self.platform.render_camera()
        after_camera = time.perf_counter()
        forward_speed = self.platform.get_forward_speed()
        self._show_camera(
            camera_image,
            extra_text=f"NAVIGATE | mission={mission.label} | node={self.missions.closest_node}/{self.missions.goal_node} | v_body={forward_speed:.3f}",
            goal_view=mission.goal_view,
        )
        after_ui = time.perf_counter()
        if self.session.exit_requested:
            self.platform.stop_motion()
            return "completed"
        self.context.push(camera_image)
        self.tick += 1
        after_preprocess = time.perf_counter()

        if self.session.estop_requested:
            return "estop"

        if not self.context.ready():
            self.platform.apply_command(MotionCommand(0.0, 0.0, 0.0))
            after_command = time.perf_counter()
            self._profile_timing(
                "navigate",
                {
                    "camera": after_camera - tick_start,
                    "ui": after_ui - after_camera,
                    "preprocess": after_preprocess - after_ui,
                    "infer": 0.0,
                    "command": after_command - after_preprocess,
                    "total": after_command - tick_start,
                },
            )
            return self._common_failure_check() or "navigate"

        try:
            before_infer = time.perf_counter()
            result = self.high_level.predict_navigation(
                self.context.frames,
                mission.topomap,
                self.missions.closest_node,
                self.missions.goal_node,
                topomap_tensor=getattr(mission, "topomap_tensor", None),
            )
            after_infer = time.perf_counter()
        except Exception as exc:
            print(f"[System] ⚠️ predict_navigation raised {type(exc).__name__}: {exc}; issuing safe stop and failing")
            self.safe_stop()
            return "failed"
        self.missions.closest_node = result.closest_node
        command = self.middle_layer.waypoint_to_command(result.chosen_waypoint)
        pre_position, pre_yaw = self.platform.get_pose()
        command = self._stabilize_mujoco_route(command, pre_position, pre_yaw, mission.goal_position)
        self.platform.apply_command(command)
        after_command = time.perf_counter()
        self._profile_timing(
            "navigate",
            {
                "camera": after_camera - tick_start,
                "ui": after_ui - after_camera,
                "preprocess": after_preprocess - after_ui,
                "infer": after_infer - before_infer,
                "command": after_command - after_infer,
                "total": after_command - tick_start,
            },
        )
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
        tick_start = time.perf_counter()
        camera_image = self.platform.render_camera()
        after_camera = time.perf_counter()
        forward_speed = self.platform.get_forward_speed()
        self._show_camera(camera_image, extra_text=f"EXPLORE | v_body={forward_speed:.3f}")
        after_ui = time.perf_counter()
        if self.session.exit_requested:
            self.platform.stop_motion()
            return "completed"
        self.context.push(camera_image)
        self.tick += 1
        after_preprocess = time.perf_counter()

        if self.session.estop_requested:
            return "estop"

        if not self.context.ready():
            self.platform.apply_command(MotionCommand(0.0, 0.0, 0.0))
            after_command = time.perf_counter()
            self._profile_timing(
                "explore",
                {
                    "camera": after_camera - tick_start,
                    "ui": after_ui - after_camera,
                    "preprocess": after_preprocess - after_ui,
                    "infer": 0.0,
                    "command": after_command - after_preprocess,
                    "total": after_command - tick_start,
                },
            )
            return self._common_failure_check() or "explore"

        try:
            before_infer = time.perf_counter()
            result = self.high_level.predict_exploration(self.context.frames)
            after_infer = time.perf_counter()
        except Exception as exc:
            print(f"[System] ⚠️ predict_exploration raised {type(exc).__name__}: {exc}; issuing safe stop and failing")
            self.safe_stop()
            return "failed"
        command = self.middle_layer.waypoint_to_command(result.chosen_waypoint)
        self.platform.apply_command(command)
        after_command = time.perf_counter()
        self._profile_timing(
            "explore",
            {
                "camera": after_camera - tick_start,
                "ui": after_ui - after_camera,
                "preprocess": after_preprocess - after_ui,
                "infer": after_infer - before_infer,
                "command": after_command - after_infer,
                "total": after_command - tick_start,
            },
        )
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
        _safeguarded_states = {"estop", "failed", "completed", "recovery", "keyboard"}
        while current_name not in {"completed", "failed"}:
            # 全局摔倒保护：除了已在 estop/failed/recovery 的状态外，任何状态
            # 检测到 is_fallen 都立刻切 failed，避免 stand/idle 等状态
            # 遗漏摔倒判定。
            if current_name not in _safeguarded_states and self.platform.is_fallen():
                print(
                    f"[System] Global safety trip: platform reported fallen in state={current_name}, "
                    f"tick={self.tick}, height={self.platform.get_height():.3f}"
                )
                self.transition_to("failed")
                current_name = self.state_name
                continue
            next_name = self.state.step(self)
            self.transition_to(next_name)
            current_name = self.state_name

        self.finalize()
        if current_name == "completed" and self.args.mode in {"stand", "explore", "keyboard", "walk-test", "estop"}:
            return 0
        if current_name == "completed" and self.session.exit_requested:
            return 0
        if current_name == "completed" and self.goal_reached:
            return 0
        if current_name == "completed" and self.args.mode in {"navigate", "mission"}:
            return 1
        return 1 if current_name == "failed" else 0
