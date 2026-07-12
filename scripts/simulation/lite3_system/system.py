from __future__ import annotations

import atexit
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import select
import sys
import time

import cv2
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
from research.policy_backend.runtime import NavigationPolicyBackend


class Lite3System:
    def __init__(
        self,
        args,
        platform: NavigationPlatformBase | None = None,
        result_prefix: str = "lite3_state_machine",
        close_platform_on_finalize: bool = True,
        session: NavigationSession | None = None,
        high_level_backend: NavigationPolicyBackend | None = None,
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
        self.logs_dir = getattr(self.session, "logs_dir", self.session.run_dir / "logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.timing_csv_path = self.logs_dir / "timing_profile.csv"
        self.timing_text_path = self.logs_dir / "timing_profile.txt"
        self.high_level = high_level_backend if high_level_backend is not None else Lite3HighLevelNoMaD(
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
            action_scale_m=getattr(args, "policy_action_scale_m", 1.0),
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
            transform_fn=self.high_level.preprocess_frame,
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
        self.collision_detected = False
        self.fall_detected = False
        self.stuck_detected = False
        self.video_dir = None
        self.video_path = None
        self._video_writer = None
        self._video_size = None
        self._video_frames = 0
        self._video_started_at = None
        self._last_video_frame_at = 0.0
        self._video_lock = threading.Lock()
        self._atexit_registered = False
        self.result_prefix = result_prefix
        self.close_platform_on_finalize = close_platform_on_finalize
        self.is_standing = bool(getattr(self.platform, "_nomad_is_standing", False))
        self.stand_counter = 0
        # The right-side viewer/video panel tracks the subgoal that actually
        # conditions the latest NoMaD action, not the mission endpoint.
        self.last_subgoal_node: int | None = None
        self.last_subgoal_view = None
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
        self._async_viewer_enabled = (
            self.camera_enabled
            and self.platform.environment_domain() == "real"
            and bool(getattr(args, "async_camera_viewer", True))
        )
        self._viewer_stop = threading.Event()
        self._viewer_lock = threading.Lock()
        self._viewer_key_lock = threading.Lock()
        self._viewer_key_queue: list[int] = []
        self._viewer_thread = None
        self._viewer_state = {"extra_text": "", "goal_view": None, "footer_lines": []}
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
            self.video_dir = self.session.start_video_recording(label=f"{result_prefix}_{args.mode}")
            self.video_path = self.video_dir / "navigation_record.mp4"
            print(f"[Recorder] Saving run video to: {self.video_path.resolve()}")
        self._start_async_viewer()

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
        self.collision_detected = self.collision_detected or self.platform.has_navigation_collision()

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

    def _start_async_viewer(self) -> None:
        if not self._async_viewer_enabled:
            return
        fps = max(1.0, float(getattr(self.args, "camera_viewer_fps", 15.0)))
        self._viewer_thread = threading.Thread(
            target=self._async_viewer_loop,
            name="lite3-real-camera-viewer",
            args=(fps,),
            daemon=True,
        )
        self._viewer_thread.start()
        print(f"[Camera] Async real-camera viewer enabled at {fps:.1f} FPS.")

    def _stop_async_viewer(self) -> None:
        self._viewer_stop.set()
        if self._viewer_thread is not None and self._viewer_thread.is_alive():
            self._viewer_thread.join(timeout=2.0)
        self._viewer_thread = None

    def _update_async_viewer_state(self, extra_text: str, goal_view=None, footer_lines=None) -> None:
        if not self._async_viewer_enabled:
            return
        with self._viewer_lock:
            self._viewer_state = {
                "extra_text": extra_text,
                "goal_view": goal_view,
                "footer_lines": list(footer_lines or []),
            }

    def _queue_async_viewer_key(self, key_code: int) -> None:
        if key_code == 255:
            return
        with self._viewer_key_lock:
            self._viewer_key_queue.append(key_code)
            if len(self._viewer_key_queue) > 16:
                self._viewer_key_queue = self._viewer_key_queue[-16:]

    def _pop_async_viewer_key(self) -> int:
        with self._viewer_key_lock:
            if not self._viewer_key_queue:
                return 255
            return self._viewer_key_queue.pop(0)

    def _drain_async_viewer_ui_keys(self, camera_image) -> None:
        while True:
            key_code = self._pop_async_viewer_key()
            if key_code == 255:
                return
            self._handle_ui_key(key_code, camera_image)

    def _async_viewer_loop(self, fps: float) -> None:
        period = 1.0 / max(fps, 1.0)
        consecutive_failures = 0
        # 注意：这里吞掉单次异常（相机抖动、imshow 偶发失败等）只打印告警继续循环，
        # 不再 break——否则一次相机错误会让整段录像和右栏显示同时停摆。
        while not self._viewer_stop.is_set():
            loop_start = time.perf_counter()
            with self._viewer_lock:
                state = dict(self._viewer_state)
            try:
                camera_image = self.platform.render_camera()
                display_goal = state.get("goal_view")
                key_code = self.legacy.show_fpv_realtime(
                    camera_image,
                    self.tick,
                    str(state.get("extra_text") or ""),
                    goal_view=display_goal,
                    footer_lines=state.get("footer_lines") or [],
                )
                if self._should_write_wall_clock_video_frame():
                    try:
                        self._write_run_video_frame(camera_image, display_goal, label=self.state_name)
                    except Exception as write_exc:
                        if not self._viewer_stop.is_set():
                            print(
                                f"[Recorder] Async viewer write skipped ({type(write_exc).__name__}): {write_exc}"
                            )
                if key_code != 255:
                    self._queue_async_viewer_key(key_code)
                    if key_code == 27:
                        self.session.request_exit()
                    elif key_code in (ord("e"), ord("E")):
                        self.session.request_estop()
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                if not self._viewer_stop.is_set() and consecutive_failures <= 3:
                    print(
                        f"[Camera] Async viewer iteration failed ({type(exc).__name__}): {exc}; "
                        f"continuing (consecutive failures={consecutive_failures})"
                    )
                if consecutive_failures >= 60:
                    # 60 次连续失败（约 4 秒@15Hz）才放弃，给临时性的相机/显示故障留出恢复机会。
                    if not self._viewer_stop.is_set():
                        print("[Camera] Async viewer giving up after 60 consecutive failures.")
                    break
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, period - elapsed))

    def _show_keyboard_camera(self, camera_image) -> int:
        if not self.camera_enabled:
            return 255
        if self._async_viewer_enabled:
            self._update_async_viewer_state(
                extra_text=(
                    f"KEYBOARD | vx={self.keyboard_vx:.2f} "
                    f"vy={self.keyboard_vy:.2f} wz={self.keyboard_wz:.2f}"
                ),
                goal_view=None,
                footer_lines=[self._keyboard_help_line()],
            )
            key_code = self._pop_async_viewer_key()
            return key_code
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
        if self._should_write_wall_clock_video_frame():
            try:
                self._write_run_video_frame(camera_image, label="keyboard", every_n=1)
            except Exception as exc:
                print(f"[Recorder] Keyboard video write skipped ({type(exc).__name__}): {exc}")
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
            self.fall_detected = True
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

    def _compose_video_frame(self, camera_image, display_goal=None):
        # 固定输出"FPV + goal"双面板尺寸，无论 display_goal 是否到位：
        # 这样 cv2.VideoWriter 在第一帧就锁定到双面板宽度，后续不会因尺寸跳变而被
        # resize 压扁或丢帧；任务尚未提供子目标时，右半使用同尺寸的灰色占位面板。
        from PIL import Image as _PILImage

        fpv = camera_image.convert("RGB")
        if display_goal is None:
            goal = _PILImage.new("RGB", fpv.size, (32, 32, 32))
        else:
            goal = display_goal.convert("RGB")
            if goal.size != fpv.size:
                resampling = getattr(type(goal), "Resampling", None)
                if resampling is None:
                    resample_mode = getattr(_PILImage, "Resampling", _PILImage).BILINEAR
                else:
                    resample_mode = resampling.BILINEAR
                goal = goal.resize(fpv.size, resample_mode)
        combined = np.concatenate([np.asarray(fpv), np.asarray(goal)], axis=1)
        return cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    def _write_video_metadata(self, finished: bool = False) -> None:
        if self.video_dir is None or self.video_path is None:
            return
        payload = {
            "video_path": str(self.video_path.resolve()),
            "video_dir": str(self.video_dir.resolve()),
            "frames": int(self._video_frames),
            "fps": float(getattr(self.args, "record_fps", 10.0)),
            "mode": self.args.mode,
            "map": getattr(self.args, "map", None),
            "backend": self.platform.environment_domain(),
            "started_at": self._video_started_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "finished": bool(finished),
        }
        (self.video_dir / "recording_meta.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _should_write_wall_clock_video_frame(self) -> bool:
        if self.video_path is None:
            return False
        record_fps = max(1.0, float(getattr(self.args, "record_fps", 10.0)))
        now = time.perf_counter()
        if now - self._last_video_frame_at < 1.0 / record_fps:
            return False
        self._last_video_frame_at = now
        return True

    def _open_video_writer(self, size: tuple[int, int], fps: float):
        """按优先级尝试 H264(avc1) → mp4v → MJPG(.avi) 三种 codec，返回成功打开的 writer。"""
        candidates = [
            ("avc1", str(self.video_path)),
            ("mp4v", str(self.video_path)),
            ("MJPG", str(self.video_path.with_suffix(".avi"))),
        ]
        for codec, path in candidates:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(path, fourcc, fps, size)
            if writer.isOpened():
                if path != str(self.video_path):
                    print(f"[Recorder] Codec {codec} fallback active; saving to: {path}")
                    self.video_path = Path(path)
                else:
                    print(f"[Recorder] Codec={codec}, fps={fps}, size={size}")
                return writer
            writer.release()
        return None

    def _write_run_video_frame(self, camera_image, display_goal=None, label: str | None = None, every_n: int = 1) -> None:
        if self.video_path is None:
            return
        if every_n > 1 and self.tick % every_n != 0:
            return
        frame = self._compose_video_frame(camera_image, display_goal)
        height, width = frame.shape[:2]
        with self._video_lock:
            if self._video_writer is None:
                fps = max(1.0, float(getattr(self.args, "record_fps", 10.0)))
                self._video_size = (width, height)
                self._video_started_at = datetime.now().isoformat(timespec="seconds")
                self._video_writer = self._open_video_writer(self._video_size, fps)
                if self._video_writer is None:
                    print(f"[Recorder] Unable to open video writer at any codec: {self.video_path}")
                    self.video_path = None  # 一次失败后不再反复尝试，避免每帧都打印告警
                    return
                # 注册一次性 atexit，确保 Ctrl+C / 异常退出时也会 release writer，
                # 否则 mp4v/avc1 容器没有写入 moov atom，整段视频只能播前几秒。
                if not self._atexit_registered:
                    atexit.register(self._close_video_writer_safe)
                    self._atexit_registered = True
                self._write_video_metadata(finished=False)
            if (width, height) != self._video_size:
                frame = cv2.resize(frame, self._video_size, interpolation=cv2.INTER_AREA)
            self._video_writer.write(frame)
            self._video_frames += 1

    def _close_video_writer_safe(self) -> None:
        """atexit 钩子：能 release 就 release，吞掉所有异常防止退出报错。"""
        try:
            self._close_video_writer()
        except Exception as exc:
            print(f"[Recorder] atexit close failed ({type(exc).__name__}): {exc}")

    def _close_video_writer(self) -> None:
        with self._video_lock:
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
            self._write_video_metadata(finished=True)

    def _profile_timing(self, label: str, timings: dict[str, float]) -> None:
        if not bool(getattr(self.args, "profile_timing", False)):
            return
        interval = max(1, int(getattr(self.args, "profile_interval", 10)))
        should_print = self.tick % interval == 0
        timing_text = " ".join(f"{name}={value * 1000.0:.1f}ms" for name, value in timings.items())
        fps_parts = []
        infer_time = float(timings.get("infer", 0.0))
        total_time = float(timings.get("total", 0.0))
        infer_fps = None
        loop_fps = None
        if infer_time > 1e-9:
            infer_fps = 1.0 / infer_time
            fps_parts.append(f"infer_fps={infer_fps:.2f}")
        if total_time > 1e-9:
            loop_fps = 1.0 / total_time
            fps_parts.append(f"loop_fps={loop_fps:.2f}")
        fps_text = f" | {' '.join(fps_parts)}" if fps_parts else ""
        camera_status = self.platform.camera_status()
        suffix = f" | {camera_status}" if camera_status else ""
        line = f"[Timing] {label} tick={self.tick} {timing_text}{fps_text}{suffix}"
        if should_print:
            print(line)
        self._write_timing_profile(label, timings, infer_fps, loop_fps, camera_status, line)

    def _write_timing_profile(
        self,
        label: str,
        timings: dict[str, float],
        infer_fps: float | None,
        loop_fps: float | None,
        camera_status: str,
        line: str,
    ) -> None:
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        fields = [
            "timestamp",
            "label",
            "tick",
            "camera_ms",
            "ui_ms",
            "preprocess_ms",
            "infer_ms",
            "command_ms",
            "total_ms",
            "infer_fps",
            "loop_fps",
            "camera_status",
        ]
        row = {
            "timestamp": timestamp,
            "label": label,
            "tick": int(self.tick),
            "camera_ms": float(timings.get("camera", 0.0)) * 1000.0,
            "ui_ms": float(timings.get("ui", 0.0)) * 1000.0,
            "preprocess_ms": float(timings.get("preprocess", 0.0)) * 1000.0,
            "infer_ms": float(timings.get("infer", 0.0)) * 1000.0,
            "command_ms": float(timings.get("command", 0.0)) * 1000.0,
            "total_ms": float(timings.get("total", 0.0)) * 1000.0,
            "infer_fps": "" if infer_fps is None else infer_fps,
            "loop_fps": "" if loop_fps is None else loop_fps,
            "camera_status": camera_status,
        }
        try:
            write_header = not self.timing_csv_path.exists()
            with self.timing_csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            with self.timing_text_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            print(f"[Timing] Unable to write timing log ({type(exc).__name__}): {exc}")

    def _show_camera(self, camera_image, extra_text: str, goal_view=None) -> None:
        key_code = 255
        display_goal = self._display_goal_view(goal_view)
        if self._async_viewer_enabled:
            self._update_async_viewer_state(extra_text, display_goal, self._footer_lines())
            self._drain_async_viewer_ui_keys(camera_image)
        elif self.camera_enabled:
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
        goal_for_save = display_goal if self.state_name == "navigate" else None
        # Sync 路径（MuJoCo / async 关闭）也走 wall-clock pacing：
        # 这样视频帧率稳定为 record_fps，避免推理慢/快导致视频比真实时长短或抖动。
        if not self._async_viewer_enabled and self._should_write_wall_clock_video_frame():
            try:
                self._write_run_video_frame(camera_image, goal_for_save, label=label, every_n=1)
            except Exception as exc:
                print(f"[Recorder] Sync video write skipped ({type(exc).__name__}): {exc}")

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
            self.fall_detected = True
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
            self.stuck_detected = True
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
        # 任务切换后清空实时子目标缓存，避免显示上一段 mission 的节点
        self.last_subgoal_node = None
        self.last_subgoal_view = None
        # 但是立刻用即将开始任务的"下一节点"作为 viewer 占位推送一次，避免
        # navigate 起始瞬间 async viewer 还在显示 stand 阶段的 None placeholder。
        if current_goal is not None and current_goal.topomap is not None and len(current_goal.topomap) > 0:
            preview_idx = int(np.clip(self.missions.closest_node + 1, 0, len(current_goal.topomap) - 1))
            preview_view = current_goal.topomap[preview_idx]
            if self._async_viewer_enabled:
                self._update_async_viewer_state(
                    extra_text=f"NAVIGATE | mission={current_goal.label} | warming up",
                    goal_view=preview_view,
                    footer_lines=self._footer_lines(),
                )

    def step_navigation(self) -> str:
        mission = self._current_mission()
        if mission is None:
            return "failed"

        tick_start = time.perf_counter()
        camera_image = self.platform.render_camera()
        after_camera = time.perf_counter()
        forward_speed = self.platform.get_forward_speed()
        # 计算可视化用的实时子目标：
        #   - 优先使用上一次推理实际条件化的节点 (self.last_subgoal_node)
        #   - 任务刚启动尚未推理时，回退为 closest_node 的下一个节点 (NoMaD 即将瞄准的"下一站")
        #   - 仅当列表越界等异常情况下才退化为 mission 终点
        subgoal_node_for_display = self.last_subgoal_node
        if subgoal_node_for_display is None:
            subgoal_node_for_display = min(self.missions.closest_node + 1, self.missions.goal_node)
        subgoal_node_for_display = int(np.clip(subgoal_node_for_display, 0, len(mission.topomap) - 1))
        subgoal_view_for_display = self.last_subgoal_view
        if subgoal_view_for_display is None:
            subgoal_view_for_display = mission.topomap[subgoal_node_for_display]
        self._show_camera(
            camera_image,
            extra_text=(
                f"NAVIGATE | mission={mission.label} | "
                f"node={self.missions.closest_node}/{self.missions.goal_node} | "
                f"subgoal={subgoal_node_for_display} | v_body={forward_speed:.3f}"
            ),
            goal_view=subgoal_view_for_display,
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
        # 记录本次推理实际条件化的 topomap 节点，下一周期的相机叠加显示与录像将使用它，
        # 而不是 mission 的最终目标图像。
        selected_node_clamped = int(np.clip(result.selected_node, 0, len(mission.topomap) - 1))
        self.last_subgoal_node = selected_node_clamped
        self.last_subgoal_view = mission.topomap[selected_node_clamped]
        # 立刻把新子目标推回异步 viewer 状态，避免右栏要等下一次 step_navigation 才刷新——
        # 在 ~6Hz 的高层节奏下，这一推送可以让右栏在 ~67ms 内（async viewer 15Hz）完成切换，
        # 否则用户看到的右栏会"卡"在上一周期的子目标上接近 150ms。
        if self._async_viewer_enabled:
            self._update_async_viewer_state(
                extra_text=(
                    f"NAVIGATE | mission={mission.label} | "
                    f"node={self.missions.closest_node}/{self.missions.goal_node} | "
                    f"subgoal={selected_node_clamped} | v_body={forward_speed:.3f}"
                ),
                goal_view=self.last_subgoal_view,
                footer_lines=self._footer_lines(),
            )
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
        self._stop_async_viewer()
        self._close_video_writer()
        if self.video_path is not None:
            print(f"[Recorder] Video saved to: {self.video_path.resolve()}")
        if not bool(getattr(self.args, "suppress_legacy_results", False)):
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
            try:
                self.legacy.run_generate_topomap(self.args)
            finally:
                self._stop_async_viewer()
                self._close_video_writer()
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
                self.fall_detected = True
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
