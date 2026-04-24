#!/usr/bin/env python3
"""Unified NoMaD navigation host for MuJoCo and real-robot backends.

支持两种运行模式：
  1. 批量模式（默认）：从命令行或 JSON 计划文件加载预定义任务序列
  2. 在线交互模式（--interactive）：启动后进入 idle 状态，通过 stdin 实时接收任务
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = CURRENT_DIR.parent
SIM_ROOT = SCRIPTS_ROOT / "simulation"
for candidate in (SCRIPTS_ROOT, SIM_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from nomad_mujoco_lite3_state_machine import build_parser as build_state_machine_parser
from project_paths import repo_path
from lite3_system.topomap import validate_task_topomap_args
from lite3_system.session import NavigationSession


RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class HostTaskResult:
    index: int
    mode: str
    map_name: str
    backend: str
    exit_code: int
    status: str


def _normalize_bridge_kwargs(payload: dict | None) -> dict:
    """Accept either a raw kwargs dict or a wrapped {"bridge_kwargs": {...}} payload."""
    if not payload:
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("bridge_kwargs"), dict):
        return dict(payload["bridge_kwargs"])
    return dict(payload)


class Lite3NavigationHost:
    def __init__(self, host_args, task_args_list: list[argparse.Namespace]) -> None:
        self.host_args = host_args
        self.task_args_list = task_args_list
        self.results: list[HostTaskResult] = []
        self.session = NavigationSession(run_label=f"{self.host_args.platform}_{self.host_args.backend}_host")
        self._shared_platform = None
        self._shared_mujoco_map = None
        self._validate_plan_consistency()

    def _load_bridge_kwargs(self) -> dict:
        if not self.host_args.bridge_config:
            return {}
        config_path = Path(self.host_args.bridge_config).expanduser().resolve()
        return _normalize_bridge_kwargs(json.loads(config_path.read_text(encoding="utf-8")))

    def _validate_task(self, task_args: argparse.Namespace) -> None:
        if task_args.mode == "mission":
            has_mission = bool(
                task_args.mission_topomap
                or task_args.mission_dataset_traj
                or task_args.random
                or getattr(task_args, "goal_source", "topomap") in {"random_points", "capture_queue"}
            )
            if not has_mission:
                raise ValueError("Mission mode requires mission-topomap, goal-source=random_points/capture_queue, or --random.")

        if self.host_args.backend == "real":
            if task_args.mode == "generate-topomap":
                raise ValueError("Real backend does not support generate-topomap; collect real topomap offline first.")
            if task_args.random:
                raise ValueError("Real backend does not support random spawn/goal generation.")
            if not self.host_args.bridge_module or not self.host_args.bridge_class:
                raise ValueError("Real backend requires --bridge-module and --bridge-class.")

        validate_task_topomap_args(task_args, expected_domain=self.host_args.backend)

    def _validate_plan_consistency(self) -> None:
        if self.host_args.backend != "mujoco" or not self.task_args_list:
            return
        map_names = {task_args.map for task_args in self.task_args_list}
        if len(map_names) > 1:
            raise ValueError(
                "A single MuJoCo navigation-host run now keeps one scene instance alive across tasks. "
                f"Please use a single map per host run, but got: {sorted(map_names)}."
            )
        self._shared_mujoco_map = next(iter(map_names))

    def _build_platform(self):
        if self.host_args.backend == "mujoco":
            return None
        from lite3_system.interfaces import ExternalBridgePlatform

        bridge_kwargs = self._load_bridge_kwargs()
        bridge_kwargs.setdefault("enable_camera", self.host_args.camera == "on")
        return ExternalBridgePlatform(
            bridge_module=self.host_args.bridge_module,
            bridge_class=self.host_args.bridge_class,
            bridge_kwargs=bridge_kwargs,
        )

    def _get_platform_for_task(self, task_args: argparse.Namespace):
        if self._shared_platform is not None:
            return self._shared_platform

        if self.host_args.backend == "real":
            if self._shared_platform is None:
                self._shared_platform = self._build_platform()
            return self._shared_platform

        from lite3_system.interfaces import Lite3LowLevelPlatform

        self._shared_platform = Lite3LowLevelPlatform(gui=not bool(task_args.no_gui), scene_name=task_args.map)
        return self._shared_platform

    def _result_prefix(self) -> str:
        return f"{self.host_args.platform}_{self.host_args.backend}_navigation_host"

    def run_task(self, task_index: int, task_args: argparse.Namespace) -> HostTaskResult:
        from lite3_system.system import Lite3System

        if task_args.mode != "estop":
            self.session.clear_estop()
        task_args.capture_enabled = True
        self._validate_task(task_args)
        platform = self._get_platform_for_task(task_args)
        system = Lite3System(
            task_args,
            platform=platform,
            result_prefix=self._result_prefix(),
            close_platform_on_finalize=False,
            session=self.session,
        )
        exit_code = system.run()
        return HostTaskResult(
            index=task_index,
            mode=task_args.mode,
            map_name=task_args.map,
            backend=self.host_args.backend,
            exit_code=exit_code,
            status="success" if exit_code == 0 else "failed",
        )

    def run(self) -> int:
        if self.host_args.dry_run:
            for task_args in self.task_args_list:
                self._validate_task(task_args)
            print(self.render_summary_markdown())
            if self.host_args.save_plan:
                output_dir = repo_path("results", "deployment", f"{RUN_TAG}_{self.host_args.platform}_{self.host_args.backend}_navigation_host")
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "host_plan.json").write_text(self.render_plan_json(), encoding="utf-8")
                (output_dir / "host_summary.md").write_text(self.render_summary_markdown(), encoding="utf-8")
                print(f"[Host] Saved dry-run host plan to: {output_dir}")
            return 0

        for index, task_args in enumerate(self.task_args_list, start=1):
            print(
                f"[Host] Running task {index}/{len(self.task_args_list)}: "
                f"mode={task_args.mode} map={task_args.map} backend={self.host_args.backend}"
            )
            result = self.run_task(index, task_args)
            self.results.append(result)
            print(f"[Host] Task {index} finished with status={result.status} exit_code={result.exit_code}")
            if result.exit_code != 0 and not self.host_args.continue_on_failure:
                break

        if self.host_args.save_plan:
            output_dir = repo_path("results", "deployment", f"{RUN_TAG}_{self.host_args.platform}_{self.host_args.backend}_navigation_host")
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "host_plan.json").write_text(self.render_plan_json(), encoding="utf-8")
            (output_dir / "host_summary.md").write_text(self.render_summary_markdown(), encoding="utf-8")
            print(f"[Host] Saved host plan to: {output_dir}")

        if self._shared_platform is not None:
            self._shared_platform.close()

        return 0 if all(result.exit_code == 0 for result in self.results) else 1

    def render_plan_json(self) -> str:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "platform": self.host_args.platform,
            "backend": self.host_args.backend,
            "tasks": [vars(task_args) for task_args in self.task_args_list],
            "results": [asdict(result) for result in self.results],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def render_summary_markdown(self) -> str:
        lines = [
            "# NoMaD Navigation Host Summary",
            "",
            f"- Platform: `{self.host_args.platform}`",
            f"- Backend: `{self.host_args.backend}`",
            f"- Generated At: `{datetime.now().isoformat(timespec='seconds')}`",
            "",
            "| Task | Mode | Map | Backend | Status | Exit Code |",
            "|---:|---|---|---|---|---:|",
        ]
        if self.results:
            for result in self.results:
                lines.append(
                    f"| {result.index} | {result.mode} | {result.map_name} | "
                    f"{result.backend} | {result.status} | {result.exit_code} |"
                )
        else:
            for index, task_args in enumerate(self.task_args_list, start=1):
                lines.append(
                    f"| {index} | {task_args.mode} | {task_args.map} | "
                    f"{self.host_args.backend} | pending(dry-run) | 0 |"
                )
        lines.extend(
            [
                "",
                "## Host Design Notes",
                "",
                "1. The host is responsible for task scheduling and backend selection.",
                "2. Supported host tasks are `stand`, `keyboard`, `navigate`, `explore`, and `estop`; `mission` remains a backward-compatible alias.",
                "3. Capture is host-managed: the host keeps one shared session and enables c/[ / ] only during host runs.",
                "4. The MuJoCo backend keeps one scene instance alive across task switches, so the viewer stays open and the robot pose is continuous.",
                "5. The real backend uses one persistent bridge instance; there is no map reload concept during task switching.",
                "6. Task topomaps must match the backend domain: MuJoCo tasks use MuJoCo topomaps, and real-robot tasks use real-world topomaps.",
            ]
        )
        return "\n".join(lines)


def build_host_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified NoMaD navigation host")
    parser.add_argument("--platform", choices=["lite3"], default="lite3")
    parser.add_argument("--backend", choices=["mujoco", "real"], default="mujoco")
    parser.add_argument("--interactive", action="store_true", help="Start in online interactive mode (stdin command loop)")
    parser.add_argument(
        "--map",
        type=str,
        default=None,
        help="Map/run label. MuJoCo defaults to easy; real backend defaults to real_hallway.",
    )
    parser.add_argument("--plan-file", type=str, default=None, help="JSON plan file containing defaults and tasks")
    parser.add_argument("--bridge-module", type=str, default=None, help="Python module for the real-robot bridge")
    parser.add_argument("--bridge-class", type=str, default=None, help="Bridge class name in the selected module")
    parser.add_argument("--bridge-config", type=str, default=None, help="Optional JSON file with bridge kwargs")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue remaining tasks after one task fails")
    parser.add_argument("--save-plan", action="store_true", help="Save the host plan and run summary")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the task plan without starting NoMaD")
    parser.add_argument("--no-gui", action="store_true", help="Run headless (no MuJoCo viewer)")
    parser.add_argument("--camera", choices=["on", "off"], default="on", help="Camera visualization on/off")
    parser.add_argument("--policy-config", type=str, default=None, help="Override policy config for interactive mode")
    parser.add_argument("--policy-checkpoint", type=str, default=None, help="Override policy checkpoint for interactive mode")
    parser.add_argument("--save-fpv", action="store_true", help="Save FPV frames for interactive tasks")
    parser.add_argument(
        "--keyboard-heartbeat-file",
        default="results/deployment/navigation_host_keyboard_active.json",
        help="Heartbeat file written while keyboard mode is active.",
    )
    return parser


def _state_machine_defaults() -> dict:
    parser = build_state_machine_parser()
    return vars(parser.parse_args([]))


def finalize_host_args(host_args) -> None:
    if host_args.map is None:
        host_args.map = "real_hallway" if host_args.backend == "real" else "easy"
    if host_args.backend == "mujoco" and host_args.map not in {"easy", "medium", "hard"}:
        raise ValueError(f"MuJoCo backend only supports map easy/medium/hard, got: {host_args.map}")


def load_task_args(host_args, remaining_args: list[str]) -> list[argparse.Namespace]:
    state_parser = build_state_machine_parser()
    state_defaults = _state_machine_defaults()

    if host_args.plan_file:
        plan_path = Path(host_args.plan_file).expanduser().resolve()
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
        defaults = dict(state_defaults)
        defaults["map"] = host_args.map
        defaults.update(plan_payload.get("defaults", {}))
        task_args_list = []
        for task_payload in plan_payload.get("tasks", []):
            merged = dict(defaults)
            merged.update(task_payload)
            task_args_list.append(argparse.Namespace(**merged))
        if not task_args_list:
            raise ValueError("Plan file must contain at least one task.")
        return task_args_list

    task_args = state_parser.parse_args(remaining_args)
    if host_args.backend == "real":
        task_args.map = host_args.map
    task_args.keyboard_heartbeat_file = host_args.keyboard_heartbeat_file
    return [task_args]


# ────────────────────────────────────────────────────────────
# 在线交互模式
# ────────────────────────────────────────────────────────────

_LEGACY_INTERACTIVE_HELP = """
╔═══════════════════════════════════════════════════════════╗
║             NoMaD 导航主机 — 在线交互模式                  ║
╠═══════════════════════════════════════════════════════════╣
║ 命令:                                                     ║
║   navigate --topomap-dir DIR 使用真实 topomap 导航到目标   ║
║   explore [--max-steps N]   无目标探索                     ║
║   stand [--stand-steps N]   原地站立                       ║
║   estop                     紧急停止                       ║
║   capture                   拍照存入队列                   ║
║   status                    查看当前状态                   ║
║   help                      显示帮助                       ║
║   quit / exit               退出                          ║
╚═══════════════════════════════════════════════════════════╝
""".strip()


INTERACTIVE_HELP = """
NoMaD navigation host - interactive mode

Commands:
  keyboard                  keyboard-control Lite3; ESC returns to idle
  navigate --topomap-dir DIR navigate with a real/sim topomap
  explore [--max-steps N]   goal-free exploration
  stand [--stand-steps N]   stand in place
  estop                     emergency stop
  capture                   capture an image into the host queue
  status                    print current status
  help                      show this help
  quit / exit               quit host
""".strip()


class InteractiveNavigationHost:
    """在线交互导航主机：启动后进入 idle 状态，通过 stdin 接收实时任务。"""

    def __init__(self, host_args) -> None:
        self.host_args = host_args
        self.session = NavigationSession(run_label=f"{host_args.platform}_{host_args.backend}_interactive")
        self._platform = None
        self._is_standing = False
        self.task_count = 0
        self.results: list[HostTaskResult] = []

    def _ensure_platform(self):
        if self._platform is not None:
            return self._platform
        if self.host_args.backend == "mujoco":
            from lite3_system.interfaces import Lite3LowLevelPlatform
            self._platform = Lite3LowLevelPlatform(
                gui=not self.host_args.no_gui,
                scene_name=self.host_args.map,
            )
        else:
            from lite3_system.interfaces import ExternalBridgePlatform
            bridge_kwargs = {}
            if self.host_args.bridge_config:
                config_path = Path(self.host_args.bridge_config).expanduser().resolve()
                bridge_kwargs = _normalize_bridge_kwargs(json.loads(config_path.read_text(encoding="utf-8")))
            bridge_kwargs.setdefault("enable_camera", self.host_args.camera == "on")
            self._platform = ExternalBridgePlatform(
                bridge_module=self.host_args.bridge_module,
                bridge_class=self.host_args.bridge_class,
                bridge_kwargs=bridge_kwargs,
            )
        return self._platform

    def _ensure_standing(self):
        platform = self._ensure_platform()
        if not self._is_standing:
            platform.standup(3.0)
            self._is_standing = True
            setattr(platform, "_nomad_is_standing", True)

    def _idle_step(self):
        """idle 状态：站立并渲染相机"""
        from lite3_system.interfaces import MotionCommand
        platform = self._ensure_platform()
        if self.host_args.backend == "real":
            return
        platform.apply_command(MotionCommand(0.0, 0.0, 0.0))

    def _build_task_args(self, tokens: list[str]) -> argparse.Namespace:
        """从用户输入解析任务参数"""
        defaults = _state_machine_defaults()
        defaults["map"] = self.host_args.map
        defaults["no_gui"] = self.host_args.no_gui
        defaults["camera"] = self.host_args.camera
        defaults["save_fpv"] = self.host_args.save_fpv
        defaults["keyboard_heartbeat_file"] = self.host_args.keyboard_heartbeat_file
        if self.host_args.policy_config:
            defaults["policy_config"] = self.host_args.policy_config
        if self.host_args.policy_checkpoint:
            defaults["policy_checkpoint"] = self.host_args.policy_checkpoint

        if not tokens:
            raise ValueError("Empty command")

        mode = tokens[0].lower()
        mode_map = {
            "navigate": "navigate",
            "nav": "navigate",
            "explore": "explore",
            "exp": "explore",
            "stand": "stand",
            "keyboard": "keyboard",
            "key": "keyboard",
            "estop": "estop",
            "stop": "estop",
        }
        if mode not in mode_map:
            raise ValueError(f"Unknown command: {mode}. Type 'help' for available commands.")
        defaults["mode"] = mode_map[mode]

        # 解析可选参数
        i = 1
        while i < len(tokens):
            if tokens[i] == "--max-steps" and i + 1 < len(tokens):
                defaults["max_steps"] = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "--stand-steps" and i + 1 < len(tokens):
                defaults["stand_steps"] = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "--scheduler" and i + 1 < len(tokens):
                defaults["scheduler"] = tokens[i + 1]
                i += 2
            elif tokens[i] == "--ddim-steps" and i + 1 < len(tokens):
                defaults["ddim_steps"] = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "--cfg-weight" and i + 1 < len(tokens):
                defaults["cfg_weight"] = float(tokens[i + 1])
                i += 2
            elif tokens[i] == "--tts":
                defaults["tts"] = True
                i += 1
            elif tokens[i] == "--tts-budget" and i + 1 < len(tokens):
                defaults["tts_budget"] = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "--tts-topk" and i + 1 < len(tokens):
                defaults["tts_topk"] = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "--tts-verifier" and i + 1 < len(tokens):
                defaults["tts_verifier"] = tokens[i + 1]
                i += 2
            elif tokens[i] == "--image-resize-mode" and i + 1 < len(tokens):
                defaults["image_resize_mode"] = tokens[i + 1]
                i += 2
            elif tokens[i] == "--close-threshold" and i + 1 < len(tokens):
                defaults["close_threshold"] = float(tokens[i + 1])
                i += 2
            elif tokens[i] == "--goal-source" and i + 1 < len(tokens):
                defaults["goal_source"] = tokens[i + 1]
                i += 2
            elif tokens[i] == "--num-goals" and i + 1 < len(tokens):
                defaults["num_goals"] = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "--topomap-dir" and i + 1 < len(tokens):
                defaults["topomap_dir"] = tokens[i + 1]
                i += 2
            elif tokens[i] == "--keyboard-vx-step" and i + 1 < len(tokens):
                defaults["keyboard_vx_step"] = float(tokens[i + 1])
                i += 2
            elif tokens[i] == "--keyboard-vy-step" and i + 1 < len(tokens):
                defaults["keyboard_vy_step"] = float(tokens[i + 1])
                i += 2
            elif tokens[i] == "--keyboard-wz-step" and i + 1 < len(tokens):
                defaults["keyboard_wz_step"] = float(tokens[i + 1])
                i += 2
            elif tokens[i] == "--keyboard-max-vx" and i + 1 < len(tokens):
                defaults["keyboard_max_vx"] = float(tokens[i + 1])
                i += 2
            elif tokens[i] == "--keyboard-max-vy" and i + 1 < len(tokens):
                defaults["keyboard_max_vy"] = float(tokens[i + 1])
                i += 2
            elif tokens[i] == "--keyboard-max-wz" and i + 1 < len(tokens):
                defaults["keyboard_max_wz"] = float(tokens[i + 1])
                i += 2
            elif tokens[i] == "--keyboard-dt" and i + 1 < len(tokens):
                defaults["keyboard_dt"] = float(tokens[i + 1])
                i += 2
            elif tokens[i] == "--keyboard-heartbeat-file" and i + 1 < len(tokens):
                defaults["keyboard_heartbeat_file"] = tokens[i + 1]
                i += 2
            elif tokens[i] == "--save-fpv":
                defaults["save_fpv"] = True
                i += 1
            else:
                i += 1

        return argparse.Namespace(**defaults)

    def _run_task(self, task_args: argparse.Namespace) -> HostTaskResult:
        from lite3_system.system import Lite3System

        self.task_count += 1
        self.session.clear_estop()
        task_args.capture_enabled = True
        validate_task_topomap_args(task_args, expected_domain=self.host_args.backend)
        platform = self._ensure_platform()

        system = Lite3System(
            task_args,
            platform=platform,
            result_prefix=f"{self.host_args.platform}_{self.host_args.backend}_interactive",
            close_platform_on_finalize=False,
            session=self.session,
        )
        exit_code = system.run()
        result = HostTaskResult(
            index=self.task_count,
            mode=task_args.mode,
            map_name=task_args.map,
            backend=self.host_args.backend,
            exit_code=exit_code,
            status="success" if exit_code == 0 else "failed",
        )
        self.results.append(result)
        return result

    def _do_capture(self):
        platform = self._ensure_platform()
        camera_image = platform.render_camera()
        position, yaw = platform.get_pose()
        capture = self.session.add_capture(
            image=camera_image,
            position=position,
            yaw=yaw,
            map_name=self.host_args.map,
            label_prefix=f"{self.host_args.map}_interactive",
        )
        print(f"  [Capture] #{capture.index}: {capture.label} @ pos=({position[0]:.2f}, {position[1]:.2f}) -> {capture.saved_path}")

    def _do_status(self):
        platform = self._ensure_platform()
        position, yaw = platform.get_pose()
        height = platform.get_height()
        captures = len(self.session.capture_queue)
        print(f"  Position: ({position[0]:.3f}, {position[1]:.3f}), yaw={yaw:.3f}, height={height:.3f}")
        print(f"  Tasks completed: {self.task_count}, Captures: {captures}")
        print(f"  Backend: {self.host_args.backend}, Map: {self.host_args.map}")
        if self.results:
            last = self.results[-1]
            print(f"  Last task: mode={last.mode}, status={last.status}")

    def _has_stdin_input(self) -> bool:
        """非阻塞检查 stdin 是否有输入"""
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            return bool(ready)
        except (ValueError, OSError):
            return False

    def _should_auto_exit(self) -> tuple[bool, str]:
        """真机模式下，检测需要自动退出 idle 循环的安全条件。

        返回 (should_exit, reason)。MuJoCo 后端永远返回 (False, "")
        （退出靠 viewer 关闭或用户 quit）。
        """
        if self.host_args.backend != "real":
            return False, ""
        platform = self._platform
        bridge = getattr(platform, "bridge", None) if platform is not None else None
        if bridge is None:
            return False, ""
        try:
            if hasattr(bridge, "is_fallen") and bridge.is_fallen():
                return True, "bridge.is_fallen() == True"
        except Exception as exc:  # noqa: BLE001
            print(f"[Host] is_fallen check raised {type(exc).__name__}: {exc}")
        ctrl = getattr(bridge, "ctrl", None)
        if ctrl is not None and hasattr(ctrl, "is_connection_stale"):
            try:
                if ctrl.is_connection_stale(timeout=2.0):
                    return True, "Lite3 motion host state stale > 2s"
            except Exception as exc:  # noqa: BLE001
                print(f"[Host] is_connection_stale check raised {type(exc).__name__}: {exc}")
        return False, ""

    def run(self) -> int:
        print(INTERACTIVE_HELP)
        self._ensure_platform()
        self._ensure_standing()
        if self.host_args.backend == "real" and self._platform is not None:
            self._platform.release_manual_control()

        # idle 循环：执行 stand 步进，同时轮询 stdin
        print("\n[Host] IDLE — 等待命令 (输入 help 查看帮助):")
        try:
            while True:
                # 真机安全自动退出：摔倒 / 运动主机链路陈旧
                should_exit, reason = self._should_auto_exit()
                if should_exit:
                    print(f"[Host] ⚠️ 自动退出: {reason}")
                    if self._platform is not None:
                        try:
                            self._platform.emergency_stop()
                        except Exception as exc:  # noqa: BLE001
                            print(f"[Host] emergency_stop during auto-exit raised: {exc}")
                    break

                # idle 步进（保持站立，渲染相机）
                self._idle_step()

                # 非阻塞读取命令
                if not self._has_stdin_input():
                    time.sleep(0.05)
                    continue

                raw_line = sys.stdin.readline()
                if not raw_line:
                    break
                line = raw_line.strip()
                if not line:
                    print("[Host] IDLE — 等待命令:")
                    continue

                tokens = line.split()
                cmd = tokens[0].lower()

                if cmd in ("quit", "exit", "q"):
                    print("[Host] Shutting down...")
                    break
                elif cmd == "help":
                    print(INTERACTIVE_HELP)
                elif cmd == "capture":
                    self._do_capture()
                elif cmd == "status":
                    self._do_status()
                elif cmd in ("navigate", "nav", "explore", "exp", "stand", "keyboard", "key", "estop", "stop"):
                    try:
                        task_args = self._build_task_args(tokens)
                        print(f"[Host] Dispatching: mode={task_args.mode}, map={task_args.map}, "
                              f"max_steps={task_args.max_steps}")
                        result = self._run_task(task_args)
                        print(f"[Host] Task #{result.index} finished: {result.status} (exit={result.exit_code})")
                        if task_args.mode == "estop":
                            print("[Host] EStop completed; leaving interactive loop.")
                            break
                    except Exception as e:
                        print(f"[Host] Task error: {e}")
                else:
                    print(f"[Host] Unknown command: {cmd}. Type 'help' for available commands.")

                if not self._platform.viewer_alive():
                    print("[Host] Viewer closed.")
                    break
                print("[Host] IDLE — 等待命令:")

        except KeyboardInterrupt:
            print("\n[Host] Interrupted.")

        # 清理
        if self._platform is not None:
            self._platform.close()

        # 保存结果摘要
        if self.results:
            output_dir = self.session.run_dir
            summary_lines = [
                f"Interactive session: {len(self.results)} tasks completed",
                f"Backend: {self.host_args.backend}, Map: {self.host_args.map}",
                "",
            ]
            for r in self.results:
                summary_lines.append(f"  Task {r.index}: mode={r.mode} status={r.status}")
            (output_dir / "interactive_summary.txt").write_text(
                "\n".join(summary_lines), encoding="utf-8"
            )
            print(f"[Host] Session saved to: {output_dir}")

        return 0


def main() -> int:
    host_parser = build_host_parser()
    host_args, remaining_args = host_parser.parse_known_args()
    finalize_host_args(host_args)

    if host_args.interactive:
        host = InteractiveNavigationHost(host_args)
        return host.run()

    task_args_list = load_task_args(host_args, remaining_args)
    host = Lite3NavigationHost(host_args, task_args_list)
    return host.run()


if __name__ == "__main__":
    raise SystemExit(main())
