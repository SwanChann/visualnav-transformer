#!/usr/bin/env python3
"""Unified NoMaD navigation host for MuJoCo and real-robot backends."""

from __future__ import annotations

import argparse
import json
import sys
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


RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class HostTaskResult:
    index: int
    mode: str
    map_name: str
    backend: str
    exit_code: int
    status: str


class Lite3NavigationHost:
    def __init__(self, host_args, task_args_list: list[argparse.Namespace]) -> None:
        self.host_args = host_args
        self.task_args_list = task_args_list
        self.results: list[HostTaskResult] = []

    def _load_bridge_kwargs(self) -> dict:
        if not self.host_args.bridge_config:
            return {}
        config_path = Path(self.host_args.bridge_config).expanduser().resolve()
        return json.loads(config_path.read_text(encoding="utf-8"))

    def _validate_task(self, task_args: argparse.Namespace) -> None:
        if task_args.mode == "mission":
            has_mission = bool(task_args.mission_topomap or task_args.mission_dataset_traj or task_args.random)
            if not has_mission:
                raise ValueError("Mission mode requires mission_topomap, mission_dataset_traj, or --random.")

        if self.host_args.backend == "real":
            if task_args.mode == "generate-topomap":
                raise ValueError("Real backend does not support generate-topomap; collect real topomap offline first.")
            if task_args.random:
                raise ValueError("Real backend does not support random spawn/goal generation.")
            if not self.host_args.bridge_module or not self.host_args.bridge_class:
                raise ValueError("Real backend requires --bridge-module and --bridge-class.")

    def _build_platform(self):
        if self.host_args.backend == "mujoco":
            return None
        from lite3_system.interfaces import ExternalBridgePlatform

        bridge_kwargs = self._load_bridge_kwargs()
        return ExternalBridgePlatform(
            bridge_module=self.host_args.bridge_module,
            bridge_class=self.host_args.bridge_class,
            bridge_kwargs=bridge_kwargs,
        )

    def _result_prefix(self) -> str:
        return f"{self.host_args.platform}_{self.host_args.backend}_navigation_host"

    def run_task(self, task_index: int, task_args: argparse.Namespace) -> HostTaskResult:
        from lite3_system.system import Lite3System

        self._validate_task(task_args)
        platform = self._build_platform()
        system = Lite3System(
            task_args,
            platform=platform,
            result_prefix=self._result_prefix(),
            close_platform_on_finalize=True,
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
                "2. The Lite3 state machine remains responsible for closed-loop execution and recovery.",
                "3. `navigate` is the single-topomap local navigation task, while `mission` is the multi-goal queue task.",
                "4. The MuJoCo backend can run directly; the real backend requires an external Python bridge module.",
            ]
        )
        return "\n".join(lines)


def build_host_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified NoMaD navigation host")
    parser.add_argument("--platform", choices=["lite3"], default="lite3")
    parser.add_argument("--backend", choices=["mujoco", "real"], default="mujoco")
    parser.add_argument("--plan-file", type=str, default=None, help="JSON plan file containing defaults and tasks")
    parser.add_argument("--bridge-module", type=str, default=None, help="Python module for the real-robot bridge")
    parser.add_argument("--bridge-class", type=str, default=None, help="Bridge class name in the selected module")
    parser.add_argument("--bridge-config", type=str, default=None, help="Optional JSON file with bridge kwargs")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue remaining tasks after one task fails")
    parser.add_argument("--save-plan", action="store_true", help="Save the host plan and run summary")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the task plan without starting NoMaD")
    return parser


def _state_machine_defaults() -> dict:
    parser = build_state_machine_parser()
    return vars(parser.parse_args([]))


def load_task_args(host_args, remaining_args: list[str]) -> list[argparse.Namespace]:
    state_parser = build_state_machine_parser()
    state_defaults = _state_machine_defaults()

    if host_args.plan_file:
        plan_path = Path(host_args.plan_file).expanduser().resolve()
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
        defaults = dict(state_defaults)
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
    return [task_args]


def main() -> int:
    host_parser = build_host_parser()
    host_args, remaining_args = host_parser.parse_known_args()
    task_args_list = load_task_args(host_args, remaining_args)
    host = Lite3NavigationHost(host_args, task_args_list)
    return host.run()


if __name__ == "__main__":
    raise SystemExit(main())
