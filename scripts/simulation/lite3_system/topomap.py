from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lite3_system.legacy_bridge import load_legacy
from project_paths import repo_path

if TYPE_CHECKING:
    from lite3_system.session import NavigationSession

TOPOMAP_META_FILENAME = "topomap_meta.json"


@dataclass
class MissionTarget:
    label: str
    topomap: list
    goal_view: object | None
    goal_position: np.ndarray | None = None
    goal_node_index: int | None = None
    topomap_tensor: object | None = None


class MissionQueue:
    def __init__(self, missions: list[MissionTarget]) -> None:
        if not missions:
            raise ValueError("Mission queue requires at least one target.")
        self.missions = missions
        self.index = 0
        self.closest_node = 0

    @property
    def current(self) -> MissionTarget:
        return self.missions[self.index]

    @property
    def goal_node(self) -> int:
        node = self.current.goal_node_index
        if node is None:
            node = len(self.current.topomap) - 1
        return int(np.clip(node, 0, len(self.current.topomap) - 1))

    def reset_localization(self) -> None:
        self.closest_node = 0

    def advance(self) -> bool:
        if self.index + 1 >= len(self.missions):
            return False
        self.index += 1
        self.reset_localization()
        return True

    def set_goal_node(self, goal_node: int) -> int:
        clamped = int(np.clip(goal_node, 0, len(self.current.topomap) - 1))
        self.current.goal_node_index = clamped
        self.current.goal_view = self.current.topomap[clamped]
        if self.closest_node > clamped:
            self.closest_node = clamped
        return clamped

    def shift_goal_node(self, delta: int) -> int:
        return self.set_goal_node(self.goal_node + int(delta))


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _repo_root(legacy) -> Path:
    return Path(legacy.PROJECT_ROOT).resolve()


def get_mujoco_topomap_root(legacy=None) -> Path:
    if legacy is None:
        return repo_path("topomaps").resolve()
    return _repo_root(legacy) / "topomaps"


def get_real_topomap_root(legacy=None) -> Path:
    if legacy is None:
        return repo_path("deployment", "topomaps", "images").resolve()
    return _repo_root(legacy) / "deployment" / "topomaps" / "images"


def get_default_mujoco_topomap_dir(map_name: str, legacy=None) -> Path:
    legacy = legacy or load_legacy()
    return Path(legacy.get_default_topomap_dir(map_name)).expanduser().resolve()


def load_topomap_metadata(topomap_dir: Path) -> dict | None:
    meta_path = topomap_dir / TOPOMAP_META_FILENAME
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def infer_topomap_domain(topomap_dir: Path, legacy=None) -> str | None:
    meta = load_topomap_metadata(topomap_dir)
    if meta is not None:
        domain = meta.get("domain")
        if domain in {"mujoco", "real"}:
            return domain
    resolved = topomap_dir.expanduser().resolve()
    if _is_relative_to(resolved, get_mujoco_topomap_root(legacy)):
        return "mujoco"
    if _is_relative_to(resolved, get_real_topomap_root(legacy)):
        return "real"
    return None


def validate_topomap_dir_for_domain(topomap_dir: str | Path, expected_domain: str, legacy=None) -> Path:
    resolved = Path(topomap_dir).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Topomap directory not found: {resolved}")
    actual_domain = infer_topomap_domain(resolved, legacy)
    if actual_domain is None:
        raise ValueError(
            "Unable to determine the topomap domain for "
            f"'{resolved}'. Place it under '{get_mujoco_topomap_root(legacy)}' or "
            f"'{get_real_topomap_root(legacy)}', or add '{TOPOMAP_META_FILENAME}' "
            f"with domain='{expected_domain}'."
        )
    if actual_domain != expected_domain:
        raise ValueError(
            f"Topomap directory '{resolved}' is tagged as '{actual_domain}', "
            f"but the current backend requires '{expected_domain}'."
        )
    return resolved


def resolve_navigation_topomap_dir(args, expected_domain: str, legacy=None) -> Path:
    legacy = legacy or load_legacy()
    if getattr(args, "topomap_dir", None):
        return validate_topomap_dir_for_domain(args.topomap_dir, expected_domain, legacy)
    if expected_domain == "mujoco":
        default_dir = get_default_mujoco_topomap_dir(args.map, legacy)
        if not default_dir.is_dir():
            raise FileNotFoundError(
                f"MuJoCo topomap directory not found: {default_dir}. "
                f"Run `python scripts/nomad_mujoco_lite3_nav.py --mode generate-topomap --map {args.map}` first."
            )
        return validate_topomap_dir_for_domain(default_dir, expected_domain, legacy)
    raise ValueError("Real backend requires an explicit real-world --topomap-dir; dataset fallback is disabled.")


def validate_task_topomap_args(args, expected_domain: str, legacy=None) -> None:
    if getattr(args, "topomap_traj", None):
        raise ValueError(
            "Dataset trajectory topomaps are offline-analysis-only. "
            "Closed-loop MuJoCo and real deployment must use domain-matched topomap directories."
        )
    mission_dataset = getattr(args, "mission_dataset_traj", None) or []
    if mission_dataset:
        raise ValueError(
            "mission_dataset_traj is disabled for the state-machine and deployment pipeline. "
            "Use MuJoCo-generated topomap directories in simulation and real-world captured topomap directories on the robot."
        )
    if getattr(args, "random", False) and expected_domain != "mujoco":
        raise ValueError("Random spawn/goal generation is MuJoCo-only.")
    if getattr(args, "goal_source", "topomap") == "random_points" and expected_domain != "mujoco":
        raise ValueError("goal-source=random_points is currently only supported on the MuJoCo backend.")

    if getattr(args, "mode", "") in {"stand", "explore", "keyboard", "walk-test", "estop", "generate-topomap"}:
        return

    if getattr(args, "goal_source", "topomap") == "capture_queue":
        return

    if getattr(args, "goal_source", "topomap") == "random_points":
        if int(getattr(args, "num_goals", 1)) < 1:
            raise ValueError("num-goals must be >= 1 for goal-source=random_points.")
        return

    mission_topomaps = getattr(args, "mission_topomap", None) or []
    if getattr(args, "mode", "") == "navigate" and mission_topomaps:
        for topomap_dir in mission_topomaps:
            validate_topomap_dir_for_domain(topomap_dir, expected_domain, legacy)
    elif getattr(args, "mode", "") == "navigate":
        if getattr(args, "topomap_dir", None):
            validate_topomap_dir_for_domain(args.topomap_dir, expected_domain, legacy)
        else:
            legacy = legacy or load_legacy()
            resolve_navigation_topomap_dir(args, expected_domain, legacy)
    elif getattr(args, "mode", "") == "mission":
        if not mission_topomaps and expected_domain == "real":
            raise ValueError("Real backend mission mode requires one or more --mission-topomap directories.")
        for topomap_dir in mission_topomaps:
            validate_topomap_dir_for_domain(topomap_dir, expected_domain, legacy)


def _is_valid_goal(scene_config, position: np.ndarray, margin: float = 0.7) -> bool:
    x_min = scene_config["corridor_x_min"] + margin
    x_max = scene_config["corridor_x_max"] - margin
    half_width = scene_config["corridor_half_width"] - margin * 0.5
    if not (x_min <= position[0] <= x_max and -half_width <= position[1] <= half_width):
        return False
    for obstacle in scene_config.get("obstacles", []):
        ox, oy = obstacle["pos"][0], obstacle["pos"][1]
        sx, sy = obstacle["size"][0], obstacle["size"][1]
        if abs(position[0] - ox) < sx + margin and abs(position[1] - oy) < sy + margin:
            return False
    return True


def _path_cumulative_lengths(path_positions: np.ndarray) -> np.ndarray:
    path_positions = np.asarray(path_positions, dtype=float)
    if len(path_positions) <= 1:
        return np.zeros(len(path_positions), dtype=float)
    diffs = np.diff(path_positions, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _slice_path_between_indices(path_positions: np.ndarray, start_index: int, goal_index: int) -> np.ndarray:
    start_index = int(max(0, start_index))
    goal_index = int(max(0, goal_index))
    if goal_index < start_index:
        start_index, goal_index = goal_index, start_index
    if goal_index == start_index:
        goal_index = min(start_index + 1, len(path_positions) - 1)
    segment = np.asarray(path_positions[start_index: goal_index + 1], dtype=float)
    if len(segment) == 1:
        segment = np.repeat(segment, 2, axis=0)
    return segment


def _goal_yaw_from_path_segment(segment: np.ndarray) -> float:
    if len(segment) < 2:
        return 0.0
    dx, dy = np.asarray(segment[-1], dtype=float) - np.asarray(segment[-2], dtype=float)
    if abs(dx) + abs(dy) <= 1e-8:
        dx, dy = np.asarray(segment[1], dtype=float) - np.asarray(segment[0], dtype=float)
    return float(np.arctan2(dy, dx)) if abs(dx) + abs(dy) > 1e-8 else 0.0


def _snapshot_platform_state(platform):
    env = getattr(platform, "env", None)
    if env is None:
        return None
    return {
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "command": env.command.copy() if hasattr(env, "command") else None,
        "last_action": env.last_action.copy() if hasattr(env, "last_action") else None,
        "sim_step": getattr(env, "sim_step", None),
        "policy_step": getattr(env, "policy_step", None),
    }


def _restore_platform_state(platform, snapshot) -> None:
    if snapshot is None:
        return
    env = getattr(platform, "env", None)
    if env is None:
        return
    env.data.qpos[:] = snapshot["qpos"]
    env.data.qvel[:] = snapshot["qvel"]
    if snapshot.get("command") is not None and hasattr(env, "command"):
        env.command[:] = snapshot["command"]
    if snapshot.get("last_action") is not None and hasattr(env, "last_action"):
        env.last_action[:] = snapshot["last_action"]
    if snapshot.get("sim_step") is not None:
        env.sim_step = int(snapshot["sim_step"])
    if snapshot.get("policy_step") is not None:
        env.policy_step = int(snapshot["policy_step"])
    env.reset_wall_clock()
    legacy = load_legacy()
    legacy.mujoco.mj_forward(env.model, env.data)
    if getattr(env, "viewer", None):
        env.viewer.sync()


def _sample_random_goal_positions(args, platform, legacy) -> list[np.ndarray]:
    current_position, _ = platform.get_pose()
    scene_config = legacy.SCENE_CONFIG
    num_goals = max(int(getattr(args, "num_goals", 1)), 1)
    min_dist = float(getattr(args, "random_goal_min_dist", 2.0))
    min_separation = float(getattr(args, "random_goal_min_separation", 1.0))
    goal_seed = int(getattr(args, "goal_seed", 0))
    rng = np.random.default_rng(goal_seed)
    path_positions = np.asarray(legacy.build_scene_reference_path(scene_config, num_points=600), dtype=float)
    cum_len = _path_cumulative_lengths(path_positions)
    current_index = int(legacy.project_position_to_path_index(current_position, path_positions))
    current_s = float(cum_len[current_index])

    candidate_indices: list[int] = []
    for path_index in range(current_index + 3, len(path_positions) - 3):
        candidate = path_positions[path_index]
        if not _is_valid_goal(scene_config, candidate):
            continue
        if float(cum_len[path_index] - current_s) < min_dist:
            continue
        candidate_indices.append(path_index)

    if not candidate_indices:
        fallback = np.asarray(scene_config["goal_pos"], dtype=float)
        return [fallback]

    rng.shuffle(candidate_indices)
    selected_indices: list[int] = []
    for candidate_index in candidate_indices:
        if any(abs(float(cum_len[candidate_index] - cum_len[other_index])) < min_separation for other_index in selected_indices):
            continue
        selected_indices.append(candidate_index)
        if len(selected_indices) >= num_goals:
            break

    if not selected_indices:
        selected_indices.append(candidate_indices[-1])

    selected_indices.sort()
    return [np.asarray(path_positions[index], dtype=float) for index in selected_indices]


def _build_random_goal_missions(args, platform, legacy) -> list[MissionTarget]:
    if not platform.supports_online_topomap_generation():
        raise ValueError("Random goal point missions require a MuJoCo simulation backend.")
    current_position, current_yaw = platform.get_pose()
    start_position = np.asarray(current_position, dtype=float)
    path_positions = np.asarray(legacy.build_scene_reference_path(legacy.SCENE_CONFIG, num_points=600), dtype=float)
    goal_positions = _sample_random_goal_positions(args, platform, legacy)
    platform.set_goal_markers(goal_positions, active_index=0)
    platform_snapshot = _snapshot_platform_state(platform)

    missions: list[MissionTarget] = []
    cursor_index = int(legacy.project_position_to_path_index(start_position, path_positions))
    for goal_index, goal_position in enumerate(goal_positions, start=1):
        goal_path_index = int(legacy.project_position_to_path_index(goal_position, path_positions))
        if goal_path_index <= cursor_index:
            goal_path_index = min(len(path_positions) - 1, cursor_index + max(int(args.topomap_nodes), 3))
        segment = _slice_path_between_indices(path_positions, cursor_index, goal_path_index)
        topomap = legacy.generate_topomap_online_from_positions(platform.env, segment, num_nodes=args.topomap_nodes)
        goal_view = legacy.capture_goal_view(platform.env, goal_position, yaw=_goal_yaw_from_path_segment(segment))
        missions.append(
            MissionTarget(
                label=f"goal-{goal_index:02d}",
                topomap=topomap,
                goal_view=goal_view,
                goal_position=np.asarray(goal_position, dtype=float),
            )
        )
        cursor_index = goal_path_index
    _restore_platform_state(platform, platform_snapshot)
    return missions


def _build_capture_queue_missions(args, platform, legacy, session: "NavigationSession | None") -> list[MissionTarget]:
    if session is None or not session.capture_queue:
        raise ValueError("goal-source=capture_queue requires at least one captured image in the shared session queue.")
    if not platform.supports_online_topomap_generation():
        raise ValueError("Capture-queue navigation currently requires a backend that can generate a topomap to the captured pose.")

    requested_count = max(int(getattr(args, "num_goals", 1)), 1)
    if requested_count == 1:
        captures = [session.get_capture_by_index(int(getattr(args, "capture_index", -1)))]
    else:
        captures = session.get_recent_captures(requested_count)
        if len(captures) < requested_count:
            raise ValueError(
                f"Requested {requested_count} capture goals, but only {len(captures)} captures exist in the queue."
            )
    if any(capture is None for capture in captures):
        raise ValueError("Capture queue selection did not resolve to a valid goal.")

    goal_positions: list[np.ndarray] = []
    for capture in captures:
        if capture.position is None:
            raise ValueError(
                f"Capture '{capture.label}' does not have a stored pose, so it cannot be used as a navigation goal."
            )
        goal_positions.append(np.asarray(capture.position, dtype=float))
    platform.set_goal_markers(goal_positions, active_index=0)

    current_position, current_yaw = platform.get_pose()
    path_positions = np.asarray(legacy.build_scene_reference_path(legacy.SCENE_CONFIG, num_points=600), dtype=float)
    cursor_index = int(legacy.project_position_to_path_index(current_position, path_positions))
    platform_snapshot = _snapshot_platform_state(platform)
    missions: list[MissionTarget] = []
    for capture in captures:
        goal_position = np.asarray(capture.position, dtype=float)
        goal_path_index = int(legacy.project_position_to_path_index(goal_position, path_positions))
        if goal_path_index <= cursor_index:
            goal_path_index = min(len(path_positions) - 1, cursor_index + max(int(args.topomap_nodes), 3))
        segment = _slice_path_between_indices(path_positions, cursor_index, goal_path_index)
        topomap = legacy.generate_topomap_online_from_positions(platform.env, segment, num_nodes=args.topomap_nodes)
        missions.append(
            MissionTarget(
                label=capture.label,
                topomap=topomap,
                goal_view=capture.image,
                goal_position=goal_position,
            )
        )
        cursor_index = goal_path_index
    _restore_platform_state(platform, platform_snapshot)
    return missions


def build_mission_queue(args, platform, session: "NavigationSession | None" = None) -> MissionQueue | None:
    legacy = load_legacy()
    missions: list[MissionTarget] = []
    expected_domain = platform.environment_domain()

    if getattr(args, "mode", "") in {"explore", "stand", "keyboard", "walk-test", "estop"}:
        return None

    if getattr(args, "random", False):
        if not platform.supports_online_topomap_generation():
            raise ValueError("Random spawn/goal topomap generation requires a MuJoCo simulation backend.")
        spawn_position, goal_position = legacy.generate_random_spawn_goal(legacy.SCENE_CONFIG)
        platform.reset_position(float(spawn_position[0]), float(spawn_position[1]))
        platform.set_scene_goal(goal_position)
        goal_view = legacy.capture_goal_view(platform.env, goal_position)
        topomap = legacy.generate_topomap_online(
            platform.env,
            spawn_position,
            goal_position,
            num_nodes=args.topomap_nodes,
        )
        missions.append(
            MissionTarget(
                label=f"random-{args.map}",
                topomap=topomap,
                goal_view=goal_view,
                goal_position=np.asarray(goal_position, dtype=float),
            )
        )

    goal_source = getattr(args, "goal_source", "topomap")
    if goal_source == "random_points":
        missions.extend(_build_random_goal_missions(args, platform, legacy))
    elif goal_source == "capture_queue":
        missions.extend(_build_capture_queue_missions(args, platform, legacy, session))

    for topomap_dir in getattr(args, "mission_topomap", []) or []:
        topo_path = validate_topomap_dir_for_domain(topomap_dir, expected_domain, legacy)
        topomap = legacy.load_topomap_from_dir(str(topo_path))
        missions.append(MissionTarget(label=topo_path.name, topomap=topomap, goal_view=topomap[-1]))

    for traj_name in getattr(args, "mission_dataset_traj", []) or []:
        raise ValueError(
            "Dataset trajectories are not allowed in the closed-loop state-machine pipeline. "
            "Use domain-matched topomap directories instead."
        )

    if not missions and getattr(args, "mode", "") == "navigate":
        topo_path = resolve_navigation_topomap_dir(args, expected_domain, legacy)
        topomap = legacy.load_topomap_from_dir(str(topo_path))
        goal_position = None
        if expected_domain == "mujoco":
            scene_config = legacy.SCENE_MAPS.get(getattr(args, "map", ""), legacy.SCENE_CONFIG)
            goal_position = np.asarray(scene_config["goal_pos"], dtype=float)
            platform.set_scene_goal(goal_position)
        missions.append(
            MissionTarget(
                label=topo_path.name,
                topomap=topomap,
                goal_view=topomap[-1],
                goal_position=goal_position,
            )
        )

    if not missions:
        return None
    return MissionQueue(missions)
