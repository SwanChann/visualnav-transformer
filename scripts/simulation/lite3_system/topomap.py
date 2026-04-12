from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lite3_system.legacy_bridge import load_legacy


@dataclass
class MissionTarget:
    label: str
    topomap: list
    goal_view: object | None
    goal_position: np.ndarray | None = None


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
        return len(self.current.topomap) - 1

    def reset_localization(self) -> None:
        self.closest_node = 0

    def advance(self) -> bool:
        if self.index + 1 >= len(self.missions):
            return False
        self.index += 1
        self.reset_localization()
        return True


def build_mission_queue(args, platform) -> MissionQueue | None:
    legacy = load_legacy()
    missions: list[MissionTarget] = []

    if getattr(args, "mode", "") == "explore":
        return None

    if getattr(args, "random", False):
        if not hasattr(platform, "env"):
            raise ValueError("Random spawn/goal topomap generation requires a simulation backend with env support.")
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

    for topomap_dir in getattr(args, "mission_topomap", []) or []:
        topo_path = Path(topomap_dir)
        topomap = legacy.load_topomap_from_dir(str(topo_path))
        missions.append(MissionTarget(label=topo_path.name, topomap=topomap, goal_view=topomap[-1]))

    for traj_name in getattr(args, "mission_dataset_traj", []) or []:
        topomap = legacy.load_topomap_from_dataset(traj_name, step=args.topomap_step)
        missions.append(MissionTarget(label=traj_name, topomap=topomap, goal_view=topomap[-1]))

    if not missions and getattr(args, "mode", "") == "navigate":
        if args.topomap_dir:
            topomap = legacy.load_topomap_from_dir(args.topomap_dir)
            missions.append(MissionTarget(label=Path(args.topomap_dir).name, topomap=topomap, goal_view=topomap[-1]))
        else:
            topomap = legacy.load_topomap_from_dataset(args.topomap_traj, step=args.topomap_step)
            missions.append(MissionTarget(label=args.topomap_traj, topomap=topomap, goal_view=topomap[-1]))

    if not missions:
        return None
    return MissionQueue(missions)
