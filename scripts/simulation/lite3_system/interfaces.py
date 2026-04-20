from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Deque

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
SIM_ROOT = CURRENT_DIR.parent
SCRIPTS_ROOT = SIM_ROOT.parent
for candidate in (SCRIPTS_ROOT, SCRIPTS_ROOT / "shared"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from lite3_system.legacy_bridge import load_legacy
from shared.nomad_inference import NoMaDInferenceModule, NoMaDInferenceSpec


@dataclass
class MotionCommand:
    linear_x: float
    linear_y: float
    yaw_rate: float


@dataclass
class NavigationResult:
    chosen_waypoint: np.ndarray
    sampled_actions: np.ndarray
    closest_node: int
    selected_node: int
    predicted_distance: float


@dataclass
class ExplorationResult:
    chosen_waypoint: np.ndarray
    sampled_actions: np.ndarray


class NavigationPlatformBase:
    """Abstract platform interface for simulation and real deployment."""

    def environment_domain(self) -> str:
        return "unknown"

    def supports_online_topomap_generation(self) -> bool:
        return False

    def set_scene_goal(self, goal_position: np.ndarray) -> None:
        return None

    def set_goal_markers(self, goal_positions: list[np.ndarray], active_index: int = 0) -> None:
        return None

    def highlight_goal_marker(self, active_index: int) -> None:
        return None

    def render_camera(self):
        raise NotImplementedError

    def standup(self, duration: float) -> None:
        raise NotImplementedError

    def apply_command(self, command: MotionCommand) -> None:
        raise NotImplementedError

    def get_pose(self):
        raise NotImplementedError

    def get_height(self) -> float:
        raise NotImplementedError

    def get_forward_speed(self) -> float:
        raise NotImplementedError

    def is_fallen(self) -> bool:
        raise NotImplementedError

    def viewer_alive(self) -> bool:
        raise NotImplementedError

    def reset_position(self, x: float, y: float, yaw: float = 0.0) -> None:
        return None

    def prepare_task(self, mode: str) -> None:
        return None

    def emergency_stop(self) -> None:
        # 中文注释：统一安全接口，默认发送零速度指令
        self.apply_command(MotionCommand(0.0, 0.0, 0.0))

    def close(self) -> None:
        return None


class Lite3HighLevelNoMaD:
    def __init__(
        self,
        scheduler_kind: str,
        ddim_steps: int,
        waypoint_index: int,
        radius: int,
        close_threshold: float,
        cfg_weight: float = 0.0,
        tts_enabled: bool = False,
        tts_budget: int = 8,
        tts_topk: int = 1,
        tts_verifier: str = "heuristic",
        policy_config: str | None = None,
        policy_checkpoint: str | None = None,
        device: str | None = None,
        image_resize_mode: str = "stretch",
    ) -> None:
        self.legacy = load_legacy()
        self.inference = NoMaDInferenceModule(
            NoMaDInferenceSpec(
                policy_config=policy_config,
                policy_checkpoint=policy_checkpoint,
                scheduler_kind=scheduler_kind,
                ddim_steps=ddim_steps,
                cfg_weight=cfg_weight,
                tts_enabled=tts_enabled,
                tts_budget=tts_budget,
                tts_topk=tts_topk,
                tts_verifier=tts_verifier,
                device=device,
                image_resize_mode=image_resize_mode,
            )
        )
        self.device = self.inference.device
        self.context_size = self.inference.context_size
        self.waypoint_index = waypoint_index
        self.radius = radius
        self.close_threshold = close_threshold

    def _build_obs_tensor(self, frame_buffer: Deque[torch.Tensor]) -> torch.Tensor:
        return self.inference.build_obs_tensor(frame_buffer).to(self.device)

    def _sample_actions(self, obs_cond: torch.Tensor, uncond: torch.Tensor | None = None) -> np.ndarray:
        return self.inference.sample_actions(obs_cond, uncond_condition=uncond)

    def predict_exploration(self, frame_buffer: Deque[torch.Tensor]) -> ExplorationResult:
        obs_tensor = self._build_obs_tensor(frame_buffer)
        image_height = int(self.inference.image_size[1])
        image_width = int(self.inference.image_size[0])
        fake_goal = torch.randn((1, 3, image_height, image_width), device=self.device)
        obs_cond = self.inference.encode_condition(obs_tensor, fake_goal, goal_mask_value=1)

        sampled_actions = self._sample_actions(obs_cond)
        mean_action = sampled_actions.mean(axis=0)
        chosen_waypoint = mean_action[min(self.waypoint_index, self.inference.len_traj_pred - 1)]
        return ExplorationResult(chosen_waypoint=chosen_waypoint, sampled_actions=sampled_actions)

    def predict_navigation(
        self,
        frame_buffer: Deque[torch.Tensor],
        topomap: list,
        closest_node: int,
        goal_node: int,
    ) -> NavigationResult:
        obs_tensor = self._build_obs_tensor(frame_buffer)
        start = max(closest_node - self.radius, 0)
        end = min(closest_node + self.radius + 1, goal_node)

        goal_tensors = []
        for goal_image in topomap[start : end + 1]:
            goal_tensors.append(self.inference.pil_to_tensor(goal_image).unsqueeze(0).to(self.device))
        goal_batch = torch.cat(goal_tensors, dim=0)
        candidate_count = goal_batch.shape[0]

        obsgoal_cond = self.inference.encode_condition(
            obs_tensor.repeat(candidate_count, 1, 1, 1),
            goal_batch,
            goal_mask_value=0,
        )
        distances = self.inference.predict_distance(obsgoal_cond)

        min_index = int(np.argmin(distances))
        raw_closest = min_index + start
        clipped_closest = int(np.clip(raw_closest, closest_node - 1, closest_node + 3))
        clipped_closest = min(clipped_closest, goal_node)
        local_index = clipped_closest - start
        selected_index = min(local_index + int(distances[min_index] < self.close_threshold), len(obsgoal_cond) - 1)
        obs_cond = obsgoal_cond[selected_index].unsqueeze(0)

        # CFG: 计算无条件编码
        uncond_cond = None
        if self.inference.cfg_weight != 0.0:
            uncond_cond = self.inference.encode_condition(
                obs_tensor,
                goal_batch[selected_index : selected_index + 1],
                goal_mask_value=1,
            )

        sampled_actions = self._sample_actions(obs_cond, uncond=uncond_cond)
        mean_action = sampled_actions.mean(axis=0)
        chosen_waypoint = mean_action[min(self.waypoint_index, self.inference.len_traj_pred - 1)]
        return NavigationResult(
            chosen_waypoint=chosen_waypoint,
            sampled_actions=sampled_actions,
            closest_node=clipped_closest,
            selected_node=start + selected_index,
            predicted_distance=float(distances[min_index]),
        )


class Lite3MiddleLayerPD:
    def __init__(self) -> None:
        self.legacy = load_legacy()

    def waypoint_to_command(self, waypoint: np.ndarray) -> MotionCommand:
        linear_x, yaw_rate = self.legacy.pd_controller(waypoint)
        return MotionCommand(linear_x=float(linear_x), linear_y=0.0, yaw_rate=float(yaw_rate))


class Lite3LowLevelPlatform(NavigationPlatformBase):
    def __init__(self, gui: bool, scene_name: str) -> None:
        self.legacy = load_legacy()
        self.legacy.SCENE_CONFIG = self.legacy.SCENE_MAPS[scene_name]
        self.env = self.legacy.MuJoCoLite3Env(gui=gui)

    def environment_domain(self) -> str:
        return "mujoco"

    def supports_online_topomap_generation(self) -> bool:
        return True

    def set_scene_goal(self, goal_position: np.ndarray) -> None:
        self.legacy.SCENE_CONFIG["goal_pos"] = [float(goal_position[0]), float(goal_position[1])]
        self.env.move_goal_marker(float(goal_position[0]), float(goal_position[1]))

    def set_goal_markers(self, goal_positions: list[np.ndarray], active_index: int = 0) -> None:
        marker_positions = [(float(goal[0]), float(goal[1])) for goal in goal_positions]
        self.env.set_goal_markers(marker_positions, active_index=active_index)

    def highlight_goal_marker(self, active_index: int) -> None:
        self.env.highlight_goal_marker(active_index)

    def render_camera(self):
        return self.env.render_camera()

    def standup(self, duration: float) -> None:
        # 中文注释：先给一个保守前进指令，让底层策略在站起过程中更稳定
        self.env.set_command(0.2, 0.0, 0.0)
        self.env.standup(duration=duration)
        self.env.reset_wall_clock()

    def apply_command(self, command: MotionCommand) -> None:
        self.env.set_command(command.linear_x, command.linear_y, command.yaw_rate)
        self.env.step_nomad_period()

    def get_pose(self):
        return self.env.get_pose()

    def get_height(self) -> float:
        return self.env.get_height()

    def get_forward_speed(self) -> float:
        return self.env.get_body_speed_forward()

    def is_fallen(self) -> bool:
        return self.env.is_fallen()

    def viewer_alive(self) -> bool:
        return self.env.viewer_alive()

    def reset_position(self, x: float, y: float, yaw: float = 0.0) -> None:
        self.env.set_robot_position(x, y, yaw=yaw)

    def prepare_task(self, mode: str) -> None:
        self.env.prepare_task(mode)

    def emergency_stop(self) -> None:
        # 中文注释：仿真和真机统一使用零速度刹停，避免状态退出时仍保留旧指令
        self.env.set_command(0.0, 0.0, 0.0)
        self.env.step_nomad_period()

    def close(self) -> None:
        self.env.close()


class ExternalBridgePlatform(NavigationPlatformBase):
    """Adapter for a Python real-robot bridge."""

    def __init__(self, bridge_module: str, bridge_class: str, bridge_kwargs: dict | None = None) -> None:
        module = importlib.import_module(bridge_module)
        bridge_type = getattr(module, bridge_class)
        self.bridge = bridge_type(**(bridge_kwargs or {}))

    def environment_domain(self) -> str:
        return "real"

    def _call(self, method_name: str, *args, default=None, required: bool = False, **kwargs):
        method = getattr(self.bridge, method_name, None)
        if method is None:
            if required:
                raise AttributeError(f"Bridge missing required method: {method_name}")
            return default
        return method(*args, **kwargs)

    def set_scene_goal(self, goal_position: np.ndarray) -> None:
        self._call("set_scene_goal", goal_position, required=False)

    def set_goal_markers(self, goal_positions: list[np.ndarray], active_index: int = 0) -> None:
        self._call("set_goal_markers", goal_positions, active_index=active_index, required=False)

    def highlight_goal_marker(self, active_index: int) -> None:
        self._call("highlight_goal_marker", active_index, required=False)

    def render_camera(self):
        return self._call("render_camera", required=True)

    def standup(self, duration: float) -> None:
        self._call("standup", duration, required=True)

    def apply_command(self, command: MotionCommand) -> None:
        if hasattr(self.bridge, "apply_command"):
            self.bridge.apply_command(command)
            return
        self._call(
            "send_command",
            command.linear_x,
            command.linear_y,
            command.yaw_rate,
            required=True,
        )

    def get_pose(self):
        return self._call("get_pose", required=True)

    def get_height(self) -> float:
        return float(self._call("get_height", default=0.0))

    def get_forward_speed(self) -> float:
        return float(self._call("get_forward_speed", default=0.0))

    def is_fallen(self) -> bool:
        return bool(self._call("is_fallen", default=False))

    def viewer_alive(self) -> bool:
        return bool(self._call("viewer_alive", default=True))

    def reset_position(self, x: float, y: float, yaw: float = 0.0) -> None:
        self._call("reset_position", x, y, yaw, required=False)

    def prepare_task(self, mode: str) -> None:
        self._call("prepare_task", mode, required=False)

    def emergency_stop(self) -> None:
        if hasattr(self.bridge, "emergency_stop"):
            self.bridge.emergency_stop()
            return
        if hasattr(self.bridge, "stop"):
            self.bridge.stop()
            return
        self.apply_command(MotionCommand(0.0, 0.0, 0.0))

    def close(self) -> None:
        self._call("close", required=False)


class ContextBuffer:
    def __init__(self, max_frames: int, transform_fn) -> None:
        self.transform_fn = transform_fn
        self.frames: Deque[torch.Tensor] = deque(maxlen=max_frames)

    def push(self, pil_image) -> None:
        self.frames.append(self.transform_fn(pil_image))

    def ready(self) -> bool:
        return len(self.frames) >= 2
