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


def scale_policy_actions(samples: np.ndarray, action_scale_m: float) -> np.ndarray:
    """Convert native NoMaD action units to target-platform metric waypoints."""
    scale = float(action_scale_m)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("action_scale_m must be finite and positive")
    values = np.asarray(samples, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 2 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite policy samples [N,T,2], got {values.shape}")
    return values * scale


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

    def has_navigation_collision(self) -> bool:
        return False

    def viewer_alive(self) -> bool:
        raise NotImplementedError

    def reset_position(self, x: float, y: float, yaw: float = 0.0) -> None:
        return None

    def prepare_task(self, mode: str) -> None:
        return None

    def emergency_stop(self) -> None:
        # 中文注释：统一安全接口，默认发送零速度指令
        self.apply_command(MotionCommand(0.0, 0.0, 0.0))

    def stop_motion(self) -> None:
        self.apply_command(MotionCommand(0.0, 0.0, 0.0))

    def set_gait(self, gait: str) -> None:
        return None

    def prepare_for_twist_control(self) -> None:
        return None

    def release_manual_control(self) -> None:
        return None

    def close(self) -> None:
        return None

    def camera_status(self) -> str:
        return ""


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
        action_scale_m: float = 1.0,
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
        self.action_scale_m = float(action_scale_m)
        scale_policy_actions(np.zeros((1, 1, 2)), self.action_scale_m)

    def preprocess_frame(self, frame):
        """Expose preprocessing through the backend boundary, not its inference internals."""
        return self.inference.pil_to_tensor(frame)

    def close(self) -> None:
        """Release backend resources; the current eager NoMaD backend owns none."""
        return None

    def _build_obs_tensor(self, frame_buffer: Deque[torch.Tensor]) -> torch.Tensor:
        return self.inference.build_obs_tensor(frame_buffer).to(self.device)

    def _sample_actions(self, obs_cond: torch.Tensor, uncond: torch.Tensor | None = None) -> np.ndarray:
        return self.inference.sample_actions(obs_cond, uncond_condition=uncond)

    def build_topomap_tensor(self, topomap: list) -> torch.Tensor:
        """Preprocess a topomap once so navigation ticks do not repeat PIL work."""
        tensors = [self.inference.pil_to_tensor(goal_image) for goal_image in topomap]
        return torch.stack(tensors, dim=0).to(self.device)

    def predict_exploration(self, frame_buffer: Deque[torch.Tensor]) -> ExplorationResult:
        obs_tensor = self._build_obs_tensor(frame_buffer)
        image_height = int(self.inference.image_size[1])
        image_width = int(self.inference.image_size[0])
        fake_goal = torch.randn((1, 3, image_height, image_width), device=self.device)
        obs_cond = self.inference.encode_condition(obs_tensor, fake_goal, goal_mask_value=1)

        sampled_actions = scale_policy_actions(self._sample_actions(obs_cond), self.action_scale_m)
        mean_action = sampled_actions.mean(axis=0)
        chosen_waypoint = mean_action[min(self.waypoint_index, self.inference.len_traj_pred - 1)]
        return ExplorationResult(chosen_waypoint=chosen_waypoint, sampled_actions=sampled_actions)

    def predict_navigation(
        self,
        frame_buffer: Deque[torch.Tensor],
        topomap: list,
        closest_node: int,
        goal_node: int,
        topomap_tensor: torch.Tensor | None = None,
    ) -> NavigationResult:
        obs_tensor = self._build_obs_tensor(frame_buffer)
        start = max(closest_node - self.radius, 0)
        end = min(closest_node + self.radius + 1, goal_node)

        if topomap_tensor is not None:
            goal_batch = topomap_tensor[start : end + 1]
        else:
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

        sampled_actions = scale_policy_actions(self._sample_actions(obs_cond, uncond=uncond_cond), self.action_scale_m)
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
    # 硬上限：扩散采样偶尔会输出远离观察尺度的 waypoint，中层在喂给 PD 控制器前先压住。
    MAX_WAYPOINT_NORM = 2.0  # 米

    def __init__(
        self,
        yaw_sign: float = 1.0,
        linear_scale: float = 1.0,
        yaw_scale: float = 1.0,
        max_linear: float | None = None,
        max_yaw: float | None = None,
    ) -> None:
        self.legacy = load_legacy()
        self.yaw_sign = float(yaw_sign)
        self.linear_scale = float(linear_scale)
        self.yaw_scale = float(yaw_scale)
        self.max_linear = None if max_linear is None else abs(float(max_linear))
        self.max_yaw = None if max_yaw is None else abs(float(max_yaw))

    def waypoint_to_command(self, waypoint: np.ndarray) -> MotionCommand:
        waypoint = np.asarray(waypoint, dtype=float)
        # NaN/Inf 守卫：扩散或编码层异常时可能产出非有限值，必须拦在进 PD 前
        if not np.all(np.isfinite(waypoint)):
            print(f"[MiddleLayer] ⚠️ waypoint contains non-finite values: {waypoint}; issuing zero command")
            return MotionCommand(linear_x=0.0, linear_y=0.0, yaw_rate=0.0)

        norm = float(np.linalg.norm(waypoint))
        if norm > self.MAX_WAYPOINT_NORM:
            waypoint = waypoint * (self.MAX_WAYPOINT_NORM / norm)

        pd_kwargs = {"yaw_sign": self.yaw_sign}
        if self.max_linear is not None:
            pd_kwargs["max_v"] = self.max_linear
        if self.max_yaw is not None:
            pd_kwargs["max_w"] = self.max_yaw
        linear_x, yaw_rate = self.legacy.pd_controller(waypoint, **pd_kwargs)
        linear_x = float(linear_x)
        yaw_rate = float(yaw_rate)
        # 二次守卫：PD 输出本身也可能被上游的 NaN 传染
        if not (np.isfinite(linear_x) and np.isfinite(yaw_rate)):
            print(f"[MiddleLayer] ⚠️ PD output non-finite: lx={linear_x}, wz={yaw_rate}; issuing zero command")
            return MotionCommand(linear_x=0.0, linear_y=0.0, yaw_rate=0.0)
        linear_x *= self.linear_scale
        yaw_rate *= self.yaw_scale
        if not (np.isfinite(linear_x) and np.isfinite(yaw_rate)):
            print(f"[MiddleLayer] ⚠️ scaled PD output non-finite: lx={linear_x}, wz={yaw_rate}; issuing zero command")
            return MotionCommand(linear_x=0.0, linear_y=0.0, yaw_rate=0.0)
        if self.max_linear is not None:
            linear_x = float(np.clip(linear_x, -self.max_linear, self.max_linear))
        if self.max_yaw is not None:
            yaw_rate = float(np.clip(yaw_rate, -self.max_yaw, self.max_yaw))
        return MotionCommand(linear_x=linear_x, linear_y=0.0, yaw_rate=yaw_rate)


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

    def has_navigation_collision(self) -> bool:
        mujoco = self.legacy.mujoco
        scene_prefixes = ("wall_", "obstacle")
        for contact_index in range(int(self.env.data.ncon)):
            contact = self.env.data.contact[contact_index]
            geom1 = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
            geom2 = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
            if geom1.startswith(scene_prefixes) != geom2.startswith(scene_prefixes):
                return True
        return False

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

    def stop_motion(self) -> None:
        self.emergency_stop()

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

    def stop_motion(self) -> None:
        if hasattr(self.bridge, "stop_motion"):
            self.bridge.stop_motion()
            return
        if hasattr(self.bridge, "stop"):
            self.bridge.stop()
            return
        self.apply_command(MotionCommand(0.0, 0.0, 0.0))

    def set_gait(self, gait: str) -> None:
        self._call("set_gait", gait, required=False)

    def prepare_for_twist_control(self) -> None:
        self._call("prepare_for_twist_control", required=False)

    def release_manual_control(self) -> None:
        self._call("release_manual_control", required=False)
        if hasattr(self, "_nomad_is_standing"):
            self._nomad_is_standing = False

    def close(self) -> None:
        self._call("close", required=False)

    def camera_status(self) -> str:
        return str(self._call("camera_status", default=""))


class ContextBuffer:
    def __init__(self, max_frames: int, transform_fn) -> None:
        self.transform_fn = transform_fn
        self.frames: Deque[torch.Tensor] = deque(maxlen=max_frames)

    def push(self, pil_image) -> None:
        self.frames.append(self.transform_fn(pil_image))

    def ready(self) -> bool:
        return len(self.frames) >= 2
