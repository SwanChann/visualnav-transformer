from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch

from lite3_system.legacy_bridge import load_legacy


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


class Lite3HighLevelNoMaD:
    def __init__(
        self,
        scheduler_kind: str,
        ddim_steps: int,
        waypoint_index: int,
        radius: int,
        close_threshold: float,
        cfg_weight: float = 0.0,
        device: str | None = None,
    ) -> None:
        self.legacy = load_legacy()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.legacy.build_nomad_model(self.device)
        self.scheduler_kind = scheduler_kind
        self.ddim_steps = ddim_steps
        self.waypoint_index = waypoint_index
        self.radius = radius
        self.close_threshold = close_threshold
        self.cfg_weight = cfg_weight

    def _build_obs_tensor(self, frame_buffer: Deque[torch.Tensor]) -> torch.Tensor:
        return self.legacy.build_obs_tensor(frame_buffer).to(self.device)

    def _sample_actions(self, obs_cond: torch.Tensor, uncond: torch.Tensor | None = None) -> np.ndarray:
        num_steps = self.ddim_steps if self.scheduler_kind == "ddim" else self.legacy.NUM_DIFFUSION_ITERS
        return self.legacy.diffusion_inference(
            self.model,
            obs_cond,
            self.device,
            scheduler_kind=self.scheduler_kind,
            num_steps=num_steps,
            guidance_scale=self.cfg_weight if self.cfg_weight else None,
            uncond=uncond,
        )

    def predict_exploration(self, frame_buffer: Deque[torch.Tensor]) -> ExplorationResult:
        obs_tensor = self._build_obs_tensor(frame_buffer)
        fake_goal = torch.randn((1, 3, *self.legacy.IMAGE_SIZE), device=self.device)
        mask = torch.ones(1, dtype=torch.long, device=self.device)

        with torch.no_grad():
            obs_cond = self.model(
                "vision_encoder",
                obs_img=obs_tensor,
                goal_img=fake_goal,
                input_goal_mask=mask,
            )

        sampled_actions = self._sample_actions(obs_cond)
        mean_action = sampled_actions.mean(axis=0)
        chosen_waypoint = mean_action[min(self.waypoint_index, self.legacy.LEN_TRAJ_PRED - 1)]
        return ExplorationResult(chosen_waypoint=chosen_waypoint, sampled_actions=sampled_actions)

    def predict_navigation(
        self,
        frame_buffer: Deque[torch.Tensor],
        topomap: list,
        closest_node: int,
        goal_node: int,
    ) -> NavigationResult:
        obs_tensor = self._build_obs_tensor(frame_buffer)
        mask = torch.zeros(1, dtype=torch.long, device=self.device)
        start = max(closest_node - self.radius, 0)
        end = min(closest_node + self.radius + 1, goal_node)

        goal_tensors = []
        for goal_image in topomap[start : end + 1]:
            goal_tensors.append(self.legacy.pil_to_tensor(goal_image).unsqueeze(0).to(self.device))
        goal_batch = torch.cat(goal_tensors, dim=0)
        candidate_count = goal_batch.shape[0]

        with torch.no_grad():
            obsgoal_cond = self.model(
                "vision_encoder",
                obs_img=obs_tensor.repeat(candidate_count, 1, 1, 1),
                goal_img=goal_batch,
                input_goal_mask=mask.repeat(candidate_count),
            )
            distances = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond).cpu().numpy().flatten()

        min_index = int(np.argmin(distances))
        raw_closest = min_index + start
        clipped_closest = int(np.clip(raw_closest, closest_node - 1, closest_node + 3))
        clipped_closest = min(clipped_closest, goal_node)
        local_index = clipped_closest - start
        selected_index = min(local_index + int(distances[min_index] < self.close_threshold), len(obsgoal_cond) - 1)
        obs_cond = obsgoal_cond[selected_index].unsqueeze(0)

        # CFG: 计算无条件编码
        uncond_cond = None
        if self.cfg_weight and self.cfg_weight != 0.0:
            mask_explore = torch.ones(1, dtype=torch.long, device=self.device)
            with torch.no_grad():
                uncond_cond = self.model(
                    "vision_encoder",
                    obs_img=obs_tensor,
                    goal_img=goal_batch[selected_index:selected_index+1],
                    input_goal_mask=mask_explore,
                )

        sampled_actions = self._sample_actions(obs_cond, uncond=uncond_cond)
        mean_action = sampled_actions.mean(axis=0)
        chosen_waypoint = mean_action[min(self.waypoint_index, self.legacy.LEN_TRAJ_PRED - 1)]
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


class Lite3LowLevelPlatform:
    def __init__(self, gui: bool, scene_name: str) -> None:
        self.legacy = load_legacy()
        self.legacy.SCENE_CONFIG = self.legacy.SCENE_MAPS[scene_name]
        self.env = self.legacy.MuJoCoLite3Env(gui=gui)

    def set_scene_goal(self, goal_position: np.ndarray) -> None:
        self.legacy.SCENE_CONFIG["goal_pos"] = [float(goal_position[0]), float(goal_position[1])]
        self.env.move_goal_marker(float(goal_position[0]), float(goal_position[1]))

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

    def close(self) -> None:
        self.env.close()


class ContextBuffer:
    def __init__(self, max_frames: int) -> None:
        self.legacy = load_legacy()
        self.frames: Deque[torch.Tensor] = deque(maxlen=max_frames)

    def push(self, pil_image) -> None:
        self.frames.append(self.legacy.pil_to_tensor(pil_image))

    def ready(self) -> bool:
        return len(self.frames) >= 2
