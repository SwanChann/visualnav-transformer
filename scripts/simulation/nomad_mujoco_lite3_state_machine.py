#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = CURRENT_DIR.parent
for candidate in (CURRENT_DIR, SCRIPTS_ROOT, SCRIPTS_ROOT / "shared"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lite3 MuJoCo state-machine system")
    parser.add_argument(
        "--mode",
        choices=["stand", "navigate", "explore", "walk-test", "generate-topomap", "mission", "estop"],
        default="navigate",
    )
    parser.add_argument("--map", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument(
        "--camera",
        choices=["on", "off"],
        default="on",
        help="Whether to show the unified FPV + goal-vision camera window.",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--waypoint", type=int, default=2)
    parser.add_argument("--standup-time", type=float, default=3.0)
    parser.add_argument("--save-fpv", action="store_true")
    parser.add_argument(
        "--topomap-traj",
        type=str,
        default=None,
        help="Deprecated in the closed-loop pipeline. Use a domain-matched --topomap-dir instead.",
    )
    parser.add_argument(
        "--topomap-dir",
        type=str,
        default=None,
        help="MuJoCo: defaults to topomaps/<map>. Real: must be a real-world collected topomap directory.",
    )
    parser.add_argument("--topomap-step", type=int, default=5)
    parser.add_argument("--topomap-nodes", type=int, default=20)
    parser.add_argument(
        "--goal-source",
        choices=["topomap", "random_points", "capture_queue"],
        default="topomap",
        help="navigate/mission goal source: topomap directory, MuJoCo random goal points, or shared capture queue.",
    )
    parser.add_argument("--num-goals", type=int, default=1, help="Number of navigation goals for random_points or capture_queue.")
    parser.add_argument("--capture-index", type=int, default=-1, help="Capture queue index for goal-source=capture_queue (-1 means selected/latest).")
    parser.add_argument("--stand-steps", type=int, default=20, help="How many control steps the stand task should hold position.")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--close-threshold", type=float, default=3.0)
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale used during inference.",
    )
    parser.add_argument("--tts", action="store_true", help="Enable test-time scaling for diffusion trajectory selection.")
    parser.add_argument("--tts-budget", type=int, default=8, help="Total number of sampled trajectories considered by TTS.")
    parser.add_argument("--tts-topk", type=int, default=1, help="How many top-ranked TTS trajectories to keep before averaging.")
    parser.add_argument(
        "--tts-verifier",
        choices=["heuristic", "forward", "conservative"],
        default="heuristic",
        help="Trajectory verifier used by TTS candidate selection.",
    )
    parser.add_argument(
        "--policy-config",
        type=str,
        default=None,
        help="Optional policy config path. Leave empty to use the default NoMaD baseline config.",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=str,
        default=None,
        help="Optional policy checkpoint path. Leave empty to use the default NoMaD baseline checkpoint.",
    )
    parser.add_argument(
        "--policy-device",
        type=str,
        default=None,
        help="Optional policy device override such as cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--image-resize-mode",
        choices=["stretch", "center_crop", "letterbox"],
        default="stretch",
        help="Camera/topomap preprocessing before NoMaD inference.",
    )
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--mission-topomap", action="append", default=None)
    parser.add_argument(
        "--mission-dataset-traj",
        action="append",
        default=None,
        help="Deprecated in the closed-loop pipeline. Use mission-topomap directories instead.",
    )
    parser.add_argument("--random-goal-min-dist", type=float, default=2.0, help="Minimum distance from current pose to sampled MuJoCo goal.")
    parser.add_argument("--random-goal-min-separation", type=float, default=1.0, help="Minimum separation between sampled MuJoCo goal points.")
    parser.add_argument("--goal-seed", type=int, default=0, help="Deterministic seed for same-map random goal selection.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    from lite3_system.system import Lite3System

    system = Lite3System(args)
    return system.run()


if __name__ == "__main__":
    raise SystemExit(main())
