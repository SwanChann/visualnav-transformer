#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lite3 MuJoCo state-machine system")
    parser.add_argument(
        "--mode",
        choices=["navigate", "explore", "walk-test", "generate-topomap", "mission"],
        default="navigate",
    )
    parser.add_argument("--map", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--waypoint", type=int, default=2)
    parser.add_argument("--standup-time", type=float, default=3.0)
    parser.add_argument("--save-fpv", action="store_true")
    parser.add_argument("--topomap-traj", type=str, default="no10vc_10_0")
    parser.add_argument("--topomap-dir", type=str, default=None)
    parser.add_argument("--topomap-step", type=int, default=5)
    parser.add_argument("--topomap-nodes", type=int, default=20)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--close-threshold", type=float, default=3.0)
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--mission-topomap", action="append", default=None)
    parser.add_argument("--mission-dataset-traj", action="append", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    from lite3_system.system import Lite3System

    system = Lite3System(args)
    return system.run()


if __name__ == "__main__":
    raise SystemExit(main())
