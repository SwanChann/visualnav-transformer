#!/usr/bin/env python3
"""Generate real deployment checklists for Lite3 and Tron1."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from project_paths import repo_path


RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class ChecklistItem:
    category: str
    item: str
    required_file: str
    exists: bool
    note: str


def weight_candidates() -> List[Path]:
    """Return possible NoMaD checkpoint locations."""
    return [
        repo_path("deployment", "model_weights", "nomad.pth"),
        repo_path("deployment", "model_weights", "nomad", "nomad.pth"),
    ]


def first_weight_path() -> Path:
    """Return the first existing weight or the primary default."""
    for candidate in weight_candidates():
        if candidate.exists():
            return candidate
    return weight_candidates()[0]


def platform_rules(platform: str) -> Dict[str, List[tuple]]:
    """Return platform-specific checklist rules."""
    common = [
        ("base", "NoMaD checkpoint", first_weight_path(), "High-level navigation checkpoint"),
        ("base", "navigation node", repo_path("deployment", "src", "navigate.py"), "Topomap localization and waypoint generation"),
        ("base", "pd controller", repo_path("deployment", "src", "pd_controller.py"), "Waypoint-to-velocity bridge"),
        ("base", "model config", repo_path("deployment", "config", "models.yaml"), "Checkpoint and model parameter registry"),
        ("base", "robot config", repo_path("deployment", "config", "robot.yaml"), "Velocity bounds and topic names"),
    ]

    if platform == "lite3":
        extra = [
            ("platform", "Lite3 MuJoCo model", repo_path("sdk_deploy", "src", "Lite3_sdk_deploy", "Lite3_description", "lite3_mjcf", "mjcf", "Lite3.xml"), "Simulation asset for integrated validation"),
            ("platform", "Lite3 locomotion policy", repo_path("sdk_deploy", "src", "Lite3_sdk_deploy", "policy", "policy.onnx"), "Low-level RL locomotion policy"),
            ("platform", "Lite3 MuJoCo runner", repo_path("scripts", "nomad_mujoco_lite3_nav.py"), "Integrated NoMaD + MuJoCo + RL navigation entry"),
        ]
    else:
        extra = [
            ("platform", "Tron1 MuJoCo helper", repo_path("scripts", "nomad_mujoco_tron1_nav.py"), "Asset validation and implementation staging entry"),
            ("platform", "Tron1 MuJoCo model", repo_path("assets", "tron1", "mujoco", "tron1.xml"), "Wheel-legged MuJoCo XML or MJCF asset"),
            ("platform", "Tron1 low-level controller", repo_path("assets", "tron1", "policy", "policy.onnx"), "Optional wheel-legged controller export"),
        ]
    return {"common": common, "extra": extra}


def build_checklist(platform: str) -> List[ChecklistItem]:
    """Build the deployment checklist for the selected platform."""
    rules = platform_rules(platform)
    items: List[ChecklistItem] = []
    for category, item, file_path, note in rules["common"] + rules["extra"]:
        items.append(
            ChecklistItem(
                category=category,
                item=item,
                required_file=str(file_path),
                exists=file_path.exists(),
                note=note,
            )
        )
    return items


def render_markdown(platform: str, items: List[ChecklistItem]) -> str:
    """Render a Markdown checklist."""
    lines = [
        f"# NoMaD Real Deployment Checklist",
        "",
        f"- Platform: `{platform}`",
        f"- Generated At: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "| Category | Item | Exists | Required File | Note |",
        "|---|---|---|---|---|",
    ]
    for item in items:
        lines.append(
            f"| {item.category} | {item.item} | {'yes' if item.exists else 'no'} | "
            f"`{item.required_file}` | {item.note} |"
        )

    lines.extend(
        [
            "",
            "## Deployment Review Steps",
            "",
            "1. Confirm the camera topic, robot command topic, and topomap input path.",
            "2. Verify the NoMaD checkpoint and the platform-specific low-level controller are both available.",
            "3. Run a static camera test before enabling closed-loop locomotion.",
            "4. Enable velocity limits and recovery mode before the first autonomous run.",
            "5. Record logs, images, and platform exceptions for thesis evidence collection.",
        ]
    )
    return "\n".join(lines)


def save_markdown(platform: str, content: str) -> Path:
    """Save the checklist under results/deployment."""
    out_dir = repo_path("results", "deployment", f"{RUN_TAG}_{platform}_checklist")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "deployment_checklist.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Real deployment checklist helper")
    parser.add_argument("--platform", choices=["lite3", "tron1"], required=True)
    parser.add_argument("--save", action="store_true", help="Save the checklist to results")
    args = parser.parse_args()

    items = build_checklist(args.platform)
    content = render_markdown(args.platform, items)
    print(content)

    if args.save:
        output_path = save_markdown(args.platform, content)
        print(f"\nSaved checklist to: {output_path}")

    missing_required = [
        item for item in items if item.category in {"base", "platform"} and not item.exists
    ]
    return 0 if not missing_required else 1


if __name__ == "__main__":
    raise SystemExit(main())
