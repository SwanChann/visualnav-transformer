#!/usr/bin/env python3
"""NoMaD + Tron1 MuJoCo integration scaffold."""

from __future__ import annotations

# Allow direct execution from any scripts/<category>/ path.
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve()
while SCRIPTS_ROOT.name != "scripts" and SCRIPTS_ROOT.parent != SCRIPTS_ROOT:
    SCRIPTS_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from tooling.project_paths import REPO_ROOT, repo_path


RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class AssetCheck:
    name: str
    path: str
    exists: bool
    required: bool
    note: str


def default_nomad_weight() -> Path:
    """Return the first existing NoMaD weight candidate."""
    candidates = [
        repo_path("deployment", "model_weights", "nomad.pth"),
        repo_path("deployment", "model_weights", "nomad", "nomad.pth"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_asset_checks(robot_name: str) -> List[AssetCheck]:
    """Build the Tron1 integration asset checklist."""
    tron1_root = repo_path("assets", robot_name)
    checks = [
        AssetCheck(
            name="nomad_weights",
            path=str(default_nomad_weight()),
            exists=default_nomad_weight().exists(),
            required=True,
            note="Shared NoMaD checkpoint used by the high-level navigation policy.",
        ),
        AssetCheck(
            name="tron1_mujoco_model",
            path=str(tron1_root / "mujoco" / f"{robot_name}.xml"),
            exists=(tron1_root / "mujoco" / f"{robot_name}.xml").exists(),
            required=True,
            note="MuJoCo XML or MJCF model for the Tron1 wheel-legged robot.",
        ),
        AssetCheck(
            name="tron1_policy",
            path=str(tron1_root / "policy" / "policy.onnx"),
            exists=(tron1_root / "policy" / "policy.onnx").exists(),
            required=False,
            note="Optional low-level wheel-legged locomotion policy exported to ONNX.",
        ),
        AssetCheck(
            name="tron1_camera_config",
            path=str(tron1_root / "config" / "camera.yaml"),
            exists=(tron1_root / "config" / "camera.yaml").exists(),
            required=False,
            note="Optional camera intrinsics/extrinsics for the MuJoCo or real robot bridge.",
        ),
        AssetCheck(
            name="tron1_bridge_notes",
            path=str(tron1_root / "README.md"),
            exists=(tron1_root / "README.md").exists(),
            required=False,
            note="Recommended place to document wheel, leg, and controller topic mappings.",
        ),
    ]
    return checks


def build_stage_plan(robot_name: str) -> List[dict]:
    """Return the implementation stages for the Tron1 MuJoCo adapter."""
    return [
        {
            "stage": "stage_1_assets",
            "goal": "Prepare the Tron1 MuJoCo model, camera settings, and NoMaD checkpoint.",
            "outputs": [
                f"assets/{robot_name}/mujoco/{robot_name}.xml",
                f"assets/{robot_name}/config/camera.yaml",
            ],
        },
        {
            "stage": "stage_2_perception",
            "goal": "Render the onboard RGB view and align it with the NoMaD observation transform.",
            "outputs": [
                "RGB frame stream at 4 Hz",
                "goal or topomap image loader",
            ],
        },
        {
            "stage": "stage_3_navigation",
            "goal": "Run NoMaD localization, goal matching, and DDPM or DDIM action sampling.",
            "outputs": [
                "sampled trajectories",
                "selected waypoint",
            ],
        },
        {
            "stage": "stage_4_bridge",
            "goal": "Convert the selected waypoint into wheel-legged chassis commands and safety-constrained references.",
            "outputs": [
                "forward velocity command",
                "yaw rate command",
                "wheel-leg mode switch or guard rails",
            ],
        },
        {
            "stage": "stage_5_execution",
            "goal": "Connect the high-level command bridge to Tron1 locomotion control in MuJoCo.",
            "outputs": [
                "wheel speed interface",
                "leg posture or stabilizer interface",
            ],
        },
        {
            "stage": "stage_6_evaluation",
            "goal": "Export navigation logs for thesis figures, comparison tables, and deployment readiness review.",
            "outputs": [
                "trajectory.txt",
                "summary.txt",
                "platform comparison notes",
            ],
        },
    ]


def render_report(robot_name: str, checks: List[AssetCheck], stages: List[dict]) -> str:
    """Render a Markdown planning report."""
    lines = [
        f"# NoMaD Tron1 MuJoCo Integration Report",
        "",
        f"- Robot: `{robot_name}`",
        f"- Repo Root: `{REPO_ROOT}`",
        f"- Generated At: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Asset Checks",
        "",
        "| Item | Exists | Required | Path | Note |",
        "|---|---|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check.name} | {'yes' if check.exists else 'no'} | "
            f"{'yes' if check.required else 'no'} | `{check.path}` | {check.note} |"
        )

    lines.extend(
        [
            "",
            "## Implementation Stages",
            "",
            "| Stage | Goal | Outputs |",
            "|---|---|---|",
        ]
    )
    for stage in stages:
        outputs = "<br>".join(stage["outputs"])
        lines.append(f"| {stage['stage']} | {stage['goal']} | {outputs} |")

    lines.extend(
        [
            "",
            "## Recommended Next Commands",
            "",
            "```bash",
            "python3 scripts/simulation/nomad_mujoco_tron1_nav.py --mode validate-assets",
            "python3 scripts/simulation/nomad_mujoco_tron1_nav.py --mode export-manifest --save",
            "```",
        ]
    )
    return "\n".join(lines)


def save_outputs(robot_name: str, checks: List[AssetCheck], stages: List[dict]) -> Path:
    """Save JSON and Markdown outputs under results/day6."""
    out_dir = repo_path("results", "day6", f"{RUN_TAG}_{robot_name}_tron1_plan")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "robot": robot_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "asset_checks": [asdict(check) for check in checks],
        "stages": stages,
    }
    json_path = out_dir / "tron1_manifest.json"
    report_path = out_dir / "tron1_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(render_report(robot_name, checks, stages), encoding="utf-8")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Tron1 MuJoCo integration helper")
    parser.add_argument(
        "--mode",
        choices=["validate-assets", "print-plan", "export-manifest"],
        default="validate-assets",
    )
    parser.add_argument("--robot", default="tron1", help="Robot asset prefix")
    parser.add_argument("--save", action="store_true", help="Save generated outputs")
    args = parser.parse_args()

    checks = build_asset_checks(args.robot)
    stages = build_stage_plan(args.robot)
    required_missing = [check for check in checks if check.required and not check.exists]

    if args.mode == "validate-assets":
        for check in checks:
            status = "OK" if check.exists else "MISSING"
            print(f"[{status}] {check.name}: {check.path}")
            print(f"        {check.note}")
        if required_missing:
            print(f"\nMissing required assets: {len(required_missing)}")
            return 1
        print("\nAll required assets are present.")
        return 0

    if args.mode == "print-plan":
        print(render_report(args.robot, checks, stages))
        if args.save:
            out_dir = save_outputs(args.robot, checks, stages)
            print(f"\nSaved report to: {out_dir}")
        return 0 if not required_missing else 1

    out_dir = save_outputs(args.robot, checks, stages)
    print(f"Saved manifest to: {out_dir}")
    return 0 if not required_missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
