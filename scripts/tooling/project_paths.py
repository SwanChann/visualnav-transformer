#!/usr/bin/env python3
"""Shared path helpers for repository scripts."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO_ROOT / "train"
DEPLOYMENT_SRC_ROOT = REPO_ROOT / "deployment" / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"

LITE3_VENDOR_ROOT = THIRD_PARTY_ROOT / "lite3"
LITE3_HOST_CONTROL_ROOT = LITE3_VENDOR_ROOT / "lite3_host_control"
LITE3_SDK_DEPLOY_ROOT = LITE3_VENDOR_ROOT / "sdk_deploy"
LITE3_RL_DEPLOY_ROOT = (
    LITE3_VENDOR_ROOT / "Lite3_rl_deploy"
    if (LITE3_VENDOR_ROOT / "Lite3_rl_deploy").exists()
    else REPO_ROOT / "Lite3_rl_deploy"
)

LITE3_MJCF_XML = (
    LITE3_SDK_DEPLOY_ROOT
    / "src"
    / "Lite3_sdk_deploy"
    / "Lite3_description"
    / "lite3_mjcf"
    / "mjcf"
    / "Lite3.xml"
)
LITE3_LOCOMOTION_POLICY = (
    LITE3_SDK_DEPLOY_ROOT / "src" / "Lite3_sdk_deploy" / "policy" / "policy.onnx"
)


def add_repo_paths() -> None:
    """Add common repo import roots to sys.path once."""
    for path in (TRAIN_ROOT, DEPLOYMENT_SRC_ROOT, SCRIPTS_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def repo_path(*parts: str) -> Path:
    """Build a path under the repository root."""
    return REPO_ROOT.joinpath(*parts)
