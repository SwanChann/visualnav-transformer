#!/usr/bin/env python3
"""Shared path helpers for thesis scripts."""

from __future__ import annotations

import sys
from pathlib import Path


# 中文注释：统一通过当前文件位置推导仓库根目录，避免硬编码绝对路径
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_ROOT = REPO_ROOT / "train"
DEPLOYMENT_SRC_ROOT = REPO_ROOT / "deployment" / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def add_repo_paths() -> None:
    """Add common repo import roots to sys.path once."""
    for path in (TRAIN_ROOT, DEPLOYMENT_SRC_ROOT, SCRIPTS_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def repo_path(*parts: str) -> Path:
    """Build a path under the repository root."""
    return REPO_ROOT.joinpath(*parts)
