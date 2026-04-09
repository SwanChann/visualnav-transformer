#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parent


def run_legacy(script_name: str) -> None:
    runpy.run_path(str(SCRIPTS_ROOT / script_name), run_name="__main__")


def load_legacy_module(script_name: str, alias: str | None = None):
    path = SCRIPTS_ROOT / script_name
    module_name = alias or f"legacy_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
