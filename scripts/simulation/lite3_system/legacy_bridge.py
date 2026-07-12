from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_legacy():
    simulation_root = Path(__file__).resolve().parent.parent
    legacy_path = simulation_root / "nomad_mujoco_lite3_nav.py"
    if not legacy_path.is_file():
        raise FileNotFoundError(f"Legacy Lite3 script not found: {legacy_path}")
    spec = importlib.util.spec_from_file_location("legacy_nomad_mujoco_lite3_nav", legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy Lite3 script: {legacy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
