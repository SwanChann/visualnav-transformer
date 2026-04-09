from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from categorized_legacy import load_legacy_module

_LEGACY = load_legacy_module("nomad_vint_dinov2.py", alias="categorized_nomad_vint_dinov2")
__doc__ = _LEGACY.__doc__

for _name in dir(_LEGACY):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_LEGACY, _name)

__all__ = [name for name in globals() if not name.startswith("_")]
