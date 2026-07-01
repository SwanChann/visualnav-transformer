#!/usr/bin/env python3
"""Legacy alias for `scripts/analysis/thesis_result_summary.py`."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve()
while SCRIPTS_ROOT.name != "scripts" and SCRIPTS_ROOT.parent != SCRIPTS_ROOT:
    SCRIPTS_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from analysis.thesis_result_summary import main


if __name__ == "__main__":
    raise SystemExit(main())
