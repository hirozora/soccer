#!/usr/bin/env python3
"""Repository-local entry point for the heterogeneous graph builder."""

from __future__ import annotations

import sys
from pathlib import Path


PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT / "src"))

from football_hgt.build_graphs import main  # noqa: E402


if __name__ == "__main__":
    main()
