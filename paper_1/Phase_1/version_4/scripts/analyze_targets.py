#!/usr/bin/env python
"""Analyze training-only next-event target distributions and dependencies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SRC = ROOT.parent / "benchmark_unified_v1/src"
sys.path[:0] = [str(ROOT / "src"), str(BENCHMARK_SRC)]

from football_hgt_targets_v4.analysis import run_analysis  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run_analysis(), indent=2))
