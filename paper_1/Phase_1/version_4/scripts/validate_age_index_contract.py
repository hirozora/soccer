#!/usr/bin/env python
"""Validate the F80 local-rank age contract over all England matches."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.age_index_validation import validate_age_index_contract  # noqa: E402
from football_hgt_targets_v4.age_pooling_study import AGE_POOL_ROOT  # noqa: E402


if __name__ == "__main__":
    print(validate_age_index_contract(AGE_POOL_ROOT / "validation/age_index_contract.json"))
