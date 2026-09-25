#!/usr/bin/env python
"""Run one Team-aware Player candidate-prior action."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.team_candidate_prior import build_team_candidate_cache  # noqa: E402
from football_hgt_targets_v4.team_candidate_prior_reporting import (  # noqa: E402
    build_final_report,
    evaluate_locked_test,
    lock_validation_decision,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("cache", "lock", "test", "report"), required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "cache":
        if args.split is None:
            parser.error("cache requires --split")
        print(build_team_candidate_cache(args.seed, args.split, args.device))
    elif args.action == "lock":
        print(lock_validation_decision())
    elif args.action == "test":
        print(evaluate_locked_test())
    else:
        print(build_final_report())


if __name__ == "__main__":
    main()

