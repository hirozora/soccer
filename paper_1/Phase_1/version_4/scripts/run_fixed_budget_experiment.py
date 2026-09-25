#!/usr/bin/env python
"""Train, resume, or evaluate one fixed-budget multitask configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.constants import SAMPLE_PLAN  # noqa: E402
from football_hgt_targets_v4.fixed_budget_study import CONFIGURATIONS, FIXED_BUDGET_ROOT  # noqa: E402
from football_hgt_targets_v4.fixed_budget_training import (  # noqa: E402
    FixedBudgetConfig,
    evaluate_fixed_budget_checkpoint,
    run_fixed_budget_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.evaluate_test:
        if args.checkpoint_path is None or not args.full_test:
            parser.error("test evaluation requires --checkpoint-path and --full-test")
        if not (FIXED_BUDGET_ROOT / "decisions/baseline_lock.json").exists():
            parser.error("test evaluation is forbidden before baseline lock")
    budget = 1 if args.smoke else args.budget
    config = FixedBudgetConfig(
        configuration=args.configuration,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        training_budget=budget,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_plan_path=None if args.full_test else SAMPLE_PLAN,
        max_train_samples=128 if args.smoke else None,
        max_validation_samples=64 if args.smoke else None,
        evaluate_test=args.evaluate_test,
        full_test=args.full_test,
        resume_from=args.resume_from,
    )
    result = (
        evaluate_fixed_budget_checkpoint(config, args.checkpoint_path)
        if args.evaluate_test
        else run_fixed_budget_training(config)
    )
    print(json.dumps({
        "configuration": args.configuration,
        "epochs_completed": result.get("epochs_completed"),
        "best_epoch": result["best_epoch"],
        "validation": result.get("validation"),
        "test": result.get("test"),
        "test_accessed": result["test_accessed"],
    }, indent=2))


if __name__ == "__main__":
    main()
