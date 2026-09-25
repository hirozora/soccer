#!/usr/bin/env python
"""Train or evaluate one strict receptive-field Partial-L2 model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.constants import SAMPLE_PLAN  # noqa: E402
from football_hgt_targets_v4.fixed_budget_training import FixedBudgetConfig, evaluate_fixed_budget_checkpoint, run_fixed_budget_training  # noqa: E402
from football_hgt_targets_v4.partial_sharing_study import training_dir as f80_training_dir  # noqa: E402
from football_hgt_targets_v4.receptive_field import RF_CONFIGURATIONS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=RF_CONFIGURATIONS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path)
    args = parser.parse_args()
    if args.evaluate_test and args.checkpoint_path is None:
        parser.error("--evaluate-test requires --checkpoint-path")
    config = FixedBudgetConfig(
        configuration=args.configuration,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        training_budget=1 if args.smoke else args.budget,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_plan_path=None if args.evaluate_test else SAMPLE_PLAN,
        max_train_samples=128 if args.smoke else None,
        max_validation_samples=64 if args.smoke else None,
        evaluate_test=args.evaluate_test,
        full_test=args.evaluate_test,
        secondary_guard_reference_path=f80_training_dir(args.seed) / "result.json",
    )
    result = (
        evaluate_fixed_budget_checkpoint(config, args.checkpoint_path)
        if args.evaluate_test
        else run_fixed_budget_training(config)
    )
    print(json.dumps({
        "configuration": args.configuration,
        "best_epoch": result.get("best_epoch"),
        "training_complete": result.get("training_complete"),
        "test_accessed": result["test_accessed"],
    }, indent=2))


if __name__ == "__main__":
    main()
