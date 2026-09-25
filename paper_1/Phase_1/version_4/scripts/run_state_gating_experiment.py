#!/usr/bin/env python
"""Train or evaluate one state-aware Partial-L2 seed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.constants import SAMPLE_PLAN  # noqa: E402
from football_hgt_targets_v4.fixed_budget_training import (  # noqa: E402
    FixedBudgetConfig,
    evaluate_fixed_budget_checkpoint,
    run_fixed_budget_training,
)
from football_hgt_targets_v4.state_gating_reporting import selected_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.evaluate_test:
        if args.checkpoint is None:
            parser.error("--evaluate-test requires --checkpoint")
        if selected_model() != "G1":
            parser.error("G1 test evaluation is forbidden because validation did not lock G1")
    config = FixedBudgetConfig(
        configuration="state_gated_partial_l2",
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        training_budget=1 if args.smoke else 24,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_plan_path=None if args.evaluate_test else SAMPLE_PLAN,
        max_train_samples=128 if args.smoke else None,
        max_validation_samples=64 if args.smoke else None,
        evaluate_test=args.evaluate_test,
        full_test=args.evaluate_test,
    )
    result = (
        evaluate_fixed_budget_checkpoint(config, args.checkpoint)
        if args.evaluate_test
        else run_fixed_budget_training(config)
    )
    print(json.dumps({
        "seed": args.seed,
        "best_epoch": result["best_epoch"],
        "test_accessed": result["test_accessed"],
        "validation": result.get("validation"),
        "test": result.get("test"),
    }, indent=2))


if __name__ == "__main__":
    main()

