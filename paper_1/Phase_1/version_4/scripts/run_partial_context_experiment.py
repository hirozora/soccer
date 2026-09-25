#!/usr/bin/env python
"""Train or evaluate one Partial-L2 task-context configuration."""

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
from football_hgt_targets_v4.partial_context_study import (  # noqa: E402
    TRAINED_CONFIGURATIONS,
    final_lock_path,
)
from football_hgt_targets_v4.partial_sharing_study import training_dir as f80_training_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=TRAINED_CONFIGURATIONS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.evaluate_test:
        if args.checkpoint_path is None or not args.full_test:
            parser.error("test evaluation requires --checkpoint-path and --full-test")
        lock_path = final_lock_path()
        if not lock_path.exists():
            parser.error("test evaluation is forbidden before final context lock")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["selected_configuration"] != args.configuration:
            parser.error("only the validation-locked configuration may access test")
        expected = Path(lock["checkpoints"][str(args.seed)]).resolve()
        if args.checkpoint_path.resolve() != expected:
            parser.error("checkpoint does not match the validation lock")

    config = FixedBudgetConfig(
        configuration=args.configuration,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        training_budget=1 if args.smoke else 24,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_plan_path=None if args.full_test else SAMPLE_PLAN,
        max_train_samples=128 if args.smoke else None,
        max_validation_samples=64 if args.smoke else None,
        evaluate_test=args.evaluate_test,
        full_test=args.full_test,
        secondary_guard_reference_path=f80_training_dir(args.seed) / "result.json",
    )
    result = (
        evaluate_fixed_budget_checkpoint(config, args.checkpoint_path)
        if args.evaluate_test
        else run_fixed_budget_training(config)
    )
    print(json.dumps({
        "configuration": args.configuration,
        "best_epoch": result["best_epoch"],
        "validation": result.get("validation"),
        "test": result.get("test"),
        "test_accessed": result["test_accessed"],
    }, indent=2))


if __name__ == "__main__":
    main()
