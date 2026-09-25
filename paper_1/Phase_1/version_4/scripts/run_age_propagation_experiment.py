#!/usr/bin/env python
"""Train or evaluate one task-conditioned age propagation model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.age_propagation_study import TRAINED, checkpoint_path, lock_path  # noqa: E402
from football_hgt_targets_v4.constants import SAMPLE_PLAN  # noqa: E402
from football_hgt_targets_v4.fixed_budget_training import FixedBudgetConfig, evaluate_fixed_budget_checkpoint, run_fixed_budget_training  # noqa: E402
from football_hgt_targets_v4.partial_sharing_study import training_dir as f80_training_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=TRAINED, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path)
    args = parser.parse_args()
    if args.evaluate_test:
        if args.checkpoint_path is None:
            parser.error("--evaluate-test requires --checkpoint-path")
        if not lock_path().exists():
            parser.error("test evaluation is forbidden before validation lock")
        lock = json.loads(lock_path().read_text())
        if args.configuration not in lock["complete_configurations"]:
            parser.error("configuration was not locked as complete on validation")
        expected = checkpoint_path(args.configuration, args.seed).resolve()
        if args.checkpoint_path.resolve() != expected:
            parser.error("checkpoint does not match validation lock")
    config = FixedBudgetConfig(
        configuration=args.configuration,
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
