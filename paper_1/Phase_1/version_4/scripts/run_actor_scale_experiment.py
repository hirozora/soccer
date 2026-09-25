#!/usr/bin/env python
"""Train or evaluate one Team/Player context-scale probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.actor_study import ACTOR_TASKS, ACTOR_VIEWS  # noqa: E402
from football_hgt_targets_v4.actor_training import (  # noqa: E402
    ActorTrainingConfig,
    evaluate_actor_checkpoint,
    run_actor_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=ACTOR_TASKS, required=True)
    parser.add_argument("--view", choices=ACTOR_VIEWS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=9e-4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path)
    args = parser.parse_args()
    if args.full_test and not args.evaluate_test:
        parser.error("--full-test requires --evaluate-test")
    if args.evaluate_only and (not args.evaluate_test or args.checkpoint_path is None):
        parser.error("--evaluate-only requires --evaluate-test and --checkpoint-path")
    config = ActorTrainingConfig(
        task=args.task,
        view=args.view,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        max_epochs=1 if args.smoke else args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=256 if args.smoke else None,
        max_validation_samples=128 if args.smoke else None,
        max_test_samples=128 if args.smoke and args.evaluate_test else None,
        evaluate_test=args.evaluate_test,
        full_test=args.full_test,
    )
    result = (
        evaluate_actor_checkpoint(config, args.checkpoint_path)
        if args.evaluate_only
        else run_actor_training(config)
    )
    print(json.dumps({
        "task": args.task,
        "view": args.view,
        "best_epoch": result["best_epoch"],
        "validation": result.get("validation"),
        "test": result.get("test"),
        "test_accessed": result["test_accessed"],
    }, indent=2))


if __name__ == "__main__":
    main()

