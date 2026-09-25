#!/usr/bin/env python
"""Train, test, or summarize one Player-history Stage A configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.player_history_reporting import (  # noqa: E402
    build_player_history_report,
    lock_player_history_decision,
)
from football_hgt_targets_v4.player_history_study import CONDITIONS  # noqa: E402
from football_hgt_targets_v4.player_history_training import (  # noqa: E402
    PlayerHistoryTrainingConfig,
    evaluate_player_history_checkpoint,
    run_player_history_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("train", "test", "lock", "report"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "lock":
        print(lock_player_history_decision()); return
    if args.action == "report":
        print(build_player_history_report()); return
    if args.condition is None or args.seed is None or args.output_dir is None:
        parser.error("train/test requires --condition, --seed, and --output-dir")
    config = PlayerHistoryTrainingConfig(
        condition=args.condition,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        max_epochs=1 if args.smoke else 24,
        batch_size=16 if args.smoke else args.batch_size,
        num_workers=0 if args.smoke else args.num_workers,
        max_train_samples=32 if args.smoke else None,
        max_validation_samples=32 if args.smoke else None,
        evaluate_test=args.action == "test",
        full_test=args.action == "test",
    )
    if args.action == "train":
        result = run_player_history_training(config)
    else:
        if args.checkpoint is None:
            parser.error("test requires --checkpoint")
        result = evaluate_player_history_checkpoint(config, args.checkpoint)
    print({"output": str(args.output_dir), "best_epoch": result["best_epoch"], "test_accessed": result["test_accessed"]})


if __name__ == "__main__":
    main()

