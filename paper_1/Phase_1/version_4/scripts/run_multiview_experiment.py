#!/usr/bin/env python
"""Train or evaluate one shared-HGT task-view fusion configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.constants import (  # noqa: E402
    FEASIBILITY_ARTIFACT,
    SAMPLE_PLAN,
)
from football_hgt_targets_v4.model import MULTIVIEW_MODES  # noqa: E402
from football_hgt_targets_v4.training import (  # noqa: E402
    TargetTrainingConfig,
    evaluate_checkpoint,
    run_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MULTIVIEW_MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=9e-4)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path)
    args = parser.parse_args()
    if args.full_test and not args.evaluate_test:
        parser.error("--full-test requires --evaluate-test")
    if args.evaluate_only and (args.checkpoint_path is None or not args.evaluate_test):
        parser.error("--evaluate-only requires --checkpoint-path and --evaluate-test")
    if not args.evaluate_only and args.checkpoint_path is not None:
        parser.error("--checkpoint-path is only valid with --evaluate-only")

    config = TargetTrainingConfig(
        task="joint",
        method=args.mode,
        joint_methods={"event": "ce", "time": "current_huber", "position": "xy"},
        joint_loss_weights={"event": 0.2, "time": 1.0, "position": 1.0},
        artifact_path=FEASIBILITY_ARTIFACT,
        sample_plan_path=None if args.full_test else SAMPLE_PLAN,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_epochs=1 if args.smoke else args.epochs,
        patience=args.patience,
        effective_batch_size=args.batch_size,
        micro_batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=256 if args.smoke else None,
        max_validation_samples=128 if args.smoke else None,
        max_test_samples=128 if args.smoke and args.evaluate_test else None,
        evaluate_test=args.evaluate_test,
        full_test=args.full_test,
        graph_variant="semantic_v3_possession",
        possession_topology="membership",
        possession_feature_level="dynamic",
        snapshot_scope="selected_events",
        fusion_mode=args.mode,
    )
    result = (
        evaluate_checkpoint(config, args.checkpoint_path)
        if args.evaluate_only
        else run_training(config)
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output_dir": str(args.output_dir),
                "best_epoch": result["best_epoch"],
                "validation": result.get("validation"),
                "test": result.get("test"),
                "fusion": result.get("fusion"),
                "test_accessed": result["test_accessed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
