#!/usr/bin/env python
"""Train one Version 4 target-formulation experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.constants import (  # noqa: E402
    EVENT_METHODS,
    FEASIBILITY_ARTIFACT,
    FULL_ARTIFACT,
    POSITION_METHODS,
    SAMPLE_PLAN,
    TIME_METHODS,
    POSSESSION_FEATURE_LEVELS,
    POSSESSION_TOPOLOGIES,
)
from football_hgt_targets_v4.training import (  # noqa: E402
    TargetTrainingConfig,
    evaluate_checkpoint,
    run_training,
)
from football_hgt_targets_v4.subgraph_views import (  # noqa: E402
    DEFAULT_SELECTOR_SEED,
    resolve_view_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("event", "time", "position", "joint"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--event-method", choices=EVENT_METHODS)
    parser.add_argument("--time-method", choices=TIME_METHODS)
    parser.add_argument("--position-method", choices=POSITION_METHODS)
    parser.add_argument("--event-loss-weight", type=float)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--sample-plan-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=9e-4)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument(
        "--graph-variant",
        choices=("semantic_v2", "semantic_v3_possession"),
        default="semantic_v2",
    )
    parser.add_argument(
        "--possession-topology", choices=POSSESSION_TOPOLOGIES, default="none"
    )
    parser.add_argument(
        "--possession-feature-level",
        choices=POSSESSION_FEATURE_LEVELS,
        default="topology",
    )
    parser.add_argument(
        "--snapshot-scope",
        choices=("selected_events", "anchor_history"),
        default="selected_events",
    )
    parser.add_argument("--context-view", default="f80")
    parser.add_argument("--selector-seed", type=int, default=DEFAULT_SELECTOR_SEED)
    args = parser.parse_args()
    if args.full_test and not args.evaluate_test:
        parser.error("--full-test requires --evaluate-test")
    if args.graph_variant == "semantic_v2" and args.possession_topology != "none":
        parser.error("Possession topology requires --graph-variant semantic_v3_possession")
    if args.possession_topology == "none" and args.possession_feature_level != "topology":
        parser.error("Possession features require an enabled Possession topology")
    try:
        resolve_view_spec(args.context_view)
    except ValueError as exc:
        parser.error(str(exc))
    if args.context_view != "f80" and (
        args.graph_variant != "semantic_v3_possession"
        or args.possession_topology == "none"
    ):
        parser.error("Sparse context views require an enabled Semantic V3 Possession topology")

    allowed = {"event": EVENT_METHODS, "time": TIME_METHODS, "position": POSITION_METHODS}
    if args.task != "joint" and args.method not in allowed[args.task]:
        parser.error(f"Unknown {args.task} method {args.method!r}")
    joint_methods = None
    if args.task == "joint":
        if not (args.event_method and args.time_method and args.position_method):
            parser.error("Joint training requires all three --*-method arguments")
        joint_methods = {
            "event": args.event_method,
            "time": args.time_method,
            "position": args.position_method,
        }
        event_loss_weight = 1.0 if args.event_loss_weight is None else args.event_loss_weight
        if not (event_loss_weight > 0.0) or not math.isfinite(event_loss_weight):
            parser.error("--event-loss-weight must be finite and positive")
        joint_loss_weights = {
            "event": event_loss_weight,
            "time": 1.0,
            "position": 1.0,
        }
    else:
        if args.event_loss_weight is not None:
            parser.error("--event-loss-weight is only valid with --task joint")
        joint_loss_weights = None
    artifact = args.artifact_path or (FULL_ARTIFACT if args.full else FEASIBILITY_ARTIFACT)
    sample_plan = args.sample_plan_path
    if sample_plan is None and not args.full:
        sample_plan = SAMPLE_PLAN
    config = TargetTrainingConfig(
        task=args.task,
        method=args.method,
        joint_methods=joint_methods,
        joint_loss_weights=joint_loss_weights,
        artifact_path=artifact,
        sample_plan_path=sample_plan,
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
        graph_variant=args.graph_variant,
        possession_topology=args.possession_topology,
        possession_feature_level=args.possession_feature_level,
        snapshot_scope=args.snapshot_scope,
        context_view=args.context_view,
        selector_seed=args.selector_seed,
    )
    if args.evaluate_only:
        if args.checkpoint_path is None:
            parser.error("--evaluate-only requires --checkpoint-path")
        if not args.evaluate_test:
            parser.error("Checkpoint evaluation in this pipeline must explicitly target test")
        result = evaluate_checkpoint(config, args.checkpoint_path)
    else:
        if args.checkpoint_path is not None:
            parser.error("--checkpoint-path is only valid with --evaluate-only")
        result = run_training(config)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "task": args.task,
                "method": args.method,
                "best_epoch": result["best_epoch"],
                "validation": result.get("validation"),
                "test": result.get("test"),
                "test_accessed": result["test_accessed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
