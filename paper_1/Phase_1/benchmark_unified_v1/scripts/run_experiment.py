#!/usr/bin/env python
"""Train and evaluate one model/contract/window/seed configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_benchmark.constants import DEFAULT_ARTIFACT_PATH  # noqa: E402
from football_benchmark.protocol import ProtocolArtifacts  # noqa: E402
from football_benchmark.training import (  # noqa: E402
    TrainingConfig,
    evaluate_checkpoint,
    run_training,
)


COMPATIBLE = {
    "seq2event": {"hgt", "seq2event"},
    "unified_lem": {"hgt", "unified_lem"},
    "nmstpp": {"hgt", "nmstpp"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", choices=sorted(COMPATIBLE), required=True)
    parser.add_argument(
        "--model", choices=["hgt", "seq2event", "unified_lem", "nmstpp"], required=True
    )
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--effective-batch-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--sample-plan-path", type=Path)
    parser.add_argument(
        "--unified-event-loss-mode",
        choices=("legacy_balanced", "unweighted", "sqrt_capped"),
        default="legacy_balanced",
    )
    parser.add_argument(
        "--graph-variant",
        choices=("legacy", "semantic_v2"),
        default="legacy",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Evaluate this selected checkpoint without retraining",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Do not read or evaluate the locked test split (required for tuning)",
    )
    args = parser.parse_args()
    if args.model not in COMPATIBLE[args.contract]:
        parser.error(f"{args.model} is not compatible with {args.contract}")
    if args.graph_variant != "legacy" and args.model != "hgt":
        parser.error("--graph-variant is only available for HGT")
    if not args.artifact_path.exists():
        parser.error(f"Missing protocol artifact: {args.artifact_path}; run build_protocol.py")
    if args.sample_plan_path is not None and not args.sample_plan_path.exists():
        parser.error(f"Missing sample plan: {args.sample_plan_path}")
    if args.checkpoint is not None and not args.checkpoint.exists():
        parser.error(f"Missing checkpoint: {args.checkpoint}")
    output_dir = args.output_dir or (
        ROOT
        / "experiments/runs"
        / args.contract
        / args.model
        / f"k{args.window_size}"
        / f"lr{args.learning_rate:g}"
        / f"seed{args.seed}"
    )
    smoke_limit_train = 256 if args.model == "hgt" else 512
    config = TrainingConfig(
        contract=args.contract,
        family=args.model,
        window_size=args.window_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=output_dir,
        device=args.device,
        max_epochs=1 if args.smoke else args.epochs,
        patience=args.patience,
        effective_batch_size=args.effective_batch_size,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        max_train_samples=smoke_limit_train if args.smoke else args.max_train_samples,
        max_validation_samples=(
            128 if args.smoke else args.max_validation_samples
        ),
        max_test_samples=128 if args.smoke else args.max_test_samples,
        evaluate_test=not args.validation_only,
        sample_plan_path=args.sample_plan_path,
        unified_event_loss_mode=args.unified_event_loss_mode,
        graph_variant=args.graph_variant,
    )
    artifacts = ProtocolArtifacts.load(args.artifact_path)
    result = (
        evaluate_checkpoint(config, artifacts, args.checkpoint)
        if args.checkpoint is not None
        else run_training(config, artifacts)
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "best_epoch": result["best_epoch"],
                "validation_loss": result["validation"]["loss"],
                "test_event_macro_f1": (
                    result["test"]["event"]["macro_f1"]
                    if result["test"] is not None
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
