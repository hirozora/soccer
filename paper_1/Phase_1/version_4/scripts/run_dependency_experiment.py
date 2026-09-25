#!/usr/bin/env python
"""Run one cache, probe, attribution, or reporting action."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.dependency_reporting import build_dependency_report  # noqa: E402
from football_hgt_targets_v4.dependency_study import (  # noqa: E402
    CONFIGS_BY_FAMILY,
    DEPENDENCY_ROOT,
    ProbeConfig,
    build_context_cache,
    evaluate_test_probe,
    train_probe,
)
from football_hgt_targets_v4.position_attribution import run_position_attribution  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("cache", "train", "test", "attribution", "report"), required=True)
    parser.add_argument("--family", choices=("position", "time"))
    parser.add_argument("--configuration")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "cache":
        if args.split is None: parser.error("cache requires --split")
        print(build_context_cache(args.seed, args.split, args.device)); return
    if args.action == "attribution":
        print(run_position_attribution(args.output_dir)); return
    if args.action == "report":
        print(build_dependency_report(args.output_dir)); return
    if args.family is None or args.configuration not in CONFIGS_BY_FAMILY.get(args.family, ()):
        parser.error("train/test requires a valid --family and --configuration")
    default_output = DEPENDENCY_ROOT / ("smoke" if args.smoke else args.action) / args.family / args.configuration / f"seed{args.seed}"
    config = ProbeConfig(
        family=args.family,
        name=args.configuration,
        seed=args.seed,
        output_dir=args.output_dir or default_output,
        device=args.device,
        max_epochs=1 if args.smoke else args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        max_train_samples=2048 if args.smoke else None,
        max_validation_samples=512 if args.smoke else None,
    )
    if args.action == "train":
        result = train_probe(config)
    else:
        if args.checkpoint is None: parser.error("test requires --checkpoint")
        result = evaluate_test_probe(config, args.checkpoint)
    print({"output": str(config.output_dir), "best_epoch": result["best_epoch"], "test_accessed": result["test_accessed"]})


if __name__ == "__main__":
    main()
