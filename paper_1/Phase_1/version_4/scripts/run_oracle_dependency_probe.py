#!/usr/bin/env python
"""Run one Oracle dependency cache, head, lock, or report action."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.oracle_dependency import (  # noqa: E402
    OracleProbeConfig,
    build_oracle_cache,
    evaluate_test_oracle_probe,
    train_oracle_probe,
)
from football_hgt_targets_v4.oracle_dependency_reporting import (  # noqa: E402
    build_final_report,
    lock_validation_decision,
)
from football_hgt_targets_v4.oracle_dependency_study import (  # noqa: E402
    CONDITIONS,
    FAMILIES,
    ORACLE_DEPENDENCY_ROOT,
    training_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("cache", "train", "test", "lock", "report"), required=True)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--condition", choices=CONDITIONS)
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
        print(build_oracle_cache(args.seed, args.split, args.device)); return
    if args.action == "lock": print(lock_validation_decision()); return
    if args.action == "report": print(build_final_report()); return
    if args.family is None or args.condition is None:
        parser.error("train/test requires --family and --condition")
    default = ORACLE_DEPENDENCY_ROOT / ("smoke" if args.smoke else args.action) / args.family / args.condition / f"seed{args.seed}"
    config = OracleProbeConfig(
        family=args.family, condition=args.condition, seed=args.seed,
        output_dir=args.output_dir or default, device=args.device,
        max_epochs=1 if args.smoke else args.epochs,
        patience=args.patience, batch_size=args.batch_size,
        max_train_samples=2048 if args.smoke else None,
        max_validation_samples=512 if args.smoke else None,
    )
    if args.action == "train":
        result = train_oracle_probe(config)
    else:
        checkpoint = args.checkpoint or training_dir(args.family, args.condition, args.seed) / "best.pt"
        result = evaluate_test_oracle_probe(config, checkpoint)
    print({"output": str(config.output_dir), "best_epoch": result["best_epoch"], "test_accessed": result["test_accessed"]})


if __name__ == "__main__":
    main()
