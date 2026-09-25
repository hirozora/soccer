#!/usr/bin/env python
"""Execute the registered Position-only, same-architecture confirmation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(VERSION_ROOT / "src"), str(VERSION_ROOT.parent / "benchmark_unified_v1/src")]

import torch
from football_hgt_targets_v4.constants import CONFIRMATION_SEEDS
from football_hgt_targets_v4.position_head_refit import (
    ROOT, evaluate_test, load_cache, report, run_dir, select, train_seed, write_json,
)
from football_hgt_targets_v4.position_refit_verification import verify_online
from football_hgt_targets_v4.position_refit_figures import finalize_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "smoke", "train", "select", "test", "report", "pipeline"), required=True)
    parser.add_argument("--seed", type=int, choices=CONFIRMATION_SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.resume_from and (args.stage != "train" or args.seed is None):
        parser.error("--resume-from requires --stage train and --seed")
    torch.set_num_threads(1)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; select --device cpu")
    seeds = (args.seed,) if args.seed is not None else CONFIRMATION_SEEDS

    def verify() -> None:
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path[:2]), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_position_head_refit.py"],
                       cwd=VERSION_ROOT, env=env, check=True)
        for seed in seeds:
            _, train_source = load_cache(seed, "train", args.root)
            write_json(args.root / "verification" / f"train_seed{seed}.json", train_source)
            verify_online(seed, args.root, device=args.device)
            print(f"verify seed={seed}: passed", flush=True)

    def train() -> None:
        for seed in seeds:
            output = run_dir(seed, args.root)
            completed = False
            if (output / "result.json").exists() and not args.resume_from:
                result = json.loads((output / "result.json").read_text())
                completed = result["completed_epoch"] == 30 or result["stopped_early"]
            if not completed:
                resume = args.resume_from or (output / "last.pt" if (output / "last.pt").exists() else None)
                result = train_seed(seed, args.root, args.device, resume)
            verify_online(seed, args.root, output / "best_position.pt", args.device)
            print(f"train seed={seed}: best={result['best_epoch']}, epochs={result['completed_epoch']}", flush=True)

    locked = (args.root / "selection/position_refit_lock.json").exists()
    if args.stage == "verify" or (args.stage == "pipeline" and not locked):
        verify()
    if args.stage == "smoke" or (args.stage == "pipeline" and not locked):
        train_seed(seeds[0], args.root, args.device, smoke=True)
        print("one-epoch smoke: passed", flush=True)
    if args.stage == "train" or (args.stage == "pipeline" and not locked):
        train()
    if args.stage in ("select", "pipeline"):
        print(select(args.root), flush=True)
    if args.stage in ("test", "pipeline"):
        lock = json.loads((args.root / "selection/position_refit_lock.json").read_text())
        if lock["passed"]:
            for seed in seeds:
                evaluate_test(seed, args.root, args.device)
                verify_online(seed, args.root, run_dir(seed, args.root) / "best_position.pt", args.device)
                print(f"test seed={seed}: complete", flush=True)
        else:
            print("Validation did not pass: test skipped without loading test artifacts.", flush=True)
    if args.stage in ("report", "pipeline"):
        print(report(args.root), flush=True)
        finalize_artifacts(args.root)


if __name__ == "__main__":
    main()
