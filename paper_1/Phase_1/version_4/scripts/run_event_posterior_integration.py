#!/usr/bin/env python
"""Run the registered Event posterior residual integration experiment."""

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
from football_hgt_targets_v4.event_posterior_integration import ROOT, MODES, evaluate_test, run_dir, select, train_seed, write_json
from football_hgt_targets_v4.event_posterior_online import verify_online
from football_hgt_targets_v4.event_posterior_reporting import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "smoke", "train", "select", "test", "report", "pipeline"), required=True)
    parser.add_argument("--seed", type=int, choices=CONFIRMATION_SEEDS)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.resume_from and (args.stage != "train" or args.seed is None or args.mode is None):
        parser.error("--resume-from requires --stage train --seed --mode")
    if args.stage == "pipeline" and (args.seed is not None or args.mode is not None):
        parser.error("Pipeline requires all preregistered seeds and modes")
    torch.set_num_threads(1)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")
    seeds = (args.seed,) if args.seed is not None else CONFIRMATION_SEEDS
    locked = (args.root / "selection/event_posterior_lock.json").exists()
    def status(stage):
        write_json(args.root / "status.json", {"stage": stage, "pid": os.getpid()})
    if args.stage == "verify" or (args.stage == "pipeline" and not locked):
        status("verify")
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path[:2]), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_event_posterior_integration.py"], cwd=VERSION_ROOT, env=env, check=True)
        for seed in seeds:
            verify_online(seed, args.root, device=args.device)
            print(f"verify seed={seed}: passed", flush=True)
    if args.stage == "smoke" or (args.stage == "pipeline" and not locked):
        status("smoke")
        train_seed(seeds[0], args.root, args.device, mode=args.mode, smoke=True)
    if args.stage == "train" or (args.stage == "pipeline" and not locked):
        status("train")
        for seed in seeds:
            train_seed(seed, args.root, args.device, args.mode, args.resume_from)
            for mode in ((args.mode,) if args.mode else MODES):
                verify_online(seed, args.root, run_dir(seed, mode, args.root) / "best_event.pt", args.device)
    if args.stage in ("select", "pipeline"):
        status("select")
        lock = select(args.root)
        print(json.dumps(lock["comparisons"], indent=2), flush=True)
    if args.stage in ("test", "pipeline"):
        status("test_gate")
        lock = json.loads((args.root / "selection/event_posterior_lock.json").read_text())
        if lock["passed"]:
            for seed in seeds:
                evaluate_test(seed, args.root, args.device)
                print(f"test seed={seed}: complete", flush=True)
        else:
            print("Validation failed: retaining Original, no test files read.", flush=True)
    if args.stage in ("report", "pipeline"):
        status("report")
        report(args.root)
        status("completed")
        print(args.root / "report/README.md", flush=True)


if __name__ == "__main__":
    main()
