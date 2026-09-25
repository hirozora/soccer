#!/usr/bin/env python
"""Independent two-stage coverage/history study with validation-only locks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

VERSION = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(VERSION / "src"), str(VERSION.parent / "benchmark_unified_v1/src")]
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False
torch.use_deterministic_algorithms(True)

from football_hgt_targets_v4.coverage_history_training import (
    ROOT, SEEDS, reference_evaluation, train_coverage, refit_coverage, select_coverage)
from football_hgt_targets_v4.coverage_history_probe import (
    MODES, cache_history, train_history, select_history)
from football_hgt_targets_v4.coverage_history_verification import audit, smoke, verify_history_online
from football_hgt_targets_v4.coverage_history_reporting import test_locked, report
from football_hgt_targets_v4.spatiotemporal_training import write_json


def verify(root, device):
    output = root / "verification"; output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path[:2])}
    with (output / "unit_tests.log").open("w") as handle:
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_coverage_history.py", "-q"],
                                cwd=VERSION, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Unit tests failed: {output / 'unit_tests.log'}")
    audit(root)
    smoke(root, device)
    write_json(output / "passed.json", {"passed": True, "cuda_device": device})


def matrix(root, job, tasks, devices, cpu=False):
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU matrix not launched")
    directory = root / "background"; directory.mkdir(parents=True, exist_ok=True)
    capacity = [f"cpu:{i}" for i in range(3)] if cpu else [f"cuda:{i}" for i in devices[:4]]
    if not capacity:
        raise ValueError("No devices provided")
    pending, active, attempts, states = list(tasks), [], {}, {}
    try:
        while pending or active:
            free = [slot for slot in capacity if slot not in [j["slot"] for j in active]]
            while pending and free:
                mode, seed = pending.pop(0)
                key = f"{mode}-{seed}"
                attempts[key] = attempts.get(key, 0) + 1
                slot = free.pop(0)
                device = "cpu" if cpu else slot
                stage = "coverage" if job in ("reference", "train", "refit") else "history"
                command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
                    "--job", job, "--mode", mode, "--seed", str(seed), "--device", device, "--root", str(root)]
                path = directory / f"{job}-{key}.log"
                handle = path.open("a")
                process = subprocess.Popen(command, cwd=VERSION, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT)
                active.append({"process": process, "handle": handle, "slot": slot, "key": key, "mode": mode, "seed": seed})
                states[key] = {"pid": process.pid, "status": "running", "attempt": attempts[key],
                               "device": device, "log": str(path), "command": command}
            for item in list(active):
                code = item["process"].poll()
                if code is None:
                    continue
                item["handle"].close()
                active.remove(item)
                states[item["key"]].update(status="complete" if code == 0 else "failed", exit_code=code)
                if code and attempts[item["key"]] < 2:
                    pending.append((item["mode"], item["seed"]))
            write_json(directory / f"{job}_status.json", states)
            if active:
                time.sleep(10)
        if any(state["status"] != "complete" for state in states.values()):
            raise RuntimeError(f"{job} failed; downstream stages remain gated")
    finally:
        for item in active:
            item["process"].terminate()
        for item in active:
            item["process"].wait()
            item["handle"].close()


def coverage(root, devices):
    matrix(root, "reference", [("original", s) for s in SEEDS], devices)
    matrix(root, "train", [(m, s) for m in ("fixed", "rotate") for s in SEEDS], devices)
    matrix(root, "refit", [(m, s) for m in ("original", "fixed", "rotate") for s in SEEDS], devices)
    select_coverage(root)


def history(root, devices):
    if not (root / "selection/coverage_lock.json").exists():
        raise RuntimeError("Stage 1 validation selection required")
    if (root / "selection/history_lock.json").exists():
        return
    matrix(root, "cache", [("selected", s) for s in SEEDS], devices)
    matrix(root, "head", [(m, s) for m in MODES for s in SEEDS], devices, cpu=True)
    matrix(root, "online", [(m, s) for m in MODES for s in SEEDS], devices)
    select_history(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "coverage", "history", "select", "test", "report", "pipeline"), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--mode", choices=("original", "fixed", "rotate", "selected", *MODES))
    parser.add_argument("--job", choices=("reference", "train", "refit", "cache", "head", "online"))
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.job:
        if args.seed is None or args.mode is None:
            parser.error("Worker jobs require --seed and --mode")
        if args.job == "reference": reference_evaluation(args.seed, args.device, root)
        elif args.job == "train": train_coverage(args.mode, args.seed, args.device, root, resume_from=args.resume_from)
        elif args.job == "refit": refit_coverage(args.mode, args.seed, args.device, root)
        elif args.job == "cache":
            for split in ("train", "validation"): cache_history(args.seed, split, args.device, root)
        elif args.job == "head": train_history(args.mode, args.seed, root, args.device, args.resume_from)
        elif args.job == "online": verify_history_online(args.seed, args.mode, args.device, root)
        return
    if args.resume_from or args.seed or args.mode:
        parser.error("Seed/mode/resume require an explicit worker --job")
    def status(stage, state="running", **extra):
        write_json(root / "background/pipeline_status.json", {"pid": os.getpid(), "stage": stage, "status": state, **extra})
    try:
        if args.stage in ("verify", "pipeline"):
            status("verify"); verify(root, args.device)
        if args.stage in ("coverage", "pipeline"):
            if not (root / "verification/passed.json").exists():
                raise RuntimeError("Run verification and CUDA smoke before training")
            if not (root / "selection/coverage_lock.json").exists():
                status("coverage"); coverage(root, args.devices)
        if args.stage in ("history", "pipeline"):
            status("history"); history(root, args.devices)
        if args.stage == "select":
            select_coverage(root); select_history(root)
        if args.stage in ("test", "pipeline"):
            status("test"); test_locked(root, args.device)
        if args.stage in ("report", "pipeline"):
            status("report"); report(root)
        status(args.stage, "completed")
    except BaseException as error:
        status(args.stage, "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
