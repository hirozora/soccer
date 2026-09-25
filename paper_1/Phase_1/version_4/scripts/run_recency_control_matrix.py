#!/usr/bin/env python
"""Run equal-size possession/recency controls with a validation-only lock."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.constants import CONFIRMATION_SEEDS  # noqa: E402
from football_hgt_targets_v4.recency_control_reporting import build_report, validate_and_lock  # noqa: E402
from football_hgt_targets_v4.recency_control_study import NEW_CONFIGS, RECENCY_CONTROL_ROOT, lock_path, test_dir, validation_dir  # noqa: E402

SINGLE_RUNNER = ROOT / "scripts/run_experiment.py"
FUSION_RUNNER = ROOT / "scripts/run_multiview_experiment.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    result: Path
    log: Path


def _command(config: str, seed: int, device: str, output: Path, *, smoke: bool = False, checkpoint: Path | None = None) -> tuple[str, ...]:
    common = ["--output-dir", str(output), "--learning-rate", "0.0009", "--seed", str(seed), "--device", f"cuda:{device}", "--epochs", "8", "--patience", "2", "--batch-size", "256", "--num-workers", "2"]
    if config == "recency_sf_b":
        command = [sys.executable, str(FUSION_RUNNER), "--mode", config, *common]
    else:
        command = [
            sys.executable, str(SINGLE_RUNNER), "--task", "joint", "--method", config,
            "--event-method", "ce", "--time-method", "current_huber", "--position-method", "xy",
            "--event-loss-weight", "0.2", "--graph-variant", "semantic_v3_possession",
            "--possession-topology", "membership", "--possession-feature-level", "dynamic",
            "--snapshot-scope", "selected_events", "--context-view", config, *common,
        ]
    if smoke:
        command.append("--smoke")
    if checkpoint is not None:
        command.extend(("--evaluate-only", "--evaluate-test", "--full-test", "--checkpoint-path", str(checkpoint)))
    return tuple(command)


def _jobs(stage: str, devices: list[str], *, smoke: bool = False, test: bool = False) -> list[Job]:
    jobs = []
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    for config in NEW_CONFIGS:
        for seed in seeds:
            if smoke:
                output = RECENCY_CONTROL_ROOT / "smoke" / config / f"seed{seed}"
                checkpoint = None
            elif test:
                output = test_dir(config, seed)
                checkpoint = validation_dir(config, seed) / "best.pt"
            else:
                output = validation_dir(config, seed)
                checkpoint = None
            jobs.append(Job(
                f"{stage}/{config}/seed{seed}",
                _command(config, seed, devices[len(jobs) % len(devices)], output, smoke=smoke, checkpoint=checkpoint),
                output / "result.json", output / "console.log",
            ))
    return jobs


def execute(jobs: list[Job], workers: int, force: bool) -> None:
    status_path = RECENCY_CONTROL_ROOT / "background/status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    states = json.loads(status_path.read_text()) if status_path.exists() else {}
    states = {key: ("interrupted" if value == "running" else value) for key, value in states.items()}
    lock = threading.Lock()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src"), environment.get("PYTHONPATH", "")))
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def update(key: str, value: str) -> None:
        with lock:
            states[key] = value
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(states, indent=2), encoding="utf-8")
            temporary.replace(status_path)

    def run(job: Job) -> None:
        if job.result.exists() and not force:
            update(job.key, "skipped")
            return
        job.log.parent.mkdir(parents=True, exist_ok=True)
        update(job.key, "running")
        with job.log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(job.command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        update(job.key, "completed" if completed.returncode == 0 else f"failed:{completed.returncode}")
        if completed.returncode:
            raise RuntimeError(f"Failed {job.key}: {job.log}")

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for future in [pool.submit(run, job) for job in jobs]:
            future.result()


def _require_lock() -> None:
    if not lock_path().exists():
        raise RuntimeError("Test access denied until validation_locked.json exists")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "validation", "lock", "test", "summarize", "pipeline"), required=True)
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workers = len(args.devices) * args.slots_per_gpu

    if args.stage == "smoke":
        execute(_jobs("smoke", args.devices, smoke=True), len(args.devices), args.force)
    elif args.stage == "validation":
        execute(_jobs("validation", args.devices), workers, args.force)
    elif args.stage == "lock":
        print(validate_and_lock())
    elif args.stage == "test":
        _require_lock()
        execute(_jobs("test", args.devices, test=True), workers, args.force)
    elif args.stage == "summarize":
        print(build_report())
    else:
        execute(_jobs("smoke", args.devices, smoke=True), len(args.devices), args.force)
        execute(_jobs("validation", args.devices), workers, args.force)
        print(validate_and_lock())
        _require_lock()
        execute(_jobs("test", args.devices, test=True), workers, args.force)
        print(build_report())


if __name__ == "__main__":
    main()
