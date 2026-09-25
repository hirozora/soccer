#!/usr/bin/env python
"""Resumable multi-GPU five-task context-view pipeline."""

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
from football_hgt_targets_v4.five_task_reporting import build_five_task_report, lock_five_task_configurations  # noqa: E402
from football_hgt_targets_v4.five_task_study import FIVE_TASK_EXPERIMENT_ROOT, FIVE_TASK_MODES, test_dir, validation_dir  # noqa: E402

RUNNER = ROOT / "scripts/run_five_task_experiment.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    result: Path
    log: Path


def _command(mode: str, seed: int, device: str, output: Path, *, smoke: bool = False, checkpoint: Path | None = None) -> tuple[str, ...]:
    command = [
        sys.executable, str(RUNNER), "--mode", mode, "--output-dir", str(output),
        "--seed", str(seed), "--device", f"cuda:{device}", "--learning-rate", "0.0009",
        "--epochs", "8", "--patience", "2", "--batch-size", "256", "--num-workers", "2",
    ]
    if smoke:
        command.append("--smoke")
    if checkpoint is not None:
        command.extend(("--evaluate-only", "--evaluate-test", "--full-test", "--checkpoint-path", str(checkpoint)))
    return tuple(command)


def _jobs(stage: str, devices: list[str], *, smoke: bool = False, test: bool = False) -> list[Job]:
    jobs: list[Job] = []
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    for mode in FIVE_TASK_MODES:
        for seed in seeds:
            if smoke:
                output = FIVE_TASK_EXPERIMENT_ROOT / "smoke" / mode / f"seed{seed}"
                checkpoint = None
            elif test:
                output = test_dir(mode, seed)
                checkpoint = validation_dir(mode, seed) / "best.pt"
            else:
                output = validation_dir(mode, seed)
                checkpoint = None
            jobs.append(Job(
                f"{stage}/{mode}/seed{seed}",
                _command(mode, seed, devices[len(jobs) % len(devices)], output, smoke=smoke, checkpoint=checkpoint),
                output / "result.json", output / "console.log",
            ))
    return jobs


def execute(jobs: list[Job], workers: int, force: bool, status_path: Path) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    states = json.loads(status_path.read_text()) if status_path.exists() else {}
    states = {key: ("interrupted" if value == "running" else value) for key, value in states.items()}
    lock = threading.Lock()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src"), environment.get("PYTHONPATH", "")))
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"

    def update(key: str, value: str) -> None:
        with lock:
            states[key] = value
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(states, indent=2), encoding="utf-8")
            temporary.replace(status_path)

    for job in jobs:
        update(job.key, "skipped" if job.result.exists() and not force else "queued")

    def run(job: Job) -> None:
        if job.result.exists() and not force:
            return
        job.log.parent.mkdir(parents=True, exist_ok=True)
        update(job.key, "running")
        with job.log.open("w", encoding="utf-8") as log:
            completed = subprocess.run(job.command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        update(job.key, "completed" if completed.returncode == 0 else f"failed:{completed.returncode}")
        if completed.returncode:
            raise RuntimeError(f"Failed {job.key}: {job.log}")

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for future in [pool.submit(run, job) for job in jobs]:
            future.result()


def _smoke_slots(requested: int) -> int:
    peaks = []
    for mode in FIVE_TASK_MODES:
        path = FIVE_TASK_EXPERIMENT_ROOT / "smoke" / mode / f"seed{CONFIRMATION_SEEDS[0]}/result.json"
        peaks.append(int(json.loads(path.read_text())["peak_cuda_memory_bytes"]))
    return 1 if max(peaks) > 10 * 1024**3 else requested


def _verify() -> None:
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_five_task_view.py"], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "smoke", "validation", "test", "summarize", "pipeline"), required=True)
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    status = FIVE_TASK_EXPERIMENT_ROOT / "background/status.json"
    if args.stage == "verify":
        _verify(); return
    if args.stage == "smoke":
        execute(_jobs("smoke", args.devices, smoke=True), len(args.devices), args.force, status); return
    if args.stage == "validation":
        execute(_jobs("validation", args.devices), len(args.devices) * args.slots_per_gpu, args.force, status)
        lock_five_task_configurations(); return
    if args.stage == "test":
        lock_five_task_configurations()
        execute(_jobs("test", args.devices, test=True), len(args.devices) * args.slots_per_gpu, args.force, status); return
    if args.stage == "summarize":
        print(build_five_task_report()); return
    _verify()
    execute(_jobs("smoke", args.devices, smoke=True), len(args.devices), args.force, status)
    slots = _smoke_slots(args.slots_per_gpu)
    execute(_jobs("validation", args.devices), len(args.devices) * slots, args.force, status)
    lock_five_task_configurations()
    execute(_jobs("test", args.devices, test=True), len(args.devices) * slots, args.force, status)
    print(build_five_task_report())


if __name__ == "__main__":
    main()
