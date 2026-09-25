#!/usr/bin/env python
"""Resumable multi-GPU Team/Player context-scale pipeline."""

from __future__ import annotations

import argparse
import hashlib
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

from football_hgt_targets_v4.actor_reporting import build_actor_report, lock_actor_winners  # noqa: E402
from football_hgt_targets_v4.actor_study import (  # noqa: E402
    ACTOR_EXPERIMENT_ROOT, ACTOR_TASKS, ACTOR_VIEWS, test_dir, validation_dir,
)
from football_hgt_targets_v4.constants import CONFIRMATION_SEEDS  # noqa: E402

RUNNER = ROOT / "scripts/run_actor_scale_experiment.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    result: Path
    log: Path


def _command(task: str, view: str, seed: int, device: str, output: Path, *, smoke: bool = False, checkpoint: Path | None = None) -> tuple[str, ...]:
    command = [
        sys.executable, str(RUNNER), "--task", task, "--view", view,
        "--output-dir", str(output), "--seed", str(seed), "--device", f"cuda:{device}",
        "--learning-rate", "0.0009", "--epochs", "8", "--patience", "2",
        "--batch-size", "256", "--num-workers", "3",
    ]
    if smoke:
        command.append("--smoke")
    if checkpoint is not None:
        command.extend(("--evaluate-only", "--evaluate-test", "--full-test", "--checkpoint-path", str(checkpoint)))
    return tuple(command)


def _jobs(stage: str, devices: list[str], *, smoke: bool = False, test: bool = False) -> list[Job]:
    jobs: list[Job] = []
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    for task in ACTOR_TASKS:
        for view in ACTOR_VIEWS:
            for seed in seeds:
                if smoke:
                    output = ACTOR_EXPERIMENT_ROOT / "smoke" / task / view / f"seed{seed}"
                    checkpoint = None
                elif test:
                    output = test_dir(task, view, seed)
                    checkpoint = validation_dir(task, view, seed) / "best.pt"
                else:
                    output = validation_dir(task, view, seed)
                    checkpoint = None
                jobs.append(Job(
                    f"{stage}/{task}/{view}/seed{seed}",
                    _command(task, view, seed, devices[len(jobs) % len(devices)], output, smoke=smoke, checkpoint=checkpoint),
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

    def run(job: Job) -> None:
        if job.result.exists() and not force:
            update(job.key, "skipped")
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


def _lock() -> Path:
    state = lock_actor_winners()
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/football_hgt_targets_v4").glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    state["source_sha256"] = digest.hexdigest()
    output = ACTOR_EXPERIMENT_ROOT / "selection/winners.json"
    output.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "smoke", "validation", "test", "summarize", "pipeline"), required=True)
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workers = len(args.devices) * args.slots_per_gpu
    status = ACTOR_EXPERIMENT_ROOT / "background/status.json"
    if args.stage == "verify":
        subprocess.run([sys.executable, "-m", "pytest", "tests/test_actor_scale.py"], cwd=ROOT, check=True); return
    if args.stage == "smoke":
        execute(_jobs("smoke", args.devices, smoke=True), min(len(args.devices), workers), args.force, status); return
    if args.stage == "validation":
        execute(_jobs("validation", args.devices), workers, args.force, status); _lock(); return
    if args.stage == "test":
        _lock(); execute(_jobs("test", args.devices, test=True), workers, args.force, status); return
    if args.stage == "summarize":
        print(build_actor_report()); return
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_actor_scale.py"], cwd=ROOT, check=True)
    execute(_jobs("smoke", args.devices, smoke=True), min(len(args.devices), workers), args.force, status)
    execute(_jobs("validation", args.devices), workers, args.force, status)
    _lock()
    execute(_jobs("test", args.devices, test=True), workers, args.force, status)
    print(build_actor_report())


if __name__ == "__main__":
    main()

