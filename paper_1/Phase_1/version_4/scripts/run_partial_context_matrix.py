#!/usr/bin/env python
"""Resumable multi-GPU Partial-L2 task-context pipeline."""

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
from football_hgt_targets_v4.partial_context_reporting import (  # noqa: E402
    build_partial_context_report,
    lock_context_from_validation,
)
from football_hgt_targets_v4.partial_context_study import (  # noqa: E402
    PARTIAL_CONTEXT_ROOT,
    TRAINED_CONFIGURATIONS,
    checkpoint_path,
    final_lock_path,
    test_dir,
    training_dir,
)

RUNNER = ROOT / "scripts/run_partial_context_experiment.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    result: Path
    log: Path


def _job(configuration: str, seed: int, device: str, stage: str, *, smoke: bool = False, test: bool = False) -> Job:
    output = (
        PARTIAL_CONTEXT_ROOT / "smoke" / configuration / f"seed{seed}"
        if smoke else test_dir(configuration, seed) if test else training_dir(configuration, seed)
    )
    command = [
        sys.executable, str(RUNNER), "--configuration", configuration,
        "--output-dir", str(output), "--seed", str(seed),
        "--device", f"cuda:{device}", "--batch-size", "256", "--num-workers", "2",
    ]
    if smoke:
        command.append("--smoke")
    if test:
        command.extend(("--evaluate-test", "--full-test", "--checkpoint-path", str(checkpoint_path(configuration, seed))))
    return Job(
        f"{stage}/{configuration}/seed{seed}", tuple(command), output / "result.json",
        output / f"console_{stage}.log",
    )


def _jobs(stage: str, devices: list[str], *, smoke: bool = False, test: bool = False) -> list[Job]:
    if test:
        lock = json.loads(final_lock_path().read_text(encoding="utf-8"))
        configurations = (lock["selected_configuration"],)
        if configurations == ("partial_l2",):
            return []
    else:
        configurations = TRAINED_CONFIGURATIONS
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    return [
        _job(configuration, seed, devices[index % len(devices)], stage, smoke=smoke, test=test)
        for index, (configuration, seed) in enumerate(
            (configuration, seed) for configuration in configurations for seed in seeds
        )
    ]


def execute(jobs: list[Job], workers: int, force: bool, *, allow_guard_failure: bool = False) -> None:
    if not jobs:
        return
    status_path = PARTIAL_CONTEXT_ROOT / "background/status.json"
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
        if completed.returncode and allow_guard_failure:
            history = job.result.parent / "history.json"
            if history.exists():
                rows = json.loads(history.read_text(encoding="utf-8"))
                if len(rows) == 24 and not any(row.get("guarded_core_eligible", False) for row in rows):
                    update(job.key, "ineligible:no_dual_guard_checkpoint")
                    return
        if completed.returncode:
            raise RuntimeError(f"Failed {job.key}: {job.log}")

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for future in [pool.submit(run, job) for job in jobs]:
            future.result()


def verify() -> None:
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_partial_context.py"], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "smoke", "validation", "lock", "test", "summarize", "pipeline"), required=True)
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "verify":
        verify(); return
    if args.stage == "smoke":
        execute(_jobs("smoke", args.devices, smoke=True), 2, args.force); return
    if args.stage == "validation":
        jobs = _jobs("validation", args.devices)
        execute(jobs, min(len(jobs), 2 * len(args.devices)), args.force, allow_guard_failure=True); return
    if args.stage == "lock":
        print(lock_context_from_validation()); return
    if args.stage == "test":
        if not final_lock_path().exists():
            raise RuntimeError("Run the validation lock before test")
        execute(_jobs("test", args.devices, test=True), min(3, len(args.devices)), args.force); return
    if args.stage == "summarize":
        print(build_partial_context_report()); return
    verify()
    execute(_jobs("smoke", args.devices, smoke=True), 2, args.force)
    validation_jobs = _jobs("validation", args.devices)
    execute(validation_jobs, min(len(validation_jobs), 2 * len(args.devices)), args.force, allow_guard_failure=True)
    lock_context_from_validation()
    execute(_jobs("test", args.devices, test=True), min(3, len(args.devices)), args.force)
    print(build_partial_context_report())


if __name__ == "__main__":
    main()
