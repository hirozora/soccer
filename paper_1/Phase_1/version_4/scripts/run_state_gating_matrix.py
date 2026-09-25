#!/usr/bin/env python
"""Resumable state-aware relation-gating pipeline."""

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
from football_hgt_targets_v4.state_gating_reporting import (  # noqa: E402
    build_state_gating_report,
    lock_state_gating_method,
    selected_model,
)
from football_hgt_targets_v4.state_gating_study import (  # noqa: E402
    STATE_GATING_ROOT,
    test_dir,
    training_dir,
)

RUNNER = ROOT / "scripts/run_state_gating_experiment.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    result: Path
    log: Path


def _jobs(stage: str, devices: list[str], *, smoke: bool = False, test: bool = False) -> list[Job]:
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    jobs = []
    for index, seed in enumerate(seeds):
        if smoke:
            output = STATE_GATING_ROOT / "smoke" / f"seed{seed}"
        elif test:
            output = test_dir(seed)
        else:
            output = training_dir(seed)
        command = [
            sys.executable, str(RUNNER), "--output-dir", str(output),
            "--seed", str(seed), "--device", f"cuda:{devices[index % len(devices)]}",
            "--batch-size", "256", "--num-workers", "2",
        ]
        if smoke:
            command.append("--smoke")
        if test:
            command.extend((
                "--evaluate-test", "--checkpoint",
                str(training_dir(seed) / "best_guarded_core.pt"),
            ))
        jobs.append(Job(
            key=f"{stage}/g1/seed{seed}",
            command=tuple(command),
            result=output / "result.json",
            log=output / f"console_{stage}.log",
        ))
    return jobs


def execute(jobs: list[Job], workers: int, force: bool) -> None:
    status_path = STATE_GATING_ROOT / "background/status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    states = json.loads(status_path.read_text()) if status_path.exists() else {}
    states = {key: ("interrupted" if value == "running" else value) for key, value in states.items()}
    lock = threading.Lock()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src"),
        environment.get("PYTHONPATH", ""),
    ))
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"

    def update(key: str, status: str) -> None:
        with lock:
            states[key] = status
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
            completed = subprocess.run(
                job.command, cwd=ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
        update(job.key, "completed" if completed.returncode == 0 else f"failed:{completed.returncode}")
        if completed.returncode:
            raise RuntimeError(f"Failed {job.key}; inspect {job.log}")

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for future in [pool.submit(run, job) for job in jobs]:
            future.result()


def verify() -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_state_gating.py", "tests/test_partial_l2.py"],
        cwd=ROOT, check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("verify", "smoke", "validation", "lock", "test", "summarize", "pipeline"),
        required=True,
    )
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "verify":
        verify(); return
    if args.stage == "smoke":
        execute(_jobs("smoke", args.devices, smoke=True), 1, args.force); return
    if args.stage == "validation":
        execute(_jobs("validation", args.devices), 3, args.force); return
    if args.stage == "lock":
        print(lock_state_gating_method()); return
    if args.stage == "test":
        lock_state_gating_method()
        if selected_model() == "G1":
            execute(_jobs("test", args.devices, test=True), 3, args.force)
        return
    if args.stage == "summarize":
        print(build_state_gating_report()); return

    verify()
    execute(_jobs("smoke", args.devices, smoke=True), 1, args.force)
    execute(_jobs("validation", args.devices), 3, args.force)
    lock_state_gating_method()
    if selected_model() == "G1":
        execute(_jobs("test", args.devices, test=True), 3, args.force)
    print(build_state_gating_report())


if __name__ == "__main__":
    main()

