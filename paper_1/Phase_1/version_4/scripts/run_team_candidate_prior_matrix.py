#!/usr/bin/env python
"""Resumable pipeline for Team-aware Player candidate-prior calibration."""

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
from football_hgt_targets_v4.team_candidate_prior_study import (  # noqa: E402
    TEAM_CANDIDATE_PRIOR_ROOT,
    cache_path,
    lock_path,
)

RUNNER = ROOT / "scripts/run_team_candidate_prior.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    output: Path
    log: Path


def cache_jobs(split: str, devices: list[str]) -> list[Job]:
    result = []
    for index, seed in enumerate(CONFIRMATION_SEEDS):
        output = cache_path(seed, split)
        result.append(Job(
            f"cache/{split}/seed{seed}",
            (
                sys.executable, str(RUNNER), "--action", "cache", "--split", split,
                "--seed", str(seed), "--device", f"cuda:{devices[index % len(devices)]}",
            ),
            output,
            output.parent / f"console_{split}.log",
        ))
    return result


def execute(jobs: list[Job], force: bool) -> None:
    status_path = TEAM_CANDIDATE_PRIOR_ROOT / "background/status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    states = json.loads(status_path.read_text()) if status_path.exists() else {}
    states = {key: "interrupted" if value == "running" else value for key, value in states.items()}
    lock = threading.Lock()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src"),
        environment.get("PYTHONPATH", ""),
    ))
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def update(key: str, value: str) -> None:
        with lock:
            states[key] = value
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(states, indent=2))
            temporary.replace(status_path)

    def run(job: Job) -> None:
        if job.output.exists() and not force:
            update(job.key, "skipped")
            return
        job.log.parent.mkdir(parents=True, exist_ok=True)
        update(job.key, "running")
        with job.log.open("w") as handle:
            completed = subprocess.run(
                job.command, cwd=ROOT, env=environment,
                stdout=handle, stderr=subprocess.STDOUT,
            )
        update(job.key, "completed" if completed.returncode == 0 else f"failed:{completed.returncode}")
        if completed.returncode:
            raise RuntimeError(f"Failed {job.key}: {job.log}")

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for future in [pool.submit(run, job) for job in jobs]:
            future.result()


def action(name: str) -> None:
    subprocess.run([sys.executable, str(RUNNER), "--action", name], cwd=ROOT, check=True)


def verify() -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_team_candidate_prior.py"],
        cwd=ROOT, check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("verify", "validation", "lock", "test", "report", "pipeline"),
        required=True,
    )
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "verify":
        verify(); return
    if args.stage == "validation":
        execute(cache_jobs("validation", args.devices), args.force); return
    if args.stage == "lock":
        action("lock"); return
    if args.stage == "test":
        if not lock_path().exists():
            raise RuntimeError("Validation method lock required")
        execute(cache_jobs("test", args.devices), args.force)
        action("test"); return
    if args.stage == "report":
        action("report"); return
    verify()
    execute(cache_jobs("validation", args.devices), args.force)
    action("lock")
    execute(cache_jobs("test", args.devices), args.force)
    action("test")
    action("report")


if __name__ == "__main__":
    main()

