#!/usr/bin/env python
"""Resumable multi-GPU Oracle dependency probe pipeline."""

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
from football_hgt_targets_v4.oracle_dependency_study import (  # noqa: E402
    CONDITIONS, FAMILIES, ORACLE_DEPENDENCY_ROOT, cache_path,
    training_dir, test_dir, validation_lock_path,
)

RUNNER = ROOT / "scripts/run_oracle_dependency_probe.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    result: Path
    log: Path


def cache_jobs(split: str, devices: list[str]) -> list[Job]:
    jobs = []
    for index, seed in enumerate(CONFIRMATION_SEEDS):
        result = cache_path(seed, split)
        jobs.append(Job(
            f"cache/{split}/seed{seed}",
            (sys.executable, str(RUNNER), "--action", "cache", "--split", split,
             "--seed", str(seed), "--device", f"cuda:{devices[index % len(devices)]}"),
            result, result.with_suffix(".log"),
        ))
    return jobs


def probe_jobs(stage: str, devices: list[str], smoke: bool = False) -> list[Job]:
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    jobs = []
    for index, (family, condition, seed) in enumerate(
        (f, c, s) for f in FAMILIES for c in CONDITIONS for s in seeds
    ):
        output = ORACLE_DEPENDENCY_ROOT / "smoke" / family / condition / f"seed{seed}" if smoke else test_dir(family, condition, seed) if stage == "test" else training_dir(family, condition, seed)
        command = [
            sys.executable, str(RUNNER), "--action", "test" if stage == "test" else "train",
            "--family", family, "--condition", condition, "--seed", str(seed),
            "--output-dir", str(output), "--device", f"cuda:{devices[index % len(devices)]}",
            "--batch-size", "1024",
        ]
        if smoke: command.append("--smoke")
        if stage == "test": command.extend(("--checkpoint", str(training_dir(family, condition, seed) / "best.pt")))
        jobs.append(Job(f"{stage}/{family}/{condition}/seed{seed}", tuple(command), output / "result.json", output / f"console_{stage}.log"))
    return jobs


def execute(jobs: list[Job], workers: int, force: bool) -> None:
    status_path = ORACLE_DEPENDENCY_ROOT / "background/status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    states = json.loads(status_path.read_text()) if status_path.exists() else {}
    states = {key: "interrupted" if value == "running" else value for key, value in states.items()}
    lock = threading.Lock()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src"), environment.get("PYTHONPATH", "")))
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def update(key: str, value: str) -> None:
        with lock:
            states[key] = value
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(states, indent=2)); temporary.replace(status_path)

    def run(job: Job) -> None:
        if job.result.exists() and not force: update(job.key, "skipped"); return
        job.log.parent.mkdir(parents=True, exist_ok=True); update(job.key, "running")
        with job.log.open("w") as handle:
            completed = subprocess.run(job.command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        update(job.key, "completed" if completed.returncode == 0 else f"failed:{completed.returncode}")
        if completed.returncode: raise RuntimeError(f"Failed {job.key}: {job.log}")

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for future in [pool.submit(run, job) for job in jobs]: future.result()


def verify() -> None:
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_oracle_dependency_probe.py"], cwd=ROOT, check=True)


def action(name: str) -> None:
    subprocess.run([sys.executable, str(RUNNER), "--action", name], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "cache", "smoke", "validation", "lock", "test", "report", "pipeline"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--slots-per-gpu", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); workers = len(args.devices) * args.slots_per_gpu
    if args.stage == "verify": verify(); return
    if args.stage == "cache":
        if args.split is None: parser.error("cache requires --split")
        execute(cache_jobs(args.split, args.devices), len(args.devices), args.force); return
    if args.stage == "smoke": execute(probe_jobs("validation", args.devices, True), min(3, len(args.devices)), args.force); return
    if args.stage == "validation": execute(probe_jobs("validation", args.devices), workers, args.force); return
    if args.stage == "lock": action("lock"); return
    if args.stage == "test":
        if not validation_lock_path().exists(): raise RuntimeError("Validation lock required")
        execute(probe_jobs("test", args.devices), workers, args.force); return
    if args.stage == "report": action("report"); return
    verify()
    execute(cache_jobs("train", args.devices) + cache_jobs("validation", args.devices), len(args.devices), args.force)
    execute(probe_jobs("validation", args.devices, True), min(3, len(args.devices)), args.force)
    execute(probe_jobs("validation", args.devices), workers, args.force)
    action("lock")
    execute(cache_jobs("test", args.devices), len(args.devices), args.force)
    execute(probe_jobs("test", args.devices), workers, args.force)
    action("report")


if __name__ == "__main__":
    main()
