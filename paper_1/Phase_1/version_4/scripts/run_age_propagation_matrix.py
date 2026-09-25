#!/usr/bin/env python
"""Resumable task-conditioned age propagation pipeline."""

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

from football_hgt_targets_v4.age_propagation_efficiency import benchmark_age_propagation  # noqa: E402
from football_hgt_targets_v4.age_propagation_reporting import build_report, lock_validation  # noqa: E402
from football_hgt_targets_v4.age_propagation_study import AGE_PROPAGATION_ROOT, TRAINED, checkpoint_path, lock_path, test_dir, training_dir  # noqa: E402
from football_hgt_targets_v4.constants import CONFIRMATION_SEEDS  # noqa: E402

RUNNER = ROOT / "scripts/run_age_propagation_experiment.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    result: Path
    log: Path


def _jobs(stage: str, devices: list[str], *, smoke: bool = False, test: bool = False) -> list[Job]:
    configurations = TRAINED
    if test:
        lock = json.loads(lock_path().read_text())
        configurations = tuple(name for name in TRAINED if name in lock["complete_configurations"])
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    jobs = []
    for index, (configuration, seed) in enumerate((c, s) for c in configurations for s in seeds):
        output = AGE_PROPAGATION_ROOT / "smoke" / configuration / f"seed{seed}" if smoke else test_dir(configuration, seed) if test else training_dir(configuration, seed)
        command = [
            sys.executable, str(RUNNER),
            "--configuration", configuration,
            "--output-dir", str(output),
            "--seed", str(seed),
            "--device", f"cuda:{devices[index % len(devices)]}",
            "--batch-size", "256",
            "--num-workers", "2",
        ]
        if smoke:
            command.append("--smoke")
        if test:
            command.extend(("--evaluate-test", "--checkpoint-path", str(checkpoint_path(configuration, seed))))
        jobs.append(Job(
            f"{stage}/{configuration}/seed{seed}", tuple(command),
            output / "result.json", output / f"console_{stage}.log",
        ))
    return jobs


def execute(jobs: list[Job], workers: int, force: bool, *, allow_ineligible: bool = False) -> None:
    status_path = AGE_PROPAGATION_ROOT / "background/status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    states = json.loads(status_path.read_text()) if status_path.exists() else {}
    states = {key: ("interrupted" if value == "running" else value) for key, value in states.items()}
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
            temporary.write_text(json.dumps(states, indent=2), encoding="utf-8")
            temporary.replace(status_path)

    def run(job: Job) -> None:
        if job.result.exists() and not force:
            update(job.key, "skipped")
            return
        job.log.parent.mkdir(parents=True, exist_ok=True)
        update(job.key, "running")
        with job.log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                job.command, cwd=ROOT, env=environment,
                stdout=handle, stderr=subprocess.STDOUT,
            )
        if completed.returncode and allow_ineligible:
            history_path = job.result.parent / "history.json"
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            if len(history) == 24 and not any(row.get("guarded_core_eligible") for row in history):
                update(job.key, "ineligible:no_dual_guard_epoch")
                return
        update(job.key, "completed" if completed.returncode == 0 else f"failed:{completed.returncode}")
        if completed.returncode:
            raise RuntimeError(f"Failed {job.key}: {job.log}")

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for future in [pool.submit(run, job) for job in jobs]:
            future.result()


def verify() -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_age_propagation.py"],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(
        "verify", "smoke", "validation", "lock", "test",
        "efficiency", "summarize", "pipeline",
    ), required=True)
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workers = len(args.devices) * args.slots_per_gpu
    if args.stage == "verify": verify(); return
    if args.stage == "smoke": execute(_jobs("smoke", args.devices, smoke=True), min(3, len(args.devices)), args.force); return
    if args.stage == "validation": execute(_jobs("validation", args.devices), workers, args.force, allow_ineligible=True); return
    if args.stage == "lock": print(lock_validation()); return
    if args.stage == "test":
        if not lock_path().exists(): raise RuntimeError("Validation lock required")
        execute(_jobs("test", args.devices, test=True), workers, args.force); return
    if args.stage == "efficiency": print(benchmark_age_propagation(f"cuda:{args.devices[0]}")); return
    if args.stage == "summarize": print(build_report()); return
    verify()
    execute(_jobs("smoke", args.devices, smoke=True), min(3, len(args.devices)), args.force)
    execute(_jobs("validation", args.devices), workers, args.force, allow_ineligible=True)
    lock_validation()
    execute(_jobs("test", args.devices, test=True), workers, args.force)
    benchmark_age_propagation(f"cuda:{args.devices[0]}")
    print(build_report())


if __name__ == "__main__":
    main()
