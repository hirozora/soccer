#!/usr/bin/env python
"""Resumable multi-GPU task-view fusion experiment pipeline."""

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

from football_hgt_targets_v4.constants import CONFIRMATION_SEEDS  # noqa: E402
from football_hgt_targets_v4.multiview_reporting import build_multiview_report  # noqa: E402
from football_hgt_targets_v4.multiview_study import (  # noqa: E402
    EXPERIMENT_MODES,
    MULTIVIEW_EXPERIMENT_ROOT,
    test_dir,
    validation_dir,
)
from football_hgt_targets_v4.subgraph_study import test_dir as subgraph_test_dir  # noqa: E402

RUNNER = ROOT / "scripts/run_multiview_experiment.py"
VERIFY = ROOT / "scripts/verify_multiview_f80.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    result: Path
    log: Path


def _command(
    mode: str,
    seed: int,
    device: str,
    output: Path,
    *,
    smoke: bool = False,
    checkpoint: Path | None = None,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(RUNNER),
        "--mode", mode,
        "--output-dir", str(output),
        "--learning-rate", "0.0009",
        "--seed", str(seed),
        "--device", f"cuda:{device}",
        "--epochs", "8",
        "--patience", "2",
        "--batch-size", "256",
        "--num-workers", "2",
    ]
    if smoke:
        command.append("--smoke")
    if checkpoint is not None:
        command.extend(
            (
                "--evaluate-only",
                "--evaluate-test",
                "--full-test",
                "--checkpoint-path", str(checkpoint),
            )
        )
    return tuple(command)


def _jobs(
    stage: str,
    devices: list[str],
    *,
    smoke: bool = False,
    test: bool = False,
) -> list[Job]:
    jobs = []
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    for mode in EXPERIMENT_MODES:
        for seed in seeds:
            if smoke:
                output = MULTIVIEW_EXPERIMENT_ROOT / "smoke" / mode / f"seed{seed}"
                checkpoint = None
            elif test:
                output = test_dir(mode, seed)
                checkpoint = validation_dir(mode, seed) / "best.pt"
            else:
                output = validation_dir(mode, seed)
                checkpoint = None
            jobs.append(
                Job(
                    f"{stage}/{mode}/seed{seed}",
                    _command(
                        mode,
                        seed,
                        devices[len(jobs) % len(devices)],
                        output,
                        smoke=smoke,
                        checkpoint=checkpoint,
                    ),
                    output / "result.json",
                    output / "console.log",
                )
            )
    return jobs


def execute(jobs: list[Job], workers: int, force: bool, status_path: Path) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    states = json.loads(status_path.read_text()) if status_path.exists() else {}
    states = {
        key: ("interrupted" if value == "running" else value)
        for key, value in states.items()
    }
    lock = threading.Lock()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "src"),
            str(ROOT.parent / "benchmark_unified_v1/src"),
            environment.get("PYTHONPATH", ""),
        ]
    )
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"

    def update(key: str, value: str) -> None:
        with lock:
            states[key] = value
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(states, indent=2), encoding="utf-8")
            temporary.replace(status_path)

    for job in jobs:
        if job.result.exists() and not force:
            update(job.key, "skipped")
        else:
            update(job.key, "queued")

    def run(job: Job) -> None:
        if job.result.exists() and not force:
            return
        job.log.parent.mkdir(parents=True, exist_ok=True)
        update(job.key, "running")
        with job.log.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                job.command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        status = "completed" if completed.returncode == 0 else f"failed:{completed.returncode}"
        update(job.key, status)
        if completed.returncode:
            raise RuntimeError(f"Failed {job.key}: {job.log}")

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for future in [pool.submit(run, job) for job in jobs]:
            future.result()


def _lock_configurations() -> Path:
    missing = [
        str(validation_dir(mode, seed) / "result.json")
        for mode in EXPERIMENT_MODES
        for seed in CONFIRMATION_SEEDS
        if not (validation_dir(mode, seed) / "result.json").exists()
    ]
    if missing:
        raise RuntimeError(f"Cannot lock incomplete validation matrix: {missing}")
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/football_hgt_targets_v4").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    state = {
        "modes": list(EXPERIMENT_MODES),
        "seeds": list(CONFIRMATION_SEEDS),
        "event_loss_weight": 0.2,
        "time_method": "current_huber",
        "position_method": "xy",
        "source_sha256": digest.hexdigest(),
        "test_accessed": False,
    }
    output = MULTIVIEW_EXPERIMENT_ROOT / "selection/configurations_locked.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return output


def _verify_f80_test() -> None:
    for seed in CONFIRMATION_SEEDS:
        root = subgraph_test_dir("f80", seed)
        if not (root / "result.json").exists() or not (root / "test_predictions.parquet").exists():
            raise RuntimeError(f"Missing reusable F80 test result: {root}")


def _smoke_slots(max_slots: int) -> int:
    peaks = []
    for mode in EXPERIMENT_MODES:
        result = MULTIVIEW_EXPERIMENT_ROOT / "smoke" / mode / f"seed{CONFIRMATION_SEEDS[0]}" / "result.json"
        values = json.loads(result.read_text())
        peaks.append(int(values["peak_cuda_memory_bytes"]))
    return 1 if max(peaks) > 10 * 1024**3 else max_slots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("verify", "smoke", "validation", "test", "summarize", "pipeline"),
        required=True,
    )
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    status = MULTIVIEW_EXPERIMENT_ROOT / "background/status.json"

    if args.stage == "verify":
        subprocess.run([sys.executable, str(VERIFY)], cwd=ROOT, check=True)
        return
    if args.stage == "smoke":
        execute(_jobs("smoke", args.devices, smoke=True), len(args.devices), args.force, status)
        return
    if args.stage == "validation":
        execute(
            _jobs("validation", args.devices),
            len(args.devices) * args.slots_per_gpu,
            args.force,
            status,
        )
        _lock_configurations()
        return
    if args.stage == "test":
        _lock_configurations()
        _verify_f80_test()
        execute(
            _jobs("test", args.devices, test=True),
            len(args.devices) * args.slots_per_gpu,
            args.force,
            status,
        )
        return
    if args.stage == "summarize":
        print(build_multiview_report())
        return

    subprocess.run([sys.executable, str(VERIFY)], cwd=ROOT, check=True)
    execute(_jobs("smoke", args.devices, smoke=True), len(args.devices), args.force, status)
    slots = _smoke_slots(args.slots_per_gpu)
    execute(_jobs("validation", args.devices), len(args.devices) * slots, args.force, status)
    _lock_configurations()
    _verify_f80_test()
    execute(_jobs("test", args.devices, test=True), len(args.devices) * slots, args.force, status)
    print(build_multiview_report())


if __name__ == "__main__":
    main()

