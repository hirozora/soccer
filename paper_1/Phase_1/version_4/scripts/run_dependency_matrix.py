#!/usr/bin/env python
"""Resumable multi-GPU orchestration for the frozen target-dependency study."""

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
from football_hgt_targets_v4.dependency_study import CONFIGS_BY_FAMILY, DEPENDENCY_ROOT  # noqa: E402

RUNNER = ROOT / "scripts/run_dependency_experiment.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    result: Path
    log: Path


def _base(action: str, device: str) -> list[str]:
    return [sys.executable, str(RUNNER), "--action", action, "--device", device]


def cache_jobs(split: str, devices: list[str]) -> list[Job]:
    jobs = []
    for index, seed in enumerate(CONFIRMATION_SEEDS):
        device = f"cuda:{devices[index % len(devices)]}"
        result = DEPENDENCY_ROOT / "cache" / f"seed{seed}/{split}.pt"
        jobs.append(Job(
            f"cache/{split}/seed{seed}",
            tuple(_base("cache", device) + ["--seed", str(seed), "--split", split]),
            result,
            result.with_suffix(".log"),
        ))
    return jobs


def probe_jobs(stage: str, devices: list[str], smoke: bool = False) -> list[Job]:
    jobs, index = [], 0
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    for family, names in CONFIGS_BY_FAMILY.items():
        for name in names:
            for seed in seeds:
                device = f"cuda:{devices[index % len(devices)]}"; index += 1
                root = DEPENDENCY_ROOT / ("smoke" if smoke else stage) / family / name / f"seed{seed}"
                command = _base("train" if stage == "train" else "test", device) + [
                    "--family", family, "--configuration", name, "--seed", str(seed), "--output-dir", str(root)
                ]
                if smoke:
                    command.append("--smoke")
                if stage == "test":
                    checkpoint = DEPENDENCY_ROOT / "train" / family / name / f"seed{seed}/best.pt"
                    command.extend(["--checkpoint", str(checkpoint)])
                jobs.append(Job(f"{stage}/{family}/{name}/seed{seed}", tuple(command), root / "result.json", root / "console.log"))
    return jobs


def execute(jobs: list[Job], workers: int, force: bool, status_path: Path) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    states: dict[str, str] = {}; lock = threading.Lock()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src"), environment.get("PYTHONPATH", "")])
    environment.setdefault("OMP_NUM_THREADS", "2")

    def status(key: str, value: str) -> None:
        with lock:
            states[key] = value
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(states, indent=2), encoding="utf-8")
            temporary.replace(status_path)

    def run(job: Job) -> None:
        if job.result.exists() and not force:
            status(job.key, "skipped"); return
        job.log.parent.mkdir(parents=True, exist_ok=True)
        status(job.key, "running")
        with job.log.open("w", encoding="utf-8") as log:
            completed = subprocess.run(job.command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        status(job.key, "completed" if completed.returncode == 0 else f"failed:{completed.returncode}")
        if completed.returncode:
            raise RuntimeError(f"Failed {job.key}: {job.log}")

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for future in [pool.submit(run, job) for job in jobs]: future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cache", "smoke", "train", "test", "pipeline"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--slots-per-gpu", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workers = len(args.devices) * args.slots_per_gpu
    status = DEPENDENCY_ROOT / "background/status.json"
    if args.stage == "cache":
        if args.split is None: parser.error("cache requires --split")
        execute(cache_jobs(args.split, args.devices), len(args.devices), args.force, status); return
    if args.stage == "smoke":
        execute(probe_jobs("train", args.devices, True), workers, args.force, status); return
    if args.stage in {"train", "test"}:
        execute(probe_jobs(args.stage, args.devices), workers, args.force, status); return
    execute(cache_jobs("train", args.devices) + cache_jobs("validation", args.devices), len(args.devices), args.force, status)
    execute(probe_jobs("train", args.devices), workers, args.force, status)
    # Test data is touched only after every pre-registered probe has a locked checkpoint.
    execute(cache_jobs("test", args.devices), len(args.devices), args.force, status)
    execute(probe_jobs("test", args.devices), workers, args.force, status)
    subprocess.run([sys.executable, str(RUNNER), "--action", "attribution"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(RUNNER), "--action", "report"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
