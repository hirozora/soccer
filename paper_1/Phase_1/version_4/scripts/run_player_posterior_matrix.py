#!/usr/bin/env python
"""Resumable multi-GPU pipeline for Player posterior conditioning Stage B."""

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
from football_hgt_targets_v4.player_posterior_study import (  # noqa: E402
    CONDITIONS,
    FAMILIES,
    PLAYER_POSTERIOR_ROOT,
    condition_cache_path,
    decision_path,
    test_dir,
    training_dir,
)

RUNNER = ROOT / "scripts/run_player_posterior_probe.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    output: Path
    log: Path


def cache_jobs(split: str) -> list[Job]:
    return [
        Job(
            f"cache/{split}/seed{seed}",
            (sys.executable, str(RUNNER), "--action", "cache", "--split", split, "--seed", str(seed)),
            condition_cache_path(seed, split),
            condition_cache_path(seed, split).parent / f"console_{split}.log",
        )
        for seed in CONFIRMATION_SEEDS
    ]


def probe_jobs(stage: str, devices: list[str], smoke: bool = False) -> list[Job]:
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    result = []
    combinations = [(family, condition, seed) for family in FAMILIES for condition in CONDITIONS for seed in seeds]
    for index, (family, condition, seed) in enumerate(combinations):
        output = (
            PLAYER_POSTERIOR_ROOT / "smoke" / family / condition / f"seed{seed}"
            if smoke else test_dir(family, condition, seed) if stage == "test"
            else training_dir(family, condition, seed)
        )
        command = [
            sys.executable, str(RUNNER), "--action", "test" if stage == "test" else "train",
            "--family", family, "--condition", condition, "--seed", str(seed),
            "--device", f"cuda:{devices[index % len(devices)]}", "--output-dir", str(output),
        ]
        if smoke:
            command.append("--smoke")
        if stage == "test":
            command.extend(("--checkpoint", str(training_dir(family, condition, seed) / "best.pt")))
        result.append(Job(
            f"{stage}/{family}/{condition}/seed{seed}", tuple(command),
            output / "result.json", output / f"console_{stage}.log",
        ))
    return result


def execute(jobs: list[Job], workers: int, force: bool) -> None:
    status_path = PLAYER_POSTERIOR_ROOT / "background/status.json"
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
            update(job.key, "skipped"); return
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

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for future in [pool.submit(run, job) for job in jobs]:
            future.result()


def action(name: str) -> None:
    subprocess.run([sys.executable, str(RUNNER), "--action", name], cwd=ROOT, check=True)


def verify() -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_player_posterior_stage_b.py"],
        cwd=ROOT, check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("verify", "cache", "smoke", "validation", "lock", "test", "report", "pipeline"), required=True
    )
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workers = len(args.devices) * args.slots_per_gpu
    if args.stage == "verify": verify(); return
    if args.stage == "cache":
        execute(cache_jobs("train") + cache_jobs("validation"), 3, args.force); return
    if args.stage == "smoke":
        execute(probe_jobs("validation", args.devices, True), min(3, len(args.devices)), args.force); return
    if args.stage == "validation":
        execute(probe_jobs("validation", args.devices), workers, args.force); return
    if args.stage == "lock": action("lock"); return
    if args.stage == "test":
        if not decision_path().exists(): raise RuntimeError("Validation decision lock required")
        execute(cache_jobs("test"), 3, args.force)
        execute(probe_jobs("test", args.devices), workers, args.force); return
    if args.stage == "report": action("report"); return
    verify()
    execute(cache_jobs("train") + cache_jobs("validation"), 3, args.force)
    execute(probe_jobs("validation", args.devices, True), min(3, len(args.devices)), args.force)
    execute(probe_jobs("validation", args.devices), workers, args.force)
    action("lock")
    execute(cache_jobs("test"), 3, args.force)
    execute(probe_jobs("test", args.devices), workers, args.force)
    action("report")


if __name__ == "__main__":
    main()

