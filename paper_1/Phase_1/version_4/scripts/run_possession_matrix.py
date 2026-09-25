#!/usr/bin/env python
"""Resumable multi-GPU Semantic V3 Possession regression pipeline."""

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
from football_hgt_targets_v4.possession_reporting import build_possession_report  # noqa: E402
from football_hgt_targets_v4.possession_study import (  # noqa: E402
    FEATURE_VARIANTS,
    POSSESSION_EXPERIMENT_ROOT,
    TOPOLOGY_VARIANTS,
    select_final_v3,
    select_topology,
    validation_dir,
)

RUNNER = ROOT / "scripts/run_experiment.py"
VERIFY = ROOT / "scripts/verify_possession_n0.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    result: Path
    log: Path


def _training_command(
    variant: str,
    seed: int,
    device: str,
    output: Path,
    topology: str,
    feature_level: str,
    *,
    j0: bool = False,
    smoke: bool = False,
    checkpoint: Path | None = None,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(RUNNER),
        "--task", "joint",
        "--method", variant,
        "--event-method", "inverse_ce" if j0 else "ce",
        "--time-method", "current_huber",
        "--position-method", "xy",
        "--event-loss-weight", "1.0" if j0 else "0.2",
        "--graph-variant", "semantic_v3_possession",
        "--possession-topology", topology,
        "--possession-feature-level", feature_level,
        "--snapshot-scope", "selected_events",
        "--output-dir", str(output),
        "--learning-rate", "0.0009",
        "--seed", str(seed),
        "--device", f"cuda:{device}",
        "--epochs", "8",
        "--patience", "2",
        "--batch-size", "256",
        # HeteroData batches contain many independent tensors. Long-lived worker
        # queues eventually exhaust ancillary FD transfer under high concurrency.
        "--num-workers", "0",
    ]
    if smoke:
        command.append("--smoke")
    if checkpoint is not None:
        command.extend(
            ["--evaluate-only", "--evaluate-test", "--full-test", "--checkpoint-path", str(checkpoint)]
        )
    return tuple(command)


def topology_jobs(devices: list[str], smoke: bool = False) -> list[Job]:
    jobs = []
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    for index, (variant, (topology, features)) in enumerate(TOPOLOGY_VARIANTS.items()):
        for seed in seeds:
            output = (
                POSSESSION_EXPERIMENT_ROOT / "smoke" / variant / f"seed{seed}"
                if smoke
                else validation_dir(variant, seed)
            )
            jobs.append(Job(
                f"{'smoke' if smoke else 'topology'}/{variant}/seed{seed}",
                _training_command(variant, seed, devices[len(jobs) % len(devices)], output, topology, features, smoke=smoke),
                output / "result.json",
                output / "console.log",
            ))
    return jobs


def feature_jobs(devices: list[str], smoke: bool = False) -> list[Job]:
    topology = "transition" if smoke else select_topology()["selected_topology"]
    jobs = []
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    for variant, features in FEATURE_VARIANTS.items():
        for seed in seeds:
            output = (
                POSSESSION_EXPERIMENT_ROOT / "smoke" / variant / f"seed{seed}"
                if smoke
                else validation_dir(variant, seed)
            )
            jobs.append(Job(
                f"{'smoke' if smoke else 'features'}/{variant}/seed{seed}",
                _training_command(variant, seed, devices[len(jobs) % len(devices)], output, topology, features, smoke=smoke),
                output / "result.json",
                output / "console.log",
            ))
    return jobs


def j0_jobs(devices: list[str]) -> list[Job]:
    selected = select_final_v3()
    jobs = []
    for seed in CONFIRMATION_SEEDS:
        output = validation_dir("v3_j0", seed)
        jobs.append(Job(
            f"j0/seed{seed}",
            _training_command(
                "v3_j0", seed, devices[len(jobs) % len(devices)], output,
                selected["selected_topology"], selected["selected_feature_level"], j0=True,
            ),
            output / "result.json", output / "console.log",
        ))
    return jobs


def test_jobs(devices: list[str]) -> list[Job]:
    selected = select_final_v3()
    topology = selected["selected_topology"]
    configurations = {
        **TOPOLOGY_VARIANTS,
        "c1_categorical": (topology, "categorical"),
        "d2_dynamic": (topology, "dynamic"),
        "v3_j0": (topology, selected["selected_feature_level"]),
    }
    jobs = []
    for variant, (variant_topology, features) in configurations.items():
        for seed in CONFIRMATION_SEEDS:
            output = POSSESSION_EXPERIMENT_ROOT / "test" / variant / f"seed{seed}"
            checkpoint = validation_dir(variant, seed) / "best.pt"
            jobs.append(Job(
                f"test/{variant}/seed{seed}",
                _training_command(
                    variant, seed, devices[len(jobs) % len(devices)], output,
                    variant_topology, features, j0=variant == "v3_j0", checkpoint=checkpoint,
                ),
                output / "result.json", output / "console.log",
            ))
    return jobs


def execute(jobs: list[Job], workers: int, force: bool, status_path: Path) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    states = json.loads(status_path.read_text()) if status_path.exists() else {}
    lock = threading.Lock()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src"), environment.get("PYTHONPATH", "")]
    )
    environment.setdefault("OMP_NUM_THREADS", "2")

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
            completed = subprocess.run(
                job.command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        update(job.key, "completed" if completed.returncode == 0 else f"failed:{completed.returncode}")
        if completed.returncode:
            raise RuntimeError(f"Failed {job.key}: {job.log}")

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for future in [pool.submit(run, job) for job in jobs]:
            future.result()


def verify_n0(force: bool = False) -> None:
    output = POSSESSION_EXPERIMENT_ROOT / "n0/equivalence.json"
    if output.exists() and not force:
        return
    subprocess.run([sys.executable, str(VERIFY), "--output", str(output)], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("verify", "smoke", "topology", "features", "j0", "test", "summarize", "pipeline"),
        required=True,
    )
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--slots-per-gpu", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workers = len(args.devices) * args.slots_per_gpu
    status = POSSESSION_EXPERIMENT_ROOT / "background/status.json"
    if args.stage == "verify":
        verify_n0(args.force); return
    if args.stage == "smoke":
        execute(topology_jobs(args.devices, True) + feature_jobs(args.devices, True), workers, args.force, status); return
    if args.stage == "topology":
        execute(topology_jobs(args.devices), workers, args.force, status); select_topology(); return
    if args.stage == "features":
        execute(feature_jobs(args.devices), workers, args.force, status); select_final_v3(); return
    if args.stage == "j0":
        execute(j0_jobs(args.devices), workers, args.force, status); return
    if args.stage == "test":
        execute(test_jobs(args.devices), workers, args.force, status); return
    if args.stage == "summarize":
        print(build_possession_report()); return
    verify_n0(args.force)
    execute(topology_jobs(args.devices, True) + feature_jobs(args.devices, True), workers, args.force, status)
    execute(topology_jobs(args.devices), workers, args.force, status)
    select_topology()
    execute(feature_jobs(args.devices), workers, args.force, status)
    select_final_v3()
    execute(j0_jobs(args.devices), workers, args.force, status)
    execute(test_jobs(args.devices), workers, args.force, status)
    print(build_possession_report())


if __name__ == "__main__":
    main()
