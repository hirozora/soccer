#!/usr/bin/env python
"""Resumable multi-GPU Semantic V3 subgraph-scale experiment pipeline."""

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
from football_hgt_targets_v4.subgraph_analysis import build_view_statistics  # noqa: E402
from football_hgt_targets_v4.subgraph_reporting import (  # noqa: E402
    build_test_report,
    select_round_b,
    select_task_winners,
)
from football_hgt_targets_v4.subgraph_study import (  # noqa: E402
    SUBGRAPH_EXPERIMENT_ROOT,
    test_dir,
    validation_dir,
)
from football_hgt_targets_v4.subgraph_views import (  # noqa: E402
    ROUND_A_VIEWS,
    ROUND_B_BY_FAMILY,
)

RUNNER = ROOT / "scripts/run_experiment.py"
VERIFY = ROOT / "scripts/verify_subgraph_f80.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    result: Path
    log: Path


def ensure_view_statistics(force: bool = False) -> Path:
    output = SUBGRAPH_EXPERIMENT_ROOT / "view_statistics"
    if (output / "manifest.json").exists() and not force:
        return output
    return build_view_statistics(output)


def _command(
    view: str,
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
        "--task", "joint",
        "--method", f"subgraph_{view}",
        "--event-method", "ce",
        "--time-method", "current_huber",
        "--position-method", "xy",
        "--event-loss-weight", "0.2",
        "--graph-variant", "semantic_v3_possession",
        "--possession-topology", "membership",
        "--possession-feature-level", "dynamic",
        "--snapshot-scope", "selected_events",
        "--context-view", view,
        "--output-dir", str(output),
        "--learning-rate", "0.0009",
        "--seed", str(seed),
        "--device", f"cuda:{device}",
        "--epochs", "8",
        "--patience", "2",
        "--batch-size", "256",
        "--num-workers", "3",
    ]
    if smoke:
        command.append("--smoke")
    if checkpoint is not None:
        command.extend(
            [
                "--evaluate-only",
                "--evaluate-test",
                "--full-test",
                "--checkpoint-path", str(checkpoint),
            ]
        )
    return tuple(command)


def _jobs_for_views(
    views: list[str] | tuple[str, ...], devices: list[str], stage: str, smoke: bool = False
) -> list[Job]:
    jobs = []
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    for view in views:
        for seed in seeds:
            output = (
                SUBGRAPH_EXPERIMENT_ROOT / "smoke" / view / f"seed{seed}"
                if smoke
                else validation_dir(view, seed)
            )
            jobs.append(
                Job(
                    f"{stage}/{view}/seed{seed}",
                    _command(view, seed, devices[len(jobs) % len(devices)], output, smoke=smoke),
                    output / "result.json",
                    output / "console.log",
                )
            )
    return jobs


def _test_jobs(devices: list[str]) -> list[Job]:
    selected = select_task_winners()
    views = sorted(set(selected["winners"].values()) | set(selected["size_matched_controls"]))
    jobs = []
    for view in views:
        if view == "f80":
            continue
        for seed in CONFIRMATION_SEEDS:
            output = test_dir(view, seed)
            checkpoint = validation_dir(view, seed) / "best.pt"
            jobs.append(
                Job(
                    f"test/{view}/seed{seed}",
                    _command(
                        view,
                        seed,
                        devices[len(jobs) % len(devices)],
                        output,
                        checkpoint=checkpoint,
                    ),
                    output / "result.json",
                    output / "console.log",
                )
            )
    return jobs


def execute(jobs: list[Job], workers: int, force: bool, status_path: Path) -> None:
    if not jobs:
        return
    status_path.parent.mkdir(parents=True, exist_ok=True)
    states = json.loads(status_path.read_text()) if status_path.exists() else {}
    states = {
        key: ("interrupted" if value == "running" else value)
        for key, value in states.items()
    }
    lock = threading.Lock()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src"), environment.get("PYTHONPATH", "")]
    )
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
            completed = subprocess.run(
                job.command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        state = "completed" if completed.returncode == 0 else f"failed:{completed.returncode}"
        update(job.key, state)
        if completed.returncode:
            raise RuntimeError(f"Failed {job.key}: {job.log}")

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for future in [pool.submit(run, job) for job in jobs]:
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("verify", "stats", "smoke", "round-a", "round-b", "controls", "test", "summarize", "pipeline"),
        required=True,
    )
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workers = len(args.devices) * args.slots_per_gpu
    status = SUBGRAPH_EXPERIMENT_ROOT / "background/status.json"
    all_round_b = tuple(view for values in ROUND_B_BY_FAMILY.values() for view in values)

    if args.stage == "verify":
        subprocess.run([sys.executable, str(VERIFY)], cwd=ROOT, check=True); return
    if args.stage == "stats":
        print(ensure_view_statistics(args.force)); return
    if args.stage == "smoke":
        execute(_jobs_for_views((*ROUND_A_VIEWS, *all_round_b), args.devices, "smoke", True), workers, args.force, status); return
    if args.stage == "round-a":
        execute(_jobs_for_views(ROUND_A_VIEWS, args.devices, "round-a"), workers, args.force, status); return
    if args.stage == "round-b":
        selection = select_round_b()
        execute(_jobs_for_views(selection["round_b_views"], args.devices, "round-b"), workers, args.force, status); return
    if args.stage == "controls":
        selected = select_task_winners()
        execute(_jobs_for_views(selected["size_matched_controls"], args.devices, "controls"), workers, args.force, status); return
    if args.stage == "test":
        execute(_test_jobs(args.devices), workers, args.force, status); return
    if args.stage == "summarize":
        print(build_test_report()); return

    subprocess.run([sys.executable, str(VERIFY)], cwd=ROOT, check=True)
    print(ensure_view_statistics(args.force))
    execute(_jobs_for_views((*ROUND_A_VIEWS, *all_round_b), args.devices, "smoke", True), workers, args.force, status)
    execute(_jobs_for_views(ROUND_A_VIEWS, args.devices, "round-a"), workers, args.force, status)
    selection = select_round_b()
    execute(_jobs_for_views(selection["round_b_views"], args.devices, "round-b"), workers, args.force, status)
    selected = select_task_winners()
    execute(_jobs_for_views(selected["size_matched_controls"], args.devices, "controls"), workers, args.force, status)
    execute(_test_jobs(args.devices), workers, args.force, status)
    print(build_test_report())


if __name__ == "__main__":
    main()
