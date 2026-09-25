#!/usr/bin/env python
"""Conditional, resumable multi-GPU fixed-budget conflict pipeline."""

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
from football_hgt_targets_v4.fixed_budget_decisions import (  # noqa: E402
    conflict_decision, lock_baseline, stage2_trigger_decision, training_boundary_decision,
)
from football_hgt_targets_v4.fixed_budget_reporting import build_fixed_budget_report  # noqa: E402
from football_hgt_targets_v4.fixed_budget_study import (  # noqa: E402
    CONFIGURATIONS, FIXED_BUDGET_ROOT, MATCHED_PAIRS, STAGE1, STAGE2_NEW, test_dir, training_dir,
)

RUNNER = ROOT / "scripts/run_fixed_budget_experiment.py"


@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    result: Path
    log: Path


def _command(configuration: str, seed: int, device: str, output: Path, budget: int, *, smoke: bool = False, resume: Path | None = None, checkpoint: Path | None = None) -> tuple[str, ...]:
    command = [
        sys.executable, str(RUNNER), "--configuration", configuration,
        "--output-dir", str(output), "--seed", str(seed), "--device", f"cuda:{device}",
        "--budget", str(budget), "--batch-size", "256", "--num-workers", "2",
    ]
    if smoke:
        command.append("--smoke")
    if resume is not None:
        command.extend(("--resume-from", str(resume)))
    if checkpoint is not None:
        command.extend(("--evaluate-test", "--full-test", "--checkpoint-path", str(checkpoint)))
    return tuple(command)


def execute(jobs: list[Job], workers: int, force: bool = False) -> None:
    status_path = FIXED_BUDGET_ROOT / "background/status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    states = json.loads(status_path.read_text()) if status_path.exists() else {}
    states = {key: ("interrupted" if value == "running" else value) for key, value in states.items()}
    lock = threading.Lock()
    job_devices = {
        job.key: job.command[job.command.index("--device") + 1]
        for job in jobs
    }
    unique_devices = set(job_devices.values())
    slots_per_device = max(1, workers // max(len(unique_devices), 1))
    device_slots = {
        device: threading.Semaphore(slots_per_device) for device in unique_devices
    }
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
        with device_slots[job_devices[job.key]]:
            update(job.key, "running")
            with job.log.open("w", encoding="utf-8") as log:
                completed = subprocess.run(job.command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        update(job.key, "completed" if completed.returncode == 0 else f"failed:{completed.returncode}")
        if completed.returncode:
            raise RuntimeError(f"Failed {job.key}: {job.log}")

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for future in [pool.submit(run, job) for job in jobs]:
            future.result()


def _training_jobs(configurations: tuple[str, ...] | list[str], devices: list[str], budget: int, stage: str, *, smoke: bool = False, resume: bool = False) -> list[Job]:
    jobs = []
    seeds = (CONFIRMATION_SEEDS[0],) if smoke else CONFIRMATION_SEEDS
    for configuration in configurations:
        for seed in seeds:
            output = FIXED_BUDGET_ROOT / "smoke" / configuration / f"seed{seed}" if smoke else training_dir(configuration, seed)
            jobs.append(Job(
                f"{stage}/{configuration}/seed{seed}",
                _command(configuration, seed, devices[len(jobs) % len(devices)], output, budget, smoke=smoke, resume=(output / "last.pt") if resume else None),
                output / f"complete_epoch_{1 if smoke else budget}.json",
                output / f"console_{stage}.log",
            ))
    return jobs


def _extended_configurations(boundary: dict) -> list[str]:
    result = []
    for pair in boundary["pairs"]:
        if pair["extend_to_24"]:
            result.extend((pair["three"], pair["five"]))
    return result


def _f80_budget(boundary: dict) -> int:
    pair = next(value for value in boundary["pairs"] if value["three"] == "three_f80")
    return 24 if pair["extend_to_24"] else 16


def _checkpoint(configuration: str) -> str:
    return "best_joint.pt" if CONFIGURATIONS[configuration]["checkpoint_metric"] == "joint_five" else "best_core.pt"


def _test_jobs(configurations: list[str], devices: list[str]) -> list[Job]:
    jobs = []
    for label in configurations:
        configuration = "five_f80" if label == "t5_core" else label
        for seed in CONFIRMATION_SEEDS:
            output = test_dir(label, seed)
            source = training_dir(configuration, seed)
            budget = json.loads((source / "result.json").read_text())["epochs_completed"]
            checkpoint_name = "best_core.pt" if label == "t5_core" else _checkpoint(configuration)
            jobs.append(Job(
                f"test/{label}/seed{seed}",
                _command(configuration, seed, devices[len(jobs) % len(devices)], output, budget, checkpoint=source / checkpoint_name),
                output / "result.json", output / "console_test.log",
            ))
    return jobs


def _verify() -> None:
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_fixed_budget_conflict.py"], cwd=ROOT, check=True)


def run_pipeline(devices: list[str], slots: int, force: bool) -> None:
    _verify()
    smoke_configs = ("three_f80", "five_f80", "t4_player", "t5_player_adapter")
    execute(_training_jobs(smoke_configs, devices, 1, "smoke", smoke=True), len(devices), force)
    execute(_training_jobs(STAGE1, devices, 16, "length"), len(devices) * slots, force)
    boundary = training_boundary_decision()
    extended = _extended_configurations(boundary)
    if extended:
        execute(_training_jobs(extended, devices, 24, "extend", resume=True), len(devices) * slots, force)
        boundary = training_boundary_decision_after_extension(boundary)
    stage2 = stage2_trigger_decision()
    trained = list(STAGE1)
    if stage2["run_stage2"]:
        budget = _f80_budget(boundary)
        execute(_training_jobs(STAGE2_NEW, devices, budget, "conflict"), len(devices) * slots, force)
        trained.extend(STAGE2_NEW)
        trained.append("t5_core")
        conflict = conflict_decision()
        if conflict["run_player_adapter"]:
            execute(_training_jobs(("t5_player_adapter",), devices, budget, "adapter"), len(devices) * slots, force)
            trained.append("t5_player_adapter")
    lock_baseline()
    execute(_test_jobs(trained, devices), len(devices) * slots, force)
    print(build_fixed_budget_report())


def training_boundary_decision_after_extension(original: dict) -> dict:
    # Preserve the preregistered extension decision; add final boundary status for reporting.
    final = {**original, "final_budget": {}, "final_training_boundary": {}}
    for pair in original["pairs"]:
        budget = 24 if pair["extend_to_24"] else 16
        for configuration in (pair["three"], pair["five"]):
            final["final_budget"][configuration] = budget
            metric = "core_etp_loss" if configuration.startswith("three_") else "joint_active_loss"
            seed_states = []
            for seed in CONFIRMATION_SEEDS:
                result = json.loads((training_dir(configuration, seed) / "result.json").read_text())
                history = result["history"]
                if len(history) != budget:
                    raise RuntimeError(f"{configuration}/seed{seed} has {len(history)} epochs, expected {budget}")
                if budget == 24:
                    previous = sum(row[metric] for row in history[18:21]) / 3
                    recent = sum(row[metric] for row in history[21:24]) / 3
                    best_epoch = min(history, key=lambda row: row[metric])["epoch"]
                    hit = best_epoch in {23, 24} and recent < previous - 1e-4
                else:
                    previous = recent = None
                    best_epoch = min(history, key=lambda row: row[metric])["epoch"]
                    hit = False
                seed_states.append({"seed": seed, "best_epoch": best_epoch, "previous_mean": previous, "recent_mean": recent, "hit": hit})
            final["final_training_boundary"][configuration] = {
                "seeds": seed_states,
                "boundary_hit_at_final_cap": sum(value["hit"] for value in seed_states) >= 2,
            }
    path = FIXED_BUDGET_ROOT / "decisions/training_boundary.json"
    path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "length", "extend", "conflict", "adapter", "test", "summarize", "pipeline"), required=True)
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workers = len(args.devices) * args.slots_per_gpu
    if args.stage == "verify":
        _verify(); return
    if args.stage == "length":
        execute(_training_jobs(STAGE1, args.devices, 16, "length"), workers, args.force)
        training_boundary_decision(); return
    if args.stage == "extend":
        boundary = training_boundary_decision()
        execute(_training_jobs(_extended_configurations(boundary), args.devices, 24, "extend", resume=True), workers, args.force)
        training_boundary_decision_after_extension(boundary); return
    if args.stage == "conflict":
        boundary = json.loads((FIXED_BUDGET_ROOT / "decisions/training_boundary.json").read_text())
        if stage2_trigger_decision()["run_stage2"]:
            execute(_training_jobs(STAGE2_NEW, args.devices, _f80_budget(boundary), "conflict"), workers, args.force)
            conflict_decision()
        return
    if args.stage == "adapter":
        boundary = json.loads((FIXED_BUDGET_ROOT / "decisions/training_boundary.json").read_text())
        if conflict_decision()["run_player_adapter"]:
            execute(_training_jobs(("t5_player_adapter",), args.devices, _f80_budget(boundary), "adapter"), workers, args.force)
        return
    if args.stage == "test":
        lock = lock_baseline()
        configurations = list(STAGE1)
        configurations.extend([name for name in (*STAGE2_NEW, "t5_player_adapter") if (training_dir(name, CONFIRMATION_SEEDS[0]) / "result.json").exists()])
        if all((training_dir(name, CONFIRMATION_SEEDS[0]) / "result.json").exists() for name in STAGE2_NEW):
            configurations.append("t5_core")
        execute(_test_jobs(configurations, args.devices), workers, args.force); return
    if args.stage == "summarize":
        print(build_fixed_budget_report()); return
    run_pipeline(args.devices, args.slots_per_gpu, args.force)


if __name__ == "__main__":
    main()
