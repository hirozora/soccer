#!/usr/bin/env python
"""Run, resume, select, and summarize Version 4 experiment matrices."""

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
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_hgt_targets_v4.constants import (  # noqa: E402
    CONFIRMATION_METHODS,
    CONFIRMATION_SEEDS,
    EVENT_METHODS,
    EXPERIMENT_ROOT,
    LOSS_BALANCE_SCALES,
    METHODS_BY_TASK,
    ORIGINAL_JOINT_METHODS,
    OPTIMIZED_JOINT_METHODS,
    POSITION_METHODS,
    SCREEN_SEED,
    TIME_METHODS,
)
from football_hgt_targets_v4.selection import select_confirmed_methods  # noqa: E402
from football_hgt_targets_v4.loss_balance_reporting import (  # noqa: E402
    build_loss_balance_report,
    build_reference_diagnostics,
    scale_label,
    select_loss_balance,
)

RUNNER = ROOT / "scripts/run_experiment.py"


@dataclass(frozen=True)
class Job:
    stage: str
    task: str
    method: str
    seed: int
    output_dir: Path
    full: bool = False
    smoke: bool = False
    evaluate_test: bool = False
    full_test: bool = False
    joint_methods: dict[str, str] | None = None
    checkpoint_path: Path | None = None
    event_loss_weight: float | None = None

    def command(self, device: str) -> list[str]:
        command = [
            sys.executable,
            str(RUNNER),
            "--task",
            self.task,
            "--method",
            self.method,
            "--seed",
            str(self.seed),
            "--device",
            device,
            "--output-dir",
            str(self.output_dir),
            "--learning-rate",
            "0.0009",
            "--batch-size",
            "256",
            "--num-workers",
            "2",
            "--epochs",
            "12" if self.full else "8",
            "--patience",
            "3" if self.full else "2",
        ]
        if self.full:
            command.append("--full")
        if self.smoke:
            command.append("--smoke")
        if self.evaluate_test:
            command.append("--evaluate-test")
        if self.full_test:
            command.append("--full-test")
        if self.checkpoint_path is not None:
            command.extend(
                ["--evaluate-only", "--checkpoint-path", str(self.checkpoint_path)]
            )
        if self.event_loss_weight is not None:
            command.extend(["--event-loss-weight", str(self.event_loss_weight)])
        if self.joint_methods:
            command.extend(
                [
                    "--event-method",
                    self.joint_methods["event"],
                    "--time-method",
                    self.joint_methods["time"],
                    "--position-method",
                    self.joint_methods["position"],
                ]
            )
        return command


def basic_jobs(stage: str, smoke: bool = False) -> list[Job]:
    root = EXPERIMENT_ROOT / stage
    return [
        Job(stage, task, method, SCREEN_SEED, root / task / method, smoke=smoke)
        for task, methods in METHODS_BY_TASK.items()
        for method in methods
    ]


def _read_result(path: Path) -> dict[str, Any]:
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def screen_selection() -> dict[str, list[str]]:
    root = EXPERIMENT_ROOT / "screen"
    results = {
        (task, method): _read_result(root / task / method)
        for task, methods in METHODS_BY_TASK.items()
        for method in methods
    }
    inverse_accuracy = results[("event", "inverse_ce")]["validation"]["event"]["accuracy"]
    acceptable = [
        method
        for method in EVENT_METHODS
        if results[("event", method)]["validation"]["event"]["accuracy"]
        >= inverse_accuracy - 0.01
    ]
    event_rank = sorted(
        acceptable or list(EVENT_METHODS),
        key=lambda method: results[("event", method)]["validation"]["event"]["macro_f1"],
        reverse=True,
    )
    if len(event_rank) < 2:
        remaining = [method for method in EVENT_METHODS if method not in event_rank]
        remaining.sort(
            key=lambda method: results[("event", method)]["validation"]["event"]["macro_f1"],
            reverse=True,
        )
        event_rank.extend(remaining)
    time_rank = sorted(
        TIME_METHODS,
        key=lambda method: results[("time", method)]["validation"]["time"]["mae_seconds"],
    )
    position_rank = sorted(
        POSITION_METHODS,
        key=lambda method: results[("position", method)]["validation"]["position"]["distance_mae_m"],
    )
    selected = {
        "event": event_rank[:2],
        "time": time_rank[:2],
        "position": position_rank[:2],
    }
    (root / "selection.json").write_text(
        json.dumps(
            {
                "selected": selected,
                "event_inverse_accuracy": inverse_accuracy,
                "event_accuracy_floor": inverse_accuracy - 0.01,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return selected


def confirmation_jobs() -> list[Job]:
    return [
        Job(
            "confirmation",
            task,
            method,
            seed,
            EXPERIMENT_ROOT / "confirmation" / task / method / f"seed{seed}",
        )
        for task, methods in CONFIRMATION_METHODS.items()
        for method in methods
        for seed in CONFIRMATION_SEEDS
        if seed != SCREEN_SEED
    ]


def _confirmation_result_dir(task: str, method: str, seed: int) -> Path:
    if seed == SCREEN_SEED:
        return EXPERIMENT_ROOT / "screen" / task / method
    return EXPERIMENT_ROOT / "confirmation" / task / method / f"seed{seed}"


def confirmed_winners() -> dict[str, str]:
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for task, methods in CONFIRMATION_METHODS.items():
        aggregate[task] = {}
        for method in methods:
            metric_values: dict[str, list[float]] = {}
            for seed in CONFIRMATION_SEEDS:
                result = _read_result(_confirmation_result_dir(task, method, seed))
                if result.get("test_accessed"):
                    raise RuntimeError("Confirmation result unexpectedly accessed the test split")
                metrics = result["validation"]
                if task == "event":
                    values = {
                        "accuracy": metrics["event"]["accuracy"],
                        "macro_f1": metrics["event"]["macro_f1"],
                    }
                elif task == "time":
                    values = {
                        "mae_seconds": metrics["time"]["mae_seconds"],
                        "median_ae_seconds": metrics["time"]["median_ae_seconds"],
                    }
                else:
                    values = {
                        "distance_mae_m": metrics["position"]["distance_mae_m"],
                        "zone_accuracy": metrics["position"]["zone_accuracy"],
                    }
                row = {"task": task, "method": method, "seed": seed, **values}
                rows.append(row)
                for name, value in values.items():
                    metric_values.setdefault(name, []).append(float(value))
            aggregate[task][method] = {
                name: sum(values) / len(values) for name, values in metric_values.items()
            }

    inverse_accuracy = aggregate["event"]["inverse_ce"]["accuracy"]
    winners, accuracy_floor = select_confirmed_methods(aggregate)
    target = EXPERIMENT_ROOT / "confirmation/winners.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target.parent / "summary_three_seed.csv", index=False)
    target.write_text(
        json.dumps(
            {
                "winners": winners,
                "means": aggregate,
                "event_inverse_accuracy": inverse_accuracy,
                "event_accuracy_floor": accuracy_floor,
                "test_accessed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return winners


def _single_checkpoint(task: str, method: str, seed: int) -> Path:
    return _confirmation_result_dir(task, method, seed) / "best.pt"


def final_jobs() -> list[Job]:
    winners = confirmed_winners()
    joint_jobs = [
        Job(
            "final",
            "joint",
            label,
            seed,
            EXPERIMENT_ROOT / "final" / label / f"seed{seed}",
            evaluate_test=True,
            full_test=True,
            joint_methods=methods,
        )
        for label, methods in (
            ("joint_original", ORIGINAL_JOINT_METHODS),
            ("joint_optimized", winners),
        )
        for seed in CONFIRMATION_SEEDS
    ]
    evaluation_jobs = [
        Job(
            "final",
            task,
            method,
            seed,
            EXPERIMENT_ROOT
            / "final"
            / "single_task_test"
            / task
            / method
            / f"seed{seed}",
            evaluate_test=True,
            full_test=True,
            checkpoint_path=_single_checkpoint(task, method, seed),
        )
        for task, method in winners.items()
        for seed in CONFIRMATION_SEEDS
    ]
    return joint_jobs + evaluation_jobs


def loss_balance_jobs() -> list[Job]:
    return [
        Job(
            "loss_balance_validation",
            "joint",
            scale_label(scale),
            seed,
            EXPERIMENT_ROOT
            / "loss_balance_validation"
            / scale_label(scale)
            / f"seed{seed}",
            joint_methods=OPTIMIZED_JOINT_METHODS,
            event_loss_weight=scale,
        )
        for scale in LOSS_BALANCE_SCALES
        for seed in CONFIRMATION_SEEDS
    ]


def loss_balance_test_jobs(selection: dict[str, Any]) -> list[Job]:
    label = selection.get("selected")
    scale = selection.get("selected_event_loss_weight")
    if label is None or scale is None:
        return []
    return [
        Job(
            "loss_balance_test",
            "joint",
            label,
            seed,
            EXPERIMENT_ROOT / "loss_balance_test" / label / f"seed{seed}",
            evaluate_test=True,
            full_test=True,
            joint_methods=OPTIMIZED_JOINT_METHODS,
            checkpoint_path=(
                EXPERIMENT_ROOT
                / "loss_balance_validation"
                / label
                / f"seed{seed}"
                / "best.pt"
            ),
            event_loss_weight=float(scale),
        )
        for seed in CONFIRMATION_SEEDS
    ]


def summarize(stage: str) -> Path:
    root = EXPERIMENT_ROOT / stage
    rows: list[dict[str, Any]] = []
    for result_path in sorted(root.rglob("result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        config = result["config"]
        metric_split = "validation" if result.get("validation") is not None else "test"
        metrics = result[metric_split]
        row: dict[str, Any] = {
            "task": config["task"],
            "method": config["method"],
            "seed": config["seed"],
            "best_epoch": result["best_epoch"],
            "elapsed_seconds": result["elapsed_seconds"],
            "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
            "test_accessed": result["test_accessed"],
            "metric_split": metric_split,
        }
        if "event" in metrics:
            row.update(
                event_accuracy=metrics["event"]["accuracy"],
                event_macro_f1=metrics["event"]["macro_f1"],
            )
        if "time" in metrics:
            row.update(
                time_mae=metrics["time"]["mae_seconds"],
                time_median_ae=metrics["time"]["median_ae_seconds"],
            )
        if "position" in metrics:
            row.update(
                position_distance=metrics["position"]["distance_mae_m"],
                zone_accuracy=metrics["position"]["zone_accuracy"],
            )
        rows.append(row)
    output = root / "summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    if stage == "screen" and len(rows) == 10:
        screen_selection()
    return output


def execute(jobs: list[Job], devices: list[str], slots: int, force: bool) -> None:
    expanded = [f"cuda:{device}" for _ in range(slots) for device in devices]
    status_root = EXPERIMENT_ROOT / jobs[0].stage / "background"
    status_root.mkdir(parents=True, exist_ok=True)
    status_path = status_root / "status.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src"), environment.get("PYTHONPATH", "")]
    )
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    states: dict[str, str] = {}
    status_lock = threading.Lock()

    def write_status() -> None:
        with status_lock:
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(states, indent=2), encoding="utf-8")
            temporary.replace(status_path)

    def set_status(key: str, value: str) -> None:
        with status_lock:
            states[key] = value
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(states, indent=2), encoding="utf-8")
            temporary.replace(status_path)

    def run(job: Job, device: str) -> None:
        key = f"{job.task}/{job.method}/seed{job.seed}"
        result = job.output_dir / "result.json"
        if result.exists() and not force:
            set_status(key, "skipped")
            return
        job.output_dir.mkdir(parents=True, exist_ok=True)
        set_status(key, "running")
        with (job.output_dir / "console.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                job.command(device),
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        set_status(
            key,
            "completed" if completed.returncode == 0 else f"failed:{completed.returncode}",
        )
        if completed.returncode:
            raise RuntimeError(f"Failed {key}; see {job.output_dir / 'console.log'}")

    write_status()
    with ThreadPoolExecutor(max_workers=min(len(jobs), len(expanded))) as pool:
        futures = [
            pool.submit(run, job, expanded[index % len(expanded)])
            for index, job in enumerate(jobs)
        ]
        for future in futures:
            future.result()
    summarize(jobs[0].stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "smoke",
            "screen",
            "confirm",
            "final",
            "pipeline",
            "balance",
            "balance_pipeline",
            "summarize",
        ),
        required=True,
    )
    parser.add_argument(
        "--summary-stage",
        choices=(
            "smoke",
            "screen",
            "confirmation",
            "final",
            "loss_balance_validation",
            "loss_balance_test",
        ),
        default="screen",
    )
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--slots-per-gpu", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "summarize":
        print(summarize(args.summary_stage))
        return
    if args.stage == "pipeline":
        execute(confirmation_jobs(), args.devices, args.slots_per_gpu, args.force)
        confirmed_winners()
        execute(final_jobs(), args.devices, args.slots_per_gpu, args.force)
        from football_hgt_targets_v4.reporting import build_final_report

        print(build_final_report())
        return
    if args.stage == "balance_pipeline":
        execute(loss_balance_jobs(), args.devices, args.slots_per_gpu, args.force)
        selection = select_loss_balance()
        tests = loss_balance_test_jobs(selection)
        if tests:
            execute(tests, args.devices, args.slots_per_gpu, args.force)
        build_reference_diagnostics(f"cuda:{args.devices[0]}")
        print(build_loss_balance_report(selection))
        return
    jobs = {
        "smoke": lambda: basic_jobs("smoke", smoke=True),
        "screen": lambda: basic_jobs("screen"),
        "confirm": confirmation_jobs,
        "final": final_jobs,
        "balance": loss_balance_jobs,
    }[args.stage]()
    execute(jobs, args.devices, args.slots_per_gpu, args.force)


if __name__ == "__main__":
    main()
