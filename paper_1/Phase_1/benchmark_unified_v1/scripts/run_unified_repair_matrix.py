#!/usr/bin/env python
"""Tune and evaluate only the repaired Unified LEM baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_experiment.py"
ARTIFACT = ROOT / "artifacts/feasibility/protocol.pt"
SAMPLE_PLAN = ROOT / "artifacts/feasibility/sample_plan.json"
EXPERIMENT_ROOT = ROOT / "experiments/feasibility/unified_lem_repair"
LEARNING_RATES = (3e-5, 1e-4, 3e-4)
TUNING_SEED = 20260715
FINAL_SEEDS = (20260715, 20260716, 20260717)


@dataclass(frozen=True)
class Job:
    learning_rate: float
    seed: int
    output_dir: Path
    validation_only: bool = False
    checkpoint: Path | None = None
    smoke: bool = False

    def command(self, device: str) -> list[str]:
        command = [
            sys.executable,
            str(RUNNER),
            "--contract",
            "unified_lem",
            "--model",
            "unified_lem",
            "--window-size",
            "80",
            "--learning-rate",
            str(self.learning_rate),
            "--seed",
            str(self.seed),
            "--device",
            device,
            "--epochs",
            "12",
            "--patience",
            "3",
            "--artifact-path",
            str(ARTIFACT),
            "--sample-plan-path",
            str(SAMPLE_PLAN),
            "--unified-event-loss-mode",
            "sqrt_capped",
            "--num-workers",
            "2",
            "--output-dir",
            str(self.output_dir),
        ]
        if self.validation_only:
            command.append("--validation-only")
        if self.checkpoint is not None:
            command.extend(("--checkpoint", str(self.checkpoint)))
        if self.smoke:
            command.append("--smoke")
        return command


def tuning_jobs() -> list[Job]:
    return [
        Job(
            learning_rate,
            TUNING_SEED,
            EXPERIMENT_ROOT / "tuning" / f"lr{learning_rate:g}",
            validation_only=True,
        )
        for learning_rate in LEARNING_RATES
    ]


def selected_learning_rate() -> float:
    losses: dict[str, float] = {}
    for learning_rate in LEARNING_RATES:
        path = EXPERIMENT_ROOT / "tuning" / f"lr{learning_rate:g}" / "result.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            losses[str(learning_rate)] = float(result["validation"]["loss"])
    if len(losses) != len(LEARNING_RATES):
        raise RuntimeError(f"Incomplete Unified repair tuning results: {losses}")
    selected = float(min(losses, key=losses.get))
    (EXPERIMENT_ROOT / "selected_learning_rate.json").write_text(
        json.dumps(
            {
                "selected": selected,
                "validation_losses": losses,
                "event_loss_mode": "sqrt_capped",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return selected


def final_jobs(learning_rate: float) -> list[Job]:
    tuning_checkpoint = (
        EXPERIMENT_ROOT / "tuning" / f"lr{learning_rate:g}" / "best.pt"
    )
    return [
        Job(
            learning_rate,
            seed,
            EXPERIMENT_ROOT / "final" / f"seed{seed}",
            checkpoint=tuning_checkpoint if seed == TUNING_SEED else None,
        )
        for seed in FINAL_SEEDS
    ]


def execute(jobs: list[Job], devices: list[str], force: bool) -> None:
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")

    def run(job: Job, device: str) -> None:
        result_path = job.output_dir / "result.json"
        if result_path.exists() and not force:
            print(f"SKIP {result_path}", flush=True)
            return
        job.output_dir.mkdir(parents=True, exist_ok=True)
        command = job.command(device)
        print("RUN " + " ".join(command), flush=True)
        with (job.output_dir / "console.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                env=environment,
            )

    with ThreadPoolExecutor(max_workers=min(len(jobs), len(devices))) as pool:
        futures = [
            pool.submit(run, job, devices[index % len(devices)])
            for index, job in enumerate(jobs)
        ]
        for future in futures:
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("print", "smoke", "tune", "final"), default="print"
    )
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "print":
        print(
            json.dumps(
                {
                    "model": "unified_lem",
                    "event_loss_mode": "sqrt_capped",
                    "tuning_jobs": 3,
                    "additional_training_jobs": 2,
                    "reused_checkpoint_evaluations": 1,
                    "max_epochs": 12,
                    "patience": 3,
                    "learning_rates": LEARNING_RATES,
                },
                indent=2,
            )
        )
        return
    if args.stage == "smoke":
        jobs = [
            Job(
                1e-4,
                TUNING_SEED,
                EXPERIMENT_ROOT / "smoke",
                validation_only=True,
                smoke=True,
            )
        ]
    elif args.stage == "tune":
        jobs = tuning_jobs()
    else:
        jobs = final_jobs(selected_learning_rate())
    execute(jobs, args.devices, args.force)


if __name__ == "__main__":
    main()
