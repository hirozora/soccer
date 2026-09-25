#!/usr/bin/env python
"""Run the K=80 three-seed controlled feasibility benchmark."""

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

LR_GRIDS = {
    "hgt": (1e-4, 3e-4, 9e-4),
    "seq2event": (3e-3, 1e-2, 3e-2),
    "unified_lem": (3e-4, 1e-3, 3e-3),
    "nmstpp": (3e-3, 1e-2, 3e-2),
}
BASELINES = {
    "seq2event": "seq2event",
    "unified_lem": "unified_lem",
    "nmstpp": "nmstpp",
}
TUNING_SEED = 20260715
FINAL_SEEDS = (20260715, 20260716, 20260717)


@dataclass(frozen=True)
class Job:
    contract: str
    family: str
    learning_rate: float
    seed: int
    output_dir: Path
    validation_only: bool = False
    smoke: bool = False
    checkpoint: Path | None = None

    def command(self, device: str) -> list[str]:
        command = [
            sys.executable,
            str(RUNNER),
            "--contract",
            self.contract,
            "--model",
            self.family,
            "--window-size",
            "80",
            "--learning-rate",
            str(self.learning_rate),
            "--seed",
            str(self.seed),
            "--device",
            device,
            "--epochs",
            "8",
            "--patience",
            "2",
            "--artifact-path",
            str(ARTIFACT),
            "--sample-plan-path",
            str(SAMPLE_PLAN),
            "--output-dir",
            str(self.output_dir),
        ]
        if self.validation_only:
            command.append("--validation-only")
        if self.smoke:
            command.append("--smoke")
        if self.checkpoint is not None:
            command.extend(("--checkpoint", str(self.checkpoint)))
        command.extend(("--num-workers", "2"))
        if self.family == "hgt":
            command.extend(("--micro-batch-size", "256"))
        return command


def tuning_jobs() -> list[Job]:
    return [
        Job(
            contract=contract,
            family=family,
            learning_rate=learning_rate,
            seed=TUNING_SEED,
            output_dir=(
                ROOT
                / "experiments/feasibility/tuning"
                / contract
                / family
                / f"lr{learning_rate:g}"
            ),
            validation_only=True,
        )
        for contract, baseline in BASELINES.items()
        for family in ("hgt", baseline)
        for learning_rate in LR_GRIDS[family]
    ]


def selected_learning_rates() -> dict[str, float]:
    selected: dict[str, float] = {}
    details: dict[str, dict[str, float]] = {}
    for contract, baseline in BASELINES.items():
        for family in ("hgt", baseline):
            candidates: dict[str, float] = {}
            for learning_rate in LR_GRIDS[family]:
                result_path = (
                    ROOT
                    / "experiments/feasibility/tuning"
                    / contract
                    / family
                    / f"lr{learning_rate:g}"
                    / "result.json"
                )
                if result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    candidates[str(learning_rate)] = float(result["validation"]["loss"])
            if len(candidates) != len(LR_GRIDS[family]):
                raise RuntimeError(
                    f"Incomplete feasibility tuning for {contract}/{family}: {candidates}"
                )
            best = min(candidates, key=candidates.get)
            selected[f"{contract}/{family}"] = float(best)
            details[f"{contract}/{family}"] = candidates
    payload = {"selected": selected, "validation_losses": details}
    path = ROOT / "experiments/feasibility/selected_learning_rates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return selected


def final_jobs(selected: dict[str, float]) -> list[Job]:
    jobs: list[Job] = []
    for contract, baseline in BASELINES.items():
        for family in ("hgt", baseline):
            learning_rate = selected[f"{contract}/{family}"]
            tuning_checkpoint = (
                ROOT
                / "experiments/feasibility/tuning"
                / contract
                / family
                / f"lr{learning_rate:g}"
                / "best.pt"
            )
            jobs.append(
                Job(
                    contract,
                    family,
                    learning_rate,
                    TUNING_SEED,
                    ROOT
                    / "experiments/feasibility/final"
                    / contract
                    / family
                    / f"seed{TUNING_SEED}",
                    checkpoint=tuning_checkpoint,
                )
            )
            for seed in FINAL_SEEDS[1:]:
                jobs.append(
                    Job(
                        contract,
                        family,
                        learning_rate,
                        seed,
                        ROOT
                        / "experiments/feasibility/final"
                        / contract
                        / family
                        / f"seed{seed}",
                    )
                )
    return jobs


def smoke_jobs() -> list[Job]:
    return [
        Job(
            contract,
            family,
            LR_GRIDS[family][1],
            TUNING_SEED,
            ROOT / "experiments/feasibility/smoke" / contract / family,
            validation_only=True,
            smoke=True,
        )
        for contract, baseline in BASELINES.items()
        for family in ("hgt", baseline)
    ]


def _run_worker(jobs: list[Job], device: str, force: bool) -> None:
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    for job in jobs:
        result_path = job.output_dir / "result.json"
        if result_path.exists() and not force:
            print(f"SKIP {result_path}", flush=True)
            continue
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


def execute(
    jobs: list[Job],
    devices: list[str],
    force: bool,
    workers_per_device: int,
) -> None:
    if workers_per_device < 1:
        raise ValueError("workers_per_device must be positive")
    device_slots = [device for device in devices for _ in range(workers_per_device)]
    queues = [jobs[index :: len(device_slots)] for index in range(len(device_slots))]
    with ThreadPoolExecutor(max_workers=len(device_slots)) as pool:
        futures = [
            pool.submit(_run_worker, queue, device, force)
            for queue, device in zip(queues, device_slots)
            if queue
        ]
        for future in futures:
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("print", "smoke", "tune", "final"), default="print"
    )
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--workers-per-device", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "print":
        print(
            json.dumps(
                {
                    "profile": "controlled_feasibility",
                    "window_size": 80,
                    "max_epochs": 8,
                    "patience": 2,
                    "tuning_training_jobs": len(tuning_jobs()),
                    "additional_final_training_jobs": 12,
                    "reused_checkpoint_test_jobs": 6,
                    "total_unique_training_jobs": 30,
                    "test_policy": "full test only after validation selection",
                    "final_seeds": FINAL_SEEDS,
                    "hgt_micro_batch_size": 256,
                    "data_loader_workers_per_job": 2,
                    "default_concurrent_jobs_per_gpu": 3,
                },
                indent=2,
            )
        )
        return
    if not ARTIFACT.exists() or not SAMPLE_PLAN.exists():
        raise RuntimeError(
            "Build the feasibility protocol first: "
            "python scripts/build_protocol.py --profile feasibility"
        )
    if args.stage == "smoke":
        jobs = smoke_jobs()
    elif args.stage == "tune":
        jobs = tuning_jobs()
    else:
        jobs = final_jobs(selected_learning_rates())
    execute(jobs, args.devices, args.force, args.workers_per_device)


if __name__ == "__main__":
    main()
