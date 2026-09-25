#!/usr/bin/env python
"""Tune and evaluate only semantic_v2 HGT across the three contracts."""

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
EXPERIMENT_ROOT = ROOT / "experiments/feasibility/semantic_hgt_v2"
CONTRACTS = ("seq2event", "unified_lem", "nmstpp")
LEARNING_RATES = (1e-4, 3e-4, 9e-4)
TUNING_SEED = 20260715
FINAL_SEEDS = (20260715, 20260716, 20260717)


@dataclass(frozen=True)
class Job:
    contract: str
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
            self.contract,
            "--model",
            "hgt",
            "--graph-variant",
            "semantic_v2",
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
            "--effective-batch-size",
            "256",
            "--micro-batch-size",
            "256",
            "--num-workers",
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
        if self.checkpoint is not None:
            command.extend(("--checkpoint", str(self.checkpoint)))
        if self.smoke:
            command.append("--smoke")
        return command


def tuning_jobs() -> list[Job]:
    return [
        Job(
            contract,
            learning_rate,
            TUNING_SEED,
            EXPERIMENT_ROOT / "tuning" / contract / f"lr{learning_rate:g}",
            validation_only=True,
        )
        for contract in CONTRACTS
        for learning_rate in LEARNING_RATES
    ]


def selected_learning_rates() -> dict[str, float]:
    selected: dict[str, float] = {}
    losses: dict[str, dict[str, float]] = {}
    for contract in CONTRACTS:
        candidates: dict[str, float] = {}
        for learning_rate in LEARNING_RATES:
            path = (
                EXPERIMENT_ROOT
                / "tuning"
                / contract
                / f"lr{learning_rate:g}"
                / "result.json"
            )
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                candidates[str(learning_rate)] = float(result["validation"]["loss"])
        if len(candidates) != len(LEARNING_RATES):
            raise RuntimeError(f"Incomplete semantic HGT tuning for {contract}: {candidates}")
        selected[contract] = float(min(candidates, key=candidates.get))
        losses[contract] = candidates
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_ROOT / "selected_learning_rates.json").write_text(
        json.dumps({"selected": selected, "validation_losses": losses}, indent=2),
        encoding="utf-8",
    )
    return selected


def final_jobs(learning_rates: dict[str, float]) -> list[Job]:
    jobs: list[Job] = []
    for contract in CONTRACTS:
        learning_rate = learning_rates[contract]
        tuning_checkpoint = (
            EXPERIMENT_ROOT
            / "tuning"
            / contract
            / f"lr{learning_rate:g}"
            / "best.pt"
        )
        for seed in FINAL_SEEDS:
            jobs.append(
                Job(
                    contract,
                    learning_rate,
                    seed,
                    EXPERIMENT_ROOT / "final" / contract / f"seed{seed}",
                    checkpoint=tuning_checkpoint if seed == TUNING_SEED else None,
                )
            )
    return jobs


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
                    "graph_variant": "semantic_v2",
                    "contracts": CONTRACTS,
                    "learning_rates": LEARNING_RATES,
                    "tuning_jobs": 9,
                    "additional_training_jobs": 6,
                    "checkpoint_evaluations": 3,
                    "max_epochs": 8,
                    "patience": 2,
                },
                indent=2,
            )
        )
        return
    if args.stage == "smoke":
        jobs = [
            Job(
                contract,
                3e-4,
                TUNING_SEED,
                EXPERIMENT_ROOT / "smoke" / contract,
                validation_only=True,
                smoke=True,
            )
            for contract in CONTRACTS
        ]
    elif args.stage == "tune":
        jobs = tuning_jobs()
    else:
        jobs = final_jobs(selected_learning_rates())
    execute(jobs, args.devices, args.force)


if __name__ == "__main__":
    main()
