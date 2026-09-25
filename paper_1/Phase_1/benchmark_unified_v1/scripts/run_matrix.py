#!/usr/bin/env python
"""Print or execute the controlled LR-tuning and five-seed experiment matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_experiment.py"

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
NATIVE_WINDOWS = {"seq2event": 40, "unified_lem": 3, "nmstpp": 40}
SEEDS = tuple(range(20260715, 20260720))


@dataclass(frozen=True)
class Job:
    contract: str
    family: str
    window: int
    learning_rate: float
    seed: int
    output_dir: Path
    validation_only: bool = False
    smoke: bool = False

    def command(self, device: str) -> list[str]:
        command = [
            sys.executable,
            str(RUNNER),
            "--contract",
            self.contract,
            "--model",
            self.family,
            "--window-size",
            str(self.window),
            "--learning-rate",
            str(self.learning_rate),
            "--seed",
            str(self.seed),
            "--device",
            device,
            "--output-dir",
            str(self.output_dir),
        ]
        if self.validation_only:
            command.append("--validation-only")
        if self.smoke:
            command.append("--smoke")
        return command


def tuning_jobs() -> list[Job]:
    jobs: list[Job] = []
    for contract, baseline in BASELINES.items():
        for family in ("hgt", baseline):
            for learning_rate in LR_GRIDS[family]:
                jobs.append(
                    Job(
                        contract,
                        family,
                        80,
                        learning_rate,
                        20260715,
                        ROOT
                        / "experiments/tuning"
                        / contract
                        / family
                        / f"lr{learning_rate:g}",
                        validation_only=True,
                    )
                )
    return jobs


def selected_learning_rates() -> dict[str, float]:
    selected: dict[str, float] = {}
    details: dict[str, dict[str, float]] = {}
    for contract, baseline in BASELINES.items():
        for family in ("hgt", baseline):
            candidates: dict[str, float] = {}
            for learning_rate in LR_GRIDS[family]:
                path = (
                    ROOT
                    / "experiments/tuning"
                    / contract
                    / family
                    / f"lr{learning_rate:g}"
                    / "result.json"
                )
                if path.exists():
                    result = json.loads(path.read_text(encoding="utf-8"))
                    candidates[str(learning_rate)] = float(result["validation"]["loss"])
            if len(candidates) != len(LR_GRIDS[family]):
                raise RuntimeError(
                    f"Incomplete tuning results for {contract}/{family}: {candidates}"
                )
            best = min(candidates, key=candidates.get)
            selected[f"{contract}/{family}"] = float(best)
            details[f"{contract}/{family}"] = candidates
    payload = {"selected": selected, "validation_losses": details}
    path = ROOT / "experiments/selected_learning_rates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return selected


def final_jobs(selected: dict[str, float]) -> list[Job]:
    jobs: list[Job] = []
    for contract, baseline in BASELINES.items():
        for window in (80, NATIVE_WINDOWS[contract]):
            for family in ("hgt", baseline):
                for seed in SEEDS:
                    jobs.append(
                        Job(
                            contract,
                            family,
                            window,
                            selected[f"{contract}/{family}"],
                            seed,
                            ROOT
                            / "experiments/final"
                            / contract
                            / f"k{window}"
                            / family
                            / f"seed{seed}",
                        )
                    )
    return jobs


def smoke_jobs() -> list[Job]:
    defaults = {name: values[1] for name, values in LR_GRIDS.items()}
    return [
        Job(
            contract,
            family,
            80,
            defaults[family],
            20260715,
            ROOT / "experiments/smoke" / contract / family,
            smoke=True,
        )
        for contract, baseline in BASELINES.items()
        for family in ("hgt", baseline)
    ]


def _run_worker(jobs: list[Job], device: str, force: bool) -> None:
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
            )


def execute(jobs: list[Job], devices: list[str], force: bool) -> None:
    queues = [jobs[index :: len(devices)] for index in range(len(devices))]
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [
            pool.submit(_run_worker, queue, device, force)
            for queue, device in zip(queues, devices)
        ]
        for future in futures:
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["print", "smoke", "tune", "final"], default="print")
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "print":
        payload = {
            "tuning_jobs": len(tuning_jobs()),
            "final_jobs": 60,
            "tuning": [job.__dict__ | {"output_dir": str(job.output_dir)} for job in tuning_jobs()],
        }
        print(json.dumps(payload, indent=2))
        return
    if args.stage == "smoke":
        jobs = smoke_jobs()
    elif args.stage == "tune":
        jobs = tuning_jobs()
    else:
        jobs = final_jobs(selected_learning_rates())
    execute(jobs, args.devices, args.force)


if __name__ == "__main__":
    main()

