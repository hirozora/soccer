#!/usr/bin/env python
"""Derive safe per-GPU concurrency from semantic HGT smoke peak memory."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = ROOT / "experiments/feasibility/semantic_hgt_v2/smoke"
OUTPUT = ROOT / "experiments/feasibility/semantic_hgt_v2/resource_plan.json"


def gpu_totals() -> dict[int, int]:
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        int(line.split(",")[0]): int(line.split(",")[1])
        for line in text.splitlines()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--reserve-mib", type=int, default=2048)
    parser.add_argument("--safety-factor", type=float, default=1.5)
    parser.add_argument("--maximum-slots", type=int, default=3)
    args = parser.parse_args()
    result_paths = sorted(SMOKE_ROOT.glob("*/result.json"))
    if len(result_paths) != 3:
        raise RuntimeError(f"Expected three smoke results, found {len(result_paths)}")
    peaks = [
        int(json.loads(path.read_text(encoding="utf-8"))["peak_cuda_memory_bytes"])
        for path in result_paths
    ]
    peak_mib = max(peaks) / (1024**2)
    if peak_mib <= 0:
        raise RuntimeError("Smoke run did not report CUDA memory")
    totals = gpu_totals()
    slots = {
        index: max(
            1,
            min(
                args.maximum_slots,
                int((totals[index] - args.reserve_mib) // (peak_mib * args.safety_factor)),
            ),
        )
        for index in args.gpus
    }
    devices = [
        f"cuda:{index}"
        for slot in range(max(slots.values()))
        for index in args.gpus
        if slot < slots[index]
    ]
    plan = {
        "smoke_peak_bytes": max(peaks),
        "smoke_peak_mib": peak_mib,
        "reserve_mib": args.reserve_mib,
        "safety_factor": args.safety_factor,
        "slots": slots,
        "devices": devices,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(" ".join(devices))


if __name__ == "__main__":
    main()
