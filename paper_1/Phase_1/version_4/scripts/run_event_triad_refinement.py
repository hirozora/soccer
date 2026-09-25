#!/usr/bin/env python
"""Cross-fitted triad calibration, then conditional Head fitting only if needed."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(VERSION_ROOT / "src"), str(VERSION_ROOT.parent / "benchmark_unified_v1/src")]

import torch
from football_hgt_targets_v4.constants import CONFIRMATION_SEEDS
from football_hgt_targets_v4.event_triad import ROOT, MODES, calibrate, train_seed, select, evaluate_test
from football_hgt_targets_v4.event_triad_online import verify_online
from football_hgt_targets_v4.event_triad_reporting import report
from football_hgt_targets_v4.position_head_refit import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("verify", "calibrate", "smoke", "head", "select", "test", "report", "pipeline"))
    parser.add_argument("--seed", type=int, choices=CONFIRMATION_SEEDS)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.stage == "pipeline" and (args.seed is not None or args.mode is not None):
        parser.error("Pipeline requires the complete registered seed/mode matrix")
    if args.resume and (args.stage != "head" or args.seed is None or args.mode is None):
        parser.error("--resume requires --stage head --seed --mode")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    seeds = (args.seed,) if args.seed else CONFIRMATION_SEEDS
    locked = (args.root / "selection/event_triad_lock.json").exists()
    pipeline = args.stage == "pipeline"
    if locked and args.stage in ("calibrate", "smoke", "head"):
        raise RuntimeError("Locked experiment cannot be changed")

    def status(stage):
        write_json(args.root / "status.json", {"stage": stage, "pid": os.getpid()})
        print(stage, flush=True)

    if args.stage == "verify" or (pipeline and not locked):
        status("verify")
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path[:2]), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_event_triad.py"], cwd=VERSION_ROOT, env=env, check=True)
        for seed in seeds:
            verify_online(seed, args.mode or "original", args.root, args.device)
    if args.stage == "calibrate" or (pipeline and not locked):
        status("calibrate")
        calibrate(args.root)
    decision_path = args.root / "calibration/decision.json"
    needs_head = decision_path.exists() and not json.loads(decision_path.read_text())["passed"]
    if args.stage == "smoke" or (pipeline and not locked and needs_head):
        status("smoke")
        train_seed(seeds[0], args.root, args.device, args.mode, smoke=True)
    if args.stage == "head" or (pipeline and not locked and needs_head):
        status("head")
        for seed in seeds:
            train_seed(seed, args.root, args.device, args.mode, args.resume)
            for mode in ((args.mode,) if args.mode else MODES):
                verify_online(seed, mode, args.root, args.device)
    if args.stage == "select" or pipeline:
        status("select_and_online_verification")
        lock = select(args.root)
        print(json.dumps(lock["comparisons"], indent=2), flush=True)
    if args.stage == "test" or pipeline:
        status("test_gate")
        lock = json.loads((args.root / "selection/event_triad_lock.json").read_text())
        if lock["passed"]:
            for seed in seeds:
                evaluate_test(seed, args.root, args.device)
        else:
            print("Original retained. Test reads prohibited.", flush=True)
    if args.stage == "report" or pipeline:
        status("report")
        report(args.root)
        source_files = [Path(__file__), *sorted((VERSION_ROOT / "src/football_hgt_targets_v4").glob("event_triad*.py")),
                        VERSION_ROOT / "tests/test_event_triad.py"]
        recorded = json.loads(decision_path.read_text())["provenance"]
        hashes = {p: h for seed_sources in recorded.values() for p, h in seed_sources["sources"].items()}
        for checkpoint in sorted((args.root / "training").glob("*/seed*/best_event.pt")):
            provenance = torch.load(checkpoint, map_location="cpu", weights_only=False)["provenance"]
            for population in provenance.values():
                for p, h in population["sources"].items():
                    if p in hashes and hashes[p] != h:
                        raise RuntimeError(f"Input changed between stages: {p}")
                    hashes[p] = h
        for path, old_hash in hashes.items():
            if sha256(Path(path)) != old_hash:
                raise RuntimeError(f"Source changed: {path}")
        write_json(args.root / "manifest.json", {"source_inputs_unchanged": True, "sources": hashes,
            "code": {str(p): sha256(p) for p in source_files}, "python": sys.version, "torch": torch.__version__,
            "device": args.device, "threads": 1, "test_previously_seen": True,
            "outputs": {str(p): sha256(p) for p in sorted(args.root.rglob("*"))
                        if p.is_file() and p.name not in ("manifest.json", "status.json")}})
        status("completed")
        print(args.root / "report/README.md", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
