#!/usr/bin/env python
"""Evaluate relation-family removals for the selected semantic HGT checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_benchmark.models import ModelSpec, SEMANTIC_RELATION_FAMILIES, build_model  # noqa: E402
from football_benchmark.protocol import ProtocolArtifacts  # noqa: E402
from football_benchmark.training import TrainingConfig, _loader, evaluate  # noqa: E402


ARTIFACT = ROOT / "artifacts/feasibility/protocol.pt"
SAMPLE_PLAN = ROOT / "artifacts/feasibility/sample_plan.json"
EXPERIMENT_ROOT = ROOT / "experiments/feasibility/semantic_hgt_v2"
OUTPUT = ROOT / "experiments/feasibility/summary_semantic_v2/relation_ablation.json"
PART_ROOT = ROOT / "experiments/feasibility/semantic_hgt_v2/ablation_parts"
CONTRACTS = ("seq2event", "unified_lem", "nmstpp")


def metric_snapshot(metrics: dict) -> dict[str, float | None]:
    return {
        "event_accuracy": metrics["event"]["accuracy"],
        "event_macro_f1": metrics["event"]["macro_f1"],
        "position_distance_mae_m": metrics["position"]["distance_mae_m"],
        "time_mae_seconds": None if metrics["time"] is None else metrics["time"]["mae_seconds"],
    }


def resolve_checkpoint(contract: str, seed: int) -> tuple[Path, Path]:
    """Resolve trained checkpoints, including final runs evaluated from tuning."""
    run_dir = EXPERIMENT_ROOT / "final" / contract / f"seed{seed}"
    final_checkpoint = run_dir / "best.pt"
    if final_checkpoint.exists():
        return run_dir, final_checkpoint

    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"Neither checkpoint nor final result exists for {contract} seed {seed}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    learning_rate = float(result["config"]["learning_rate"])
    tuning_checkpoint = (
        EXPERIMENT_ROOT
        / "tuning"
        / contract
        / f"lr{learning_rate:g}"
        / "best.pt"
    )
    if not tuning_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing tuning checkpoint for {contract} at {tuning_checkpoint}"
        )
    return run_dir, tuning_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--contract", choices=CONTRACTS)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merged = {}
        for contract in CONTRACTS:
            path = PART_ROOT / f"{contract}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            merged.update(json.loads(path.read_text(encoding="utf-8")))
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(OUTPUT), "contracts": list(merged)}, indent=2))
        return
    artifacts = ProtocolArtifacts.load(ARTIFACT)
    result: dict[str, object] = {}
    contracts = (args.contract,) if args.contract else CONTRACTS
    for contract in contracts:
        run_dir, checkpoint = resolve_checkpoint(contract, args.seed)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = state["config"]
        config = TrainingConfig(
            contract=contract,
            family="hgt",
            window_size=80,
            learning_rate=float(source["learning_rate"]),
            seed=args.seed,
            output_dir=run_dir,
            device=args.device,
            max_epochs=int(source["max_epochs"]),
            patience=int(source["patience"]),
            effective_batch_size=256,
            micro_batch_size=256,
            num_workers=2,
            sample_plan_path=SAMPLE_PLAN,
            graph_variant="semantic_v2",
        )
        model = build_model(
            ModelSpec("hgt", contract, 80, graph_variant="semantic_v2"), artifacts
        ).to(torch.device(args.device))
        model.load_state_dict(state["model"])
        loader = _loader("test", config, artifacts, shuffle=False)
        baseline_metrics, _ = evaluate(
            model, loader, config, artifacts, torch.device(args.device)
        )
        baseline = metric_snapshot(baseline_metrics)
        removals = {}
        for family in SEMANTIC_RELATION_FAMILIES:
            loader = _loader("test", config, artifacts, shuffle=False)
            metrics, _ = evaluate(
                model,
                loader,
                config,
                artifacts,
                torch.device(args.device),
                disabled_relation_families=(family,),
            )
            snapshot = metric_snapshot(metrics)
            removals[family] = {
                "metrics": snapshot,
                "delta_minus_full": {
                    name: None
                    if value is None or baseline[name] is None
                    else value - baseline[name]
                    for name, value in snapshot.items()
                },
            }
        result[contract] = {
            "checkpoint": str(checkpoint),
            "full": baseline,
            "removals": removals,
        }
    output = PART_ROOT / f"{args.contract}.json" if args.contract else OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "contracts": list(result)}, indent=2))


if __name__ == "__main__":
    main()
