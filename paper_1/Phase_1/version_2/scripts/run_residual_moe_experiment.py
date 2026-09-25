#!/usr/bin/env python3
"""Train Version 3 full-history dense structural residual experts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader, Subset


VERSION_ROOT = Path(__file__).resolve().parents[1]
PHASE_ROOT = VERSION_ROOT.parent
V1_ROOT = PHASE_ROOT / "version_1"
sys.path.insert(0, str(PHASE_ROOT / "src"))
sys.path.insert(0, str(V1_ROOT / "src"))
sys.path.insert(0, str(VERSION_ROOT / "src"))

from football_hgt.dataset import MatchGraphRecord  # noqa: E402
from football_hgt_residual_v3.data import (  # noqa: E402
    OVERLAP_KEYS,
    FullHistoryResidualDataset,
    collate_full_history_residuals,
)
from football_hgt_residual_v3.experts import GraphViewConfig  # noqa: E402
from football_hgt_residual_v3.model import (  # noqa: E402
    FullHistoryResidualMoE,
    ModelConfig,
    ResidualMoEConfig,
)
from football_hgt_residual_v3.training import run_epoch  # noqa: E402
from football_hgt_v1.training import set_seed  # noqa: E402


DEFAULT_GRAPH_ROOT = PHASE_ROOT / "data/whyscout/processed/heterogeneous_graphs/v1"
DEFAULT_SPLIT = V1_ROOT / "data_splits/temporal_match_split_v1.csv"
DEFAULT_VOCAB = DEFAULT_GRAPH_ROOT / "metadata/vocabularies.json"
DEFAULT_OUTPUT = VERSION_ROOT / "experiments/controlled_v1"
DEFAULT_V1_RESULT = V1_ROOT / "experiments/window_ablation_stage2/k_80.json"
DEFAULT_V1_CHECKPOINT = V1_ROOT / "experiments/window_ablation_stage2/k_80.pt"
DEFAULT_BASELINE_SUMMARY = V1_ROOT / "experiments/baseline_comparison_summary_v1.json"


def select_evenly(values: list, count: int) -> list:
    if count >= len(values):
        return values
    if count == 1:
        return [values[len(values) // 2]]
    indices = [round(i * (len(values) - 1) / (count - 1)) for i in range(count)]
    return [values[index] for index in indices]


def load_records(
    split_path: Path,
    graph_root: Path,
    competition: str,
    split: str,
    max_matches: int,
) -> list[MatchGraphRecord]:
    with split_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["competition_slug"] == competition and row["split"] == split
        ]
    rows = select_evenly(rows, max_matches)
    return [
        MatchGraphRecord(
            competition_slug=row["competition_slug"],
            match_id=int(row["match_id"]),
            graph_path=graph_root / row["graph_path"],
            num_events=int(row["num_events"]),
        )
        for row in rows
    ]


def selected_step_indices(
    records: list[MatchGraphRecord], steps_per_match: int
) -> list[int]:
    selected = []
    offset = 0
    for record in records:
        steps = record.num_events - 1
        count = min(steps, steps_per_match)
        local = (
            [0]
            if count == 1
            else [round(i * (steps - 1) / (count - 1)) for i in range(count)]
        )
        selected.extend(offset + index for index in local)
        offset += steps
    return selected


def make_loader(
    records: list[MatchGraphRecord],
    view_config: GraphViewConfig,
    steps_per_match: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = FullHistoryResidualDataset(
        records,
        view_config=view_config,
        cache_size=len(records),
    )
    subset = Subset(dataset, selected_step_indices(records, steps_per_match))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_full_history_residuals,
        pin_memory=torch.cuda.is_available(),
    )


def metric_delta(candidate: dict, reference: dict, name: str) -> dict[str, float]:
    return {
        "version_3": float(candidate[name]),
        "version_1_k80": float(reference[name]),
        "absolute_difference": float(candidate[name]) - float(reference[name]),
    }


def build_comparison(
    result: dict,
    v1_result_path: Path,
    baseline_summary_path: Path,
) -> dict:
    candidate = result["best_epoch"]["validation"]
    candidate_protocol = {
        key: result["config"][key]
        for key in (
            "competition",
            "train_matches",
            "validation_matches",
            "steps_per_match",
            "epochs",
            "seed",
        )
    }
    comparison = {
        "candidate_protocol": candidate_protocol,
        "version_3_result": result["result_path"],
        "version_1_result": str(v1_result_path.resolve()),
    }
    if v1_result_path.exists():
        v1 = json.loads(v1_result_path.read_text(encoding="utf-8"))
        reference = v1["best_epoch"]["validation"]
        metrics = (
            "event_macro_f1",
            "event_accuracy",
            "time_mae_seconds",
            "position_euclidean_distance",
            "side_accuracy",
            "player_accuracy",
            "player_top3_accuracy",
            "advantage_accuracy",
        )
        comparison["against_version_1_k80"] = {
            name: metric_delta(candidate, reference, name) for name in metrics
        }
    if baseline_summary_path.exists():
        baseline = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        shared_protocol = baseline["shared_protocol"]
        protocol_matches = all(
            candidate_protocol[key] == shared_protocol[key]
            for key in candidate_protocol
        )
        candidate_row = {
            "model": "full_history_structural_residual_moe_v3",
            "seed": result["config"]["seed"],
            "sequence_length": result["config"]["view_config"]["full_history"],
            "num_parameters": result["num_parameters"],
            "event_macro_f1": candidate["event_macro_f1"],
            "event_accuracy": candidate["event_accuracy"],
            "time_mae_seconds": candidate["time_mae_seconds"],
            "position_euclidean_distance": candidate[
                "position_euclidean_distance"
            ],
        }
        rows = [candidate_row, *baseline["results"]]
        comparison.update(
            {
                "baseline_summary": str(baseline_summary_path.resolve()),
                "reference_protocol": shared_protocol,
                "protocol_matches_reference": protocol_matches,
                "comparison_is_controlled": protocol_matches,
                "ranking_by_event_macro_f1": [
                    row["model"]
                    for row in sorted(
                        rows, key=lambda row: row["event_macro_f1"], reverse=True
                    )
                ],
                "results": rows,
            }
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default="England")
    parser.add_argument("--train-matches", type=int, default=48)
    parser.add_argument("--validation-matches", type=int, default=12)
    parser.add_argument("--steps-per-match", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--full-history-window", type=int, default=80)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--adapter-channels", type=int, default=16)
    parser.add_argument("--router-hidden-channels", type=int, default=128)
    parser.add_argument("--task-embedding-channels", type=int, default=16)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument(
        "--freeze-base", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v1-result", type=Path, default=DEFAULT_V1_RESULT)
    parser.add_argument(
        "--base-checkpoint", type=Path, default=DEFAULT_V1_CHECKPOINT
    )
    parser.add_argument(
        "--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.full_history_window < 1:
        parser.error("epochs and full-history window must be positive")
    if args.router_temperature <= 0.0:
        parser.error("router temperature must be positive")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    train_records = load_records(
        args.split_path, args.graph_root, args.competition, "train", args.train_matches
    )
    validation_records = load_records(
        args.split_path,
        args.graph_root,
        args.competition,
        "validation",
        args.validation_matches,
    )
    view_config = GraphViewConfig(full_history=args.full_history_window)
    train_loader = make_loader(
        train_records,
        view_config,
        args.steps_per_match,
        args.batch_size,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_records,
        view_config,
        args.steps_per_match,
        args.batch_size,
        shuffle=False,
    )

    vocab = json.loads(args.vocab_path.read_text(encoding="utf-8"))
    base_config = ModelConfig(
        num_event_types=len(vocab["event_type_ids"]),
        num_subevent_types=len(vocab["subevent_type_ids"]),
        num_players=len(vocab["player_ids"]),
        num_teams=len(vocab["team_ids"]),
        num_tags=len(vocab["tag_ids"]),
        hidden_channels=args.hidden_channels,
    )
    residual_config = ResidualMoEConfig(
        base=base_config,
        adapter_channels=args.adapter_channels,
        task_embedding_channels=args.task_embedding_channels,
        router_hidden_channels=args.router_hidden_channels,
        router_temperature=args.router_temperature,
    )
    model = FullHistoryResidualMoE(residual_config).to(device)
    base_checkpoint = torch.load(
        args.base_checkpoint, map_location="cpu", weights_only=True
    )
    checkpoint_config = base_checkpoint.get("model_config")
    if checkpoint_config != asdict(base_config):
        raise ValueError(
            "K=80 checkpoint ModelConfig does not match the Version 3 base: "
            f"{checkpoint_config!r} != {asdict(base_config)!r}"
        )
    model.base.load_state_dict(base_checkpoint["model_state_dict"])
    model.set_base_trainable(not args.freeze_base)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    started = perf_counter()
    initial_validation = run_epoch(
        model,
        validation_loader,
        device,
        base_config.num_event_types,
        OVERLAP_KEYS,
    )
    v1_document = json.loads(args.v1_result.read_text(encoding="utf-8"))
    reference = v1_document["best_epoch"]["validation"]
    reference_config = v1_document["config"]
    equivalence_metrics = (
        "event_macro_f1",
        "event_accuracy",
        "time_mae_seconds",
        "position_euclidean_distance",
        "side_accuracy",
        "player_accuracy",
        "player_top3_accuracy",
        "advantage_accuracy",
    )
    protocol_matches_k80 = (
        args.competition == reference_config["competition"]
        and args.train_matches == reference_config["train_matches"]
        and args.validation_matches == reference_config["validation_matches"]
        and args.steps_per_match == reference_config["steps_per_match"]
        and args.full_history_window == reference_config["window_size"]
        and [record.match_id for record in train_records]
        == reference_config["train_match_ids"]
        and [record.match_id for record in validation_records]
        == reference_config["validation_match_ids"]
    )
    metric_differences = (
        {
            name: float(initial_validation[name]) - float(reference[name])
            for name in equivalence_metrics
        }
        if protocol_matches_k80
        else {}
    )
    max_equivalence_error = max(
        [abs(value) for value in metric_differences.values()] or [0.0]
    )
    internal_event_error = abs(
        float(initial_validation["residual_event_macro_f1_gain"])
    )
    if internal_event_error > 1e-12:
        raise RuntimeError(
            "Zero-residual output differs from its complete-history branch; "
            f"Event Macro-F1 difference is {internal_event_error}"
        )
    if protocol_matches_k80 and max_equivalence_error > 1e-6:
        raise RuntimeError(
            "Zero-residual Version 3 does not reproduce the K=80 checkpoint; "
            f"maximum metric difference is {max_equivalence_error}"
        )
    history = [
        {
            "epoch": 0,
            "stage": "k80_equivalence",
            "train": None,
            "validation": initial_validation,
        }
    ]
    best_macro_f1 = initial_validation["event_macro_f1"]
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    print(
        f"epoch=0 val_event_f1={initial_validation['event_macro_f1']:.4f} "
        f"k80_reference_checked={protocol_matches_k80} "
        f"max_k80_metric_error={max_equivalence_error:.3e}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            base_config.num_event_types,
            OVERLAP_KEYS,
            optimizer,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            base_config.num_event_types,
            OVERLAP_KEYS,
        )
        row = {
            "epoch": epoch,
            "stage": "dense_residual_training",
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_event_f1={validation_metrics['event_macro_f1']:.4f} "
            f"val_base_f1={validation_metrics['base_event_macro_f1']:.4f} "
            f"val_residual_gain={validation_metrics['residual_event_macro_f1_gain']:+.4f}",
            flush=True,
        )
        if validation_metrics["event_macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["event_macro_f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    elapsed = perf_counter() - started
    best_epoch = max(
        history, key=lambda row: row["validation"]["event_macro_f1"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    checkpoint_path = args.output_dir / "checkpoint.pt"
    comparison_path = args.output_dir / "comparison_with_version_1.json"
    result = {
        "experiment": "full_history_structural_residual_moe_v3",
        "selection_metric": "validation.event_macro_f1",
        "status": "provisional_single_seed_controlled_comparison",
        "result_path": str(result_path.resolve()),
        "config": {
            **vars(args),
            "graph_root": str(args.graph_root.resolve()),
            "split_path": str(args.split_path.resolve()),
            "vocab_path": str(args.vocab_path.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "v1_result": str(args.v1_result.resolve()),
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "baseline_summary": str(args.baseline_summary.resolve()),
            "base_model": asdict(base_config),
            "residual_moe": {
                key: value
                for key, value in asdict(residual_config).items()
                if key != "base"
            },
            "view_config": asdict(view_config),
            "train_match_ids": [record.match_id for record in train_records],
            "validation_match_ids": [
                record.match_id for record in validation_records
            ],
        },
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_trainable_parameters": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "initial_k80_equivalence": {
            "reference_protocol_matched": protocol_matches_k80,
            "internal_event_macro_f1_difference": internal_event_error,
            "metric_differences": metric_differences,
            "maximum_absolute_difference": max_equivalence_error,
        },
        "elapsed_seconds": elapsed,
        "best_epoch": best_epoch,
        "history": history,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "base_model_config": asdict(base_config),
            "residual_moe_config": {
                key: value
                for key, value in asdict(residual_config).items()
                if key != "base"
            },
            "view_config": asdict(view_config),
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "base_frozen": args.freeze_base,
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )
    comparison = build_comparison(result, args.v1_result, args.baseline_summary)
    comparison_path.write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "checkpoint": str(checkpoint_path),
                "comparison": str(comparison_path),
                "best_validation": best_epoch["validation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
