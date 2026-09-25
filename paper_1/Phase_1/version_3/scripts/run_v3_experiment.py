#!/usr/bin/env python3
"""Run one controlled Version 3 data-aware residual MoE experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset


VERSION_ROOT = Path(__file__).resolve().parents[1]
PHASE_ROOT = VERSION_ROOT.parent
V1_ROOT = PHASE_ROOT / "version_1"
sys.path.insert(0, str(PHASE_ROOT / "src"))
sys.path.insert(0, str(V1_ROOT / "src"))
sys.path.insert(0, str(VERSION_ROOT / "src"))

from football_hgt.dataset import MatchGraphRecord  # noqa: E402
from football_hgt_v1.model import ModelConfig  # noqa: E402
from football_hgt_v1.training import set_seed  # noqa: E402
from football_hgt_v3.data import (  # noqa: E402
    DataAwareWindowDataset,
    collate_data_aware_windows,
)
from football_hgt_v3.experts import EXPERT_NAMES, SelectionConfig  # noqa: E402
from football_hgt_v3.model import (  # noqa: E402
    DataAwareMoEConfig,
    DataAwareResidualMoE,
)
from football_hgt_v3.training import move_payload, run_epoch  # noqa: E402


DEFAULT_GRAPH_ROOT = PHASE_ROOT / "data/whyscout/processed/heterogeneous_graphs/v1"
DEFAULT_SPLIT = V1_ROOT / "data_splits/temporal_match_split_v1.csv"
DEFAULT_VOCAB = DEFAULT_GRAPH_ROOT / "metadata/vocabularies.json"
DEFAULT_CHECKPOINT = V1_ROOT / "experiments/window_ablation_stage2/k_80.pt"
DEFAULT_OUTPUT = VERSION_ROOT / "experiments/controlled"
STAGES = ("E0", "E1-S", "E1-T", "E2", "E3", "E4", "E5-POS", "E5-PLAYER", "E6")
CORE_STAGES = ("E2", "E3", "E4")


def select_evenly(values: list[Any], count: int) -> list[Any]:
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
    selected: list[int] = []
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
    selection_config: SelectionConfig,
    sampling: str,
    steps_per_match: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = DataAwareWindowDataset(
        records,
        selection_config=selection_config,
        cache_size=len(records),
    )
    source = (
        dataset
        if sampling == "all"
        else Subset(dataset, selected_step_indices(records, steps_per_match))
    )
    return DataLoader(
        source,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_data_aware_windows,
        pin_memory=torch.cuda.is_available(),
    )


def stage_configuration(stage: str, core_stage: str) -> dict[str, Any]:
    core = {
        "E2": {
            "expert_mode": "homogeneous",
            "residual_mode": "task_specific",
            "routing_mode": "dense",
            "conditioned_router": False,
        },
        "E3": {
            "expert_mode": "homogeneous",
            "residual_mode": "task_specific",
            "routing_mode": "dense",
            "conditioned_router": True,
        },
        "E4": {
            "expert_mode": "structural",
            "residual_mode": "task_specific",
            "routing_mode": "dense",
            "conditioned_router": True,
        },
    }
    if stage == "E0":
        result = {
            "expert_mode": "homogeneous",
            "residual_mode": "task_specific",
            "routing_mode": "uniform",
            "conditioned_router": False,
        }
    elif stage == "E1-S":
        result = {
            "expert_mode": "homogeneous",
            "residual_mode": "shared",
            "routing_mode": "uniform",
            "conditioned_router": False,
        }
    elif stage == "E1-T":
        result = {
            "expert_mode": "homogeneous",
            "residual_mode": "task_specific",
            "routing_mode": "uniform",
            "conditioned_router": False,
        }
    elif stage in core:
        result = dict(core[stage])
    else:
        result = dict(core[core_stage])
    result["position_correction"] = stage in {"E5-POS", "E6"}
    result["player_correction"] = stage in {"E5-PLAYER", "E6"}
    return result


def is_feasible(candidate: dict[str, Any], reference: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "time_mae": candidate["time_mae_seconds"] <= reference["time_mae_seconds"] + 0.05,
        "position_distance": candidate["position_euclidean_distance"]
        <= reference["position_euclidean_distance"] + 0.5,
        "side_accuracy": candidate["side_accuracy"] >= reference["side_accuracy"] - 0.01,
        "player_top3": candidate["player_top3_accuracy"]
        >= reference["player_top3_accuracy"] - 0.01,
        "advantage_accuracy": candidate["advantage_accuracy"]
        >= reference["advantage_accuracy"] - 0.01,
    }
    return all(checks.values()), checks


def output_equivalence(
    model: DataAwareResidualMoE,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    raw = next(iter(loader))
    payload = move_payload(raw, device)
    graph = payload["graph"]
    with torch.inference_mode():
        candidate = model(graph, payload["selections"], payload["candidate_features"])
        reference = model.base(graph)
    differences = {
        name: float((candidate[name] - reference[name]).abs().max())
        for name in (
            "event_logits",
            "log_delta",
            "position",
            "side_logits",
            "player_scores",
            "advantage_logits",
        )
    }
    model.hgt_forward_calls = 0
    return differences


def event_labels(graph_root: Path) -> list[dict[str, Any]]:
    vocab = json.loads((graph_root / "metadata/vocabularies.json").read_text(encoding="utf-8"))
    mapping: dict[int, str] = {}
    with (PHASE_ROOT / "data/whyscout/raw/mappings/eventid2name.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            mapping[int(row["event"])] = row["event_label"]
    return [
        {"class_index": index, "raw_event_id": raw_id, "name": mapping[raw_id]}
        for index, raw_id in enumerate(vocab["event_type_ids"])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--core-stage", choices=CORE_STAGES, default="E4")
    parser.add_argument("--competition", default="England")
    parser.add_argument("--train-matches", type=int, default=48)
    parser.add_argument("--validation-matches", type=int, default=12)
    parser.add_argument("--train-steps-per-match", type=int, default=96)
    parser.add_argument("--validation-steps-per-match", type=int, default=96)
    parser.add_argument("--validation-sampling", choices=("evenly", "all"), default="evenly")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--leave-one-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.epochs < 0 or args.train_steps_per_match < 1:
        parser.error("epochs must be non-negative and training steps must be positive")

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
    selection_config = SelectionConfig(full_history=80)
    train_loader = make_loader(
        train_records,
        selection_config,
        "evenly",
        args.train_steps_per_match,
        args.batch_size,
        True,
    )
    validation_loader = make_loader(
        validation_records,
        selection_config,
        args.validation_sampling,
        args.validation_steps_per_match,
        args.batch_size,
        False,
    )

    vocab = json.loads(args.vocab_path.read_text(encoding="utf-8"))
    base_config = ModelConfig(
        num_event_types=len(vocab["event_type_ids"]),
        num_subevent_types=len(vocab["subevent_type_ids"]),
        num_players=len(vocab["player_ids"]),
        num_teams=len(vocab["team_ids"]),
        num_tags=len(vocab["tag_ids"]),
        hidden_channels=64,
    )
    model_options = stage_configuration(args.stage, args.core_stage)
    model_config = DataAwareMoEConfig(base=base_config, **model_options)
    model = DataAwareResidualMoE(model_config).to(device)
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_config") != asdict(base_config):
        raise ValueError("V1 checkpoint configuration does not match Version 3")
    model.base.load_state_dict(checkpoint["model_state_dict"])
    model.freeze_base()

    equivalence = output_equivalence(model, validation_loader, device)
    maximum_equivalence_error = max(equivalence.values())
    if maximum_equivalence_error >= 1e-6:
        raise RuntimeError(
            f"Version 3 initialization does not equal V1: {maximum_equivalence_error}"
        )

    started = perf_counter()
    initial = run_epoch(model, validation_loader, device, base_config.num_event_types)
    history = [
        {
            "epoch": 0,
            "stage": "v1_equivalence",
            "feasible": True,
            "constraint_checks": {},
            "train": None,
            "validation": initial,
        }
    ]
    best_epoch = history[0]
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    diagnostic_best_epoch = None
    diagnostic_best_state = None
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    epochs = 0 if args.stage == "E0" else args.epochs
    print(
        f"stage={args.stage} epoch=0 val_event_f1={initial['event_macro_f1']:.4f} "
        f"equivalence={maximum_equivalence_error:.3e}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, base_config.num_event_types, optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, device, base_config.num_event_types
        )
        feasible, checks = is_feasible(validation_metrics, initial)
        row = {
            "epoch": epoch,
            "stage": "residual_training",
            "feasible": feasible,
            "constraint_checks": checks,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        if (
            diagnostic_best_epoch is None
            or validation_metrics["event_macro_f1"]
            > diagnostic_best_epoch["validation"]["event_macro_f1"]
        ):
            diagnostic_best_epoch = row
            diagnostic_best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if feasible and validation_metrics["event_macro_f1"] > best_epoch["validation"]["event_macro_f1"]:
            best_epoch = row
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        print(
            f"stage={args.stage} epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_f1={validation_metrics['event_macro_f1']:.4f} feasible={feasible}",
            flush=True,
        )

    model.load_state_dict(best_state)
    model.to(device)
    leave_one_out = {}
    if args.leave_one_out:
        for index, name in enumerate(EXPERT_NAMES):
            leave_one_out[name] = run_epoch(
                model,
                validation_loader,
                device,
                base_config.num_event_types,
                disabled_expert=index,
            )
    diagnostic_leave_one_out = {}
    if (
        args.leave_one_out
        and diagnostic_best_epoch is not None
        and diagnostic_best_state is not None
        and diagnostic_best_epoch["epoch"] != best_epoch["epoch"]
    ):
        model.load_state_dict(diagnostic_best_state)
        for index, name in enumerate(EXPERT_NAMES):
            diagnostic_leave_one_out[name] = run_epoch(
                model,
                validation_loader,
                device,
                base_config.num_event_types,
                disabled_expert=index,
            )
        model.load_state_dict(best_state)

    output_dir = args.output_root / args.stage.lower().replace("-", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    diagnostic_checkpoint_path = output_dir / "diagnostic_best_trained.pt"
    result = {
        "experiment": "data_aware_single_hgt_residual_moe_v3",
        "stage": args.stage,
        "status": "controlled_single_seed" if args.validation_sampling == "evenly" else "full_validation_single_seed",
        "selection_policy": {
            "primary": "validation.event_macro_f1",
            "epoch_zero_is_candidate": True,
            "constraints": {
                "time_mae_max_increase_seconds": 0.05,
                "position_distance_max_increase": 0.5,
                "side_accuracy_max_decrease": 0.01,
                "player_top3_max_decrease": 0.01,
                "advantage_accuracy_max_decrease": 0.01,
            },
        },
        "config": {
            **vars(args),
            "graph_root": str(args.graph_root.resolve()),
            "split_path": str(args.split_path.resolve()),
            "vocab_path": str(args.vocab_path.resolve()),
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "output_root": str(args.output_root.resolve()),
            "base_model": asdict(base_config),
            "model": {key: value for key, value in asdict(model_config).items() if key != "base"},
            "selection": asdict(selection_config),
            "train_match_ids": [record.match_id for record in train_records],
            "validation_match_ids": [record.match_id for record in validation_records],
        },
        "event_labels": event_labels(args.graph_root),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "initial_equivalence": {
            "maximum_absolute_output_difference": maximum_equivalence_error,
            "output_differences": equivalence,
        },
        "reference_epoch_zero": initial,
        "best_epoch": best_epoch,
        "leave_one_expert_out": leave_one_out,
        "diagnostic_best_trained_epoch": diagnostic_best_epoch,
        "diagnostic_leave_one_expert_out": diagnostic_leave_one_out,
        "elapsed_seconds": perf_counter() - started,
        "history": history,
        "result_path": str(result_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "base_model_config": asdict(base_config),
            "model_config": {key: value for key, value in asdict(model_config).items() if key != "base"},
            "selection_config": asdict(selection_config),
            "stage": args.stage,
            "best_epoch": best_epoch["epoch"],
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )
    if diagnostic_best_state is not None and diagnostic_best_epoch is not None:
        torch.save(
            {
                "base_model_config": asdict(base_config),
                "model_config": {
                    key: value for key, value in asdict(model_config).items() if key != "base"
                },
                "selection_config": asdict(selection_config),
                "stage": args.stage,
                "diagnostic_best_trained_epoch": diagnostic_best_epoch["epoch"],
                "is_primary_selected_checkpoint": diagnostic_best_epoch["epoch"]
                == best_epoch["epoch"],
                "base_checkpoint": str(args.base_checkpoint.resolve()),
                "model_state_dict": diagnostic_best_state,
            },
            diagnostic_checkpoint_path,
        )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "checkpoint": str(checkpoint_path),
                "best_epoch": best_epoch["epoch"],
                "best_validation": best_epoch["validation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
