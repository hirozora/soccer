#!/usr/bin/env python3
"""Train one controlled HGT window-length experiment."""

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
sys.path.insert(0, str(PHASE_ROOT / "src"))
sys.path.insert(0, str(VERSION_ROOT / "src"))

from football_hgt.dataset import FixedWindowDataset, MatchGraphRecord  # noqa: E402
from football_hgt_v1.data import collate_fixed_windows  # noqa: E402
from football_hgt_v1.model import FootballHGT, ModelConfig  # noqa: E402
from football_hgt_v1.training import run_epoch, set_seed  # noqa: E402


DEFAULT_GRAPH_ROOT = PHASE_ROOT / "data/whyscout/processed/heterogeneous_graphs/v1"
DEFAULT_SPLIT = VERSION_ROOT / "data_splits/temporal_match_split_v1.csv"
DEFAULT_VOCAB = DEFAULT_GRAPH_ROOT / "metadata/vocabularies.json"
DEFAULT_OUTPUT = VERSION_ROOT / "experiments/window_ablation"


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
        if count == 1:
            local = [0]
        else:
            local = [round(i * (steps - 1) / (count - 1)) for i in range(count)]
        selected.extend(offset + index for index in local)
        offset += steps
    return selected


def make_loader(
    records: list[MatchGraphRecord],
    window_size: int,
    steps_per_match: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = FixedWindowDataset(
        records,
        window_size=window_size,
        cache_size=len(records),
        validate_graph_on_load=False,
        validate_samples=False,
    )
    subset = Subset(dataset, selected_step_indices(records, steps_per_match))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fixed_windows,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--competition", default="England")
    parser.add_argument("--train-matches", type=int, default=24)
    parser.add_argument("--validation-matches", type=int, default=8)
    parser.add_argument("--steps-per-match", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.window_size < 1:
        parser.error("--window-size must be positive")

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
    train_loader = make_loader(
        train_records,
        args.window_size,
        args.steps_per_match,
        args.batch_size,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_records,
        args.window_size,
        args.steps_per_match,
        args.batch_size,
        shuffle=False,
    )

    vocab = json.loads(args.vocab_path.read_text(encoding="utf-8"))
    config = ModelConfig(
        num_event_types=len(vocab["event_type_ids"]),
        num_subevent_types=len(vocab["subevent_type_ids"]),
        num_players=len(vocab["player_ids"]),
        num_teams=len(vocab["team_ids"]),
        num_tags=len(vocab["tag_ids"]),
        hidden_channels=args.hidden_channels,
    )
    model = FootballHGT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = []
    best_macro_f1 = -1.0
    best_state = None
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, config.num_event_types, optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, device, config.num_event_types
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        print(
            f"K={args.window_size} epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_event_f1={validation_metrics['event_macro_f1']:.4f} "
            f"val_event_acc={validation_metrics['event_accuracy']:.4f}",
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
    result = {
        "experiment": "fixed_window_ablation_smoke_v1",
        "window_size": args.window_size,
        "selection_metric": "validation.event_macro_f1",
        "config": {
            **vars(args),
            "graph_root": str(args.graph_root.resolve()),
            "split_path": str(args.split_path.resolve()),
            "vocab_path": str(args.vocab_path.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "model": asdict(config),
            "train_match_ids": [record.match_id for record in train_records],
            "validation_match_ids": [record.match_id for record in validation_records],
        },
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed,
        "best_epoch": best_epoch,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"k_{args.window_size}.json"
    checkpoint_path = args.output_dir / f"k_{args.window_size}.pt"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {"model_config": asdict(config), "model_state_dict": best_state}, checkpoint_path
    )
    print(json.dumps({"result": str(result_path), "checkpoint": str(checkpoint_path), "best_validation": best_epoch["validation"]}, indent=2))


if __name__ == "__main__":
    main()
