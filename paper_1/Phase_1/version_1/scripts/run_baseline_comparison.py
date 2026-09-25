#!/usr/bin/env python3
"""Train one baseline adapter on the same controlled match/sample subset as HGT."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader, Subset


VERSION_ROOT = Path(__file__).resolve().parents[1]
PHASE_ROOT = VERSION_ROOT.parent
sys.path.insert(0, str(PHASE_ROOT / "src"))
sys.path.insert(0, str(VERSION_ROOT / "src"))

from football_hgt.dataset import MatchGraphRecord  # noqa: E402
from football_hgt_v1.baselines import (  # noqa: E402
    OGLEMExtendedAdapter,
    SequenceBaselineDataset,
    SoccerSeq2EventAdapter,
    UnifiedLEMAdapter,
    baseline_loss,
)
from football_hgt_v1.training import macro_f1, set_seed  # noqa: E402


DEFAULT_GRAPH_ROOT = PHASE_ROOT / "data/whyscout/processed/heterogeneous_graphs/v1"
DEFAULT_SPLIT = VERSION_ROOT / "data_splits/temporal_match_split_v1.csv"
DEFAULT_VOCAB = DEFAULT_GRAPH_ROOT / "metadata/vocabularies.json"
DEFAULT_OUTPUT = VERSION_ROOT / "experiments/baseline_comparison"
MODEL_LENGTHS = {
    "soccer_seq2event": 40,
    "unified_lem": 9,
    "og_lem_extended": 1,
}


def select_evenly(values: list, count: int) -> list:
    if count >= len(values):
        return values
    if count == 1:
        return [values[len(values) // 2]]
    indices = [round(i * (len(values) - 1) / (count - 1)) for i in range(count)]
    return [values[index] for index in indices]


def load_records(
    path: Path, graph_root: Path, competition: str, split: str, count: int
) -> list[MatchGraphRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["competition_slug"] == competition and row["split"] == split
        ]
    return [
        MatchGraphRecord(
            competition_slug=row["competition_slug"],
            match_id=int(row["match_id"]),
            graph_path=graph_root / row["graph_path"],
            num_events=int(row["num_events"]),
        )
        for row in select_evenly(rows, count)
    ]


def selected_indices(records: list[MatchGraphRecord], steps_per_match: int) -> list[int]:
    result = []
    offset = 0
    for record in records:
        steps = record.num_events - 1
        count = min(steps, steps_per_match)
        if count == 1:
            local = [0]
        else:
            local = [round(i * (steps - 1) / (count - 1)) for i in range(count)]
        result.extend(offset + value for value in local)
        offset += steps
    return result


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def evaluate(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_event_types: int,
) -> dict[str, float | None]:
    model.eval()
    confusion = torch.zeros((num_event_types, num_event_types), dtype=torch.long)
    loss_sum = 0.0
    count = 0
    time_absolute = 0.0
    position_euclidean = 0.0
    position_count = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            predictions = model(batch)
            loss = baseline_loss(model_name, model, predictions, batch)
            probabilities = predictions.get("event_probabilities")
            event_prediction = (
                probabilities.argmax(dim=-1)
                if probabilities is not None
                else predictions["event_logits"].argmax(dim=-1)
            )
            target = batch["target_event_type"]
            flat = target * num_event_types + event_prediction
            confusion += torch.bincount(
                flat.cpu(), minlength=num_event_types**2
            ).reshape(num_event_types, num_event_types)
            batch_size = int(target.numel())
            loss_sum += float(loss) * batch_size
            count += batch_size
            if model.supports_continuous_targets:
                seconds = torch.expm1(predictions["log_delta"]).clamp_min(0.0)
                time_absolute += float(
                    (seconds - batch["target_delta_seconds"]).abs().sum()
                )
                mask = batch["target_position_mask"].bool()
                if bool(mask.any()):
                    error = (
                        predictions["position"][mask] - batch["target_position"][mask]
                    ) * 100.0
                    position_euclidean += float(
                        torch.linalg.vector_norm(error, dim=-1).sum()
                    )
                    position_count += int(mask.sum())
    return {
        "loss": loss_sum / count,
        "event_accuracy": float(confusion.diag().sum()) / count,
        "event_macro_f1": macro_f1(confusion),
        "time_mae_seconds": time_absolute / count
        if model.supports_continuous_targets
        else None,
        "position_euclidean_distance": position_euclidean / position_count
        if model.supports_continuous_targets
        else None,
        "num_samples": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_LENGTHS), required=True)
    parser.add_argument("--competition", default="England")
    parser.add_argument("--train-matches", type=int, default=48)
    parser.add_argument("--validation-matches", type=int, default=12)
    parser.add_argument("--steps-per-match", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument(
        "--sequence-length",
        type=int,
        help="Override the adapter's default context length for an equal-window comparison.",
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override the adapter's default learning rate for controlled tuning.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    vocab = json.loads(args.vocab_path.read_text(encoding="utf-8"))
    num_event_types = len(vocab["event_type_ids"])
    sequence_length = args.sequence_length or MODEL_LENGTHS[args.model]
    if sequence_length < 1:
        raise ValueError("sequence length must be positive")
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

    def loader(records: list[MatchGraphRecord], shuffle: bool) -> DataLoader:
        dataset = SequenceBaselineDataset(
            records, sequence_length, num_event_types
        )
        subset = Subset(dataset, selected_indices(records, args.steps_per_match))
        return DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = loader(train_records, True)
    validation_loader = loader(validation_records, False)
    if args.model == "soccer_seq2event":
        model = SoccerSeq2EventAdapter(num_event_types, sequence_length)
        default_learning_rate = 1e-2
    elif args.model == "unified_lem":
        model = UnifiedLEMAdapter(num_event_types, sequence_length)
        default_learning_rate = 1e-3
    else:
        model = OGLEMExtendedAdapter(num_event_types, sequence_length)
        default_learning_rate = 1e-3
    learning_rate = args.learning_rate or default_learning_rate
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best = None
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch)
            loss = baseline_loss(args.model, model, predictions, batch)
            loss.backward()
            optimizer.step()
        metrics = evaluate(
            args.model, model, validation_loader, device, num_event_types
        )
        history.append({"epoch": epoch, "validation": metrics})
        print(
            f"model={args.model} seed={args.seed} epoch={epoch} "
            f"val_f1={metrics['event_macro_f1']:.4f} "
            f"val_acc={metrics['event_accuracy']:.4f}",
            flush=True,
        )
        if best is None or metrics["event_macro_f1"] > best["validation"]["event_macro_f1"]:
            best = history[-1]

    result = {
        "experiment": "baseline_comparison_smoke_v1",
        "model": args.model,
        "upstream_repository": (
            "baseline/Soccer-Seq2Event"
            if args.model == "soccer_seq2event"
            else "baseline/Unified-LEM"
        ),
        "sequence_length": sequence_length,
        "seed": args.seed,
        "train_matches": [record.match_id for record in train_records],
        "validation_matches": [record.match_id for record in validation_records],
        "steps_per_match": args.steps_per_match,
        "epochs": args.epochs,
        "learning_rate": learning_rate,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": perf_counter() - started,
        "best_epoch": best,
        "history": history,
    }
    output_dir = args.output_dir / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output_path), "best": best}, indent=2))


if __name__ == "__main__":
    main()
