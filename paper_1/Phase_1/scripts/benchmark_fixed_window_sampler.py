#!/usr/bin/env python3
"""Validate and benchmark causal fixed-event-window sampling on a real graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter


PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT / "src"))

from football_hgt.dataset import (  # noqa: E402
    load_match_graph,
    load_match_index,
    sample_fixed_event_window,
    validate_fixed_event_window,
)


DEFAULT_DATASET_ROOT = (
    PHASE_ROOT / "data/whyscout/processed/heterogeneous_graphs/v1"
)


def parse_window_sizes(raw_value: str) -> list[int]:
    values = [int(value.strip()) for value in raw_value.split(",") if value.strip()]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("window sizes must be positive integers")
    return values


def evenly_spaced_indices(num_steps: int, requested: int) -> list[int]:
    count = min(num_steps, requested)
    if count == 1:
        return [0]
    return [round(index * (num_steps - 1) / (count - 1)) for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument("--competition", default="England")
    parser.add_argument("--match-id", type=int)
    parser.add_argument(
        "--window-sizes", type=parse_window_sizes, default=parse_window_sizes("10,20,40,80")
    )
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    records = load_match_index(args.dataset_root, competitions=[args.competition])
    if args.match_id is not None:
        records = [record for record in records if record.match_id == args.match_id]
        if not records:
            parser.error(
                f"match {args.match_id} was not found in competition {args.competition}"
            )
    record = records[0]
    graph = load_match_graph(record.graph_path, validate=True)
    num_steps = record.num_events - 1
    sample_indices = evenly_spaced_indices(num_steps, args.samples)

    results = []
    for window_size in args.window_sizes:
        validation_indices = sorted(
            {sample_indices[0], sample_indices[len(sample_indices) // 2], sample_indices[-1]}
        )
        for current_event_index in validation_indices:
            sample = sample_fixed_event_window(
                graph, current_event_index, window_size=window_size
            )
            errors = validate_fixed_event_window(sample)
            if errors:
                raise ValueError(
                    f"K={window_size}, event={current_event_index}: {'; '.join(errors)}"
                )

        started = perf_counter()
        total_event_nodes = 0
        total_tag_edges = 0
        for current_event_index in sample_indices:
            sample = sample_fixed_event_window(
                graph, current_event_index, window_size=window_size
            )
            total_event_nodes += sample["window"]["num_events"]
            total_tag_edges += sample["edge_stores"][
                "tag__describes__event"
            ]["edge_index"].shape[1]
        elapsed = perf_counter() - started

        final_sample = sample_fixed_event_window(
            graph, sample_indices[-1], window_size=window_size
        )
        shares_event_storage = (
            final_sample["node_stores"]["event"]["raw_id"]
            .untyped_storage()
            .data_ptr()
            == graph["node_stores"]["event"]["raw_id"]
            .untyped_storage()
            .data_ptr()
        )
        results.append(
            {
                "window_size": window_size,
                "samples": len(sample_indices),
                "elapsed_seconds": elapsed,
                "samples_per_second": len(sample_indices) / elapsed,
                "mean_event_nodes": total_event_nodes / len(sample_indices),
                "mean_tag_edges": total_tag_edges / len(sample_indices),
                "shares_event_tensor_storage": shares_event_storage,
                "validation_status": "passed",
            }
        )

    output = {
        "dataset_root": str(args.dataset_root.resolve()),
        "competition_slug": record.competition_slug,
        "match_id": record.match_id,
        "graph_path": str(record.graph_path.resolve()),
        "num_events": record.num_events,
        "num_supervised_steps": num_steps,
        "results": results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
