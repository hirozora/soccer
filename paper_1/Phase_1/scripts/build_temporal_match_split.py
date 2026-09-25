#!/usr/bin/env python3
"""Create reproducible per-competition chronological match splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PHASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PHASE_ROOT / "data/whyscout"
DEFAULT_GRAPH_ROOT = DEFAULT_DATA_ROOT / "processed/heterogeneous_graphs/v1"
DEFAULT_OUTPUT_DIR = PHASE_ROOT / "version_1/data_splits"


def load_match_dates(data_root: Path) -> dict[int, str]:
    dates: dict[int, str] = {}
    for path in sorted((data_root / "raw/matches").glob("matches_*.json")):
        for match in json.loads(path.read_text(encoding="utf-8")):
            match_id = int(match["wyId"])
            if match_id in dates:
                raise ValueError(f"Duplicate match ID {match_id}")
            dates[match_id] = match["dateutc"]
    return dates


def split_boundaries(num_matches: int, train_fraction: float, val_fraction: float) -> tuple[int, int]:
    train_end = int(num_matches * train_fraction)
    val_end = train_end + int(num_matches * val_fraction)
    if train_end < 1 or val_end <= train_end or val_end >= num_matches:
        raise ValueError(
            f"Competition with {num_matches} matches is too small for requested fractions"
        )
    return train_end, val_end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be in (0, 1)")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be in (0, 1)")
    if args.train_fraction + args.val_fraction >= 1.0:
        parser.error("train and validation fractions must sum to less than 1")

    match_dates = load_match_dates(args.data_root)
    with (args.graph_root / "metadata/match_index.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        index_rows = list(csv.DictReader(handle))

    by_competition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in index_rows:
        match_id = int(row["match_id"])
        if match_id not in match_dates:
            raise ValueError(f"Match {match_id} is missing date metadata")
        row = dict(row)
        row["dateutc"] = match_dates[match_id]
        by_competition[row["competition_slug"]].append(row)

    output_rows: list[dict[str, str | int]] = []
    summary: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "unit": "match",
            "ordering": "dateutc,match_id within each competition",
            "train_fraction": args.train_fraction,
            "validation_fraction": args.val_fraction,
            "test_fraction": 1.0 - args.train_fraction - args.val_fraction,
        },
        "competitions": {},
    }
    total_counts: Counter[str] = Counter()
    total_events: Counter[str] = Counter()

    for competition, rows in sorted(by_competition.items()):
        rows.sort(key=lambda row: (row["dateutc"], int(row["match_id"])))
        train_end, val_end = split_boundaries(
            len(rows), args.train_fraction, args.val_fraction
        )
        counts: Counter[str] = Counter()
        events: Counter[str] = Counter()
        for index, row in enumerate(rows):
            split = "train" if index < train_end else "validation" if index < val_end else "test"
            counts[split] += 1
            events[split] += int(row["num_events"])
            output_rows.append(
                {
                    "competition_slug": competition,
                    "competition_id": int(row["competition_id"]),
                    "match_id": int(row["match_id"]),
                    "dateutc": row["dateutc"],
                    "split": split,
                    "graph_path": row["graph_path"],
                    "num_events": int(row["num_events"]),
                    "num_supervised_steps": int(row["num_supervised_steps"]),
                }
            )
        total_counts.update(counts)
        total_events.update(events)
        summary["competitions"][competition] = {
            "matches": dict(counts),
            "events": dict(events),
            "date_ranges": {
                split: {
                    "first": next(row["dateutc"] for i, row in enumerate(rows) if ("train" if i < train_end else "validation" if i < val_end else "test") == split),
                    "last": next(row["dateutc"] for i, row in reversed(list(enumerate(rows))) if ("train" if i < train_end else "validation" if i < val_end else "test") == split),
                }
                for split in ("train", "validation", "test")
            },
        }

    summary["totals"] = {
        "matches": dict(total_counts),
        "events": dict(total_events),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir / "temporal_match_split_v1.csv"
    with split_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary_path = args.output_dir / "temporal_match_split_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split_path": str(split_path), "summary_path": str(summary_path), **summary["totals"]}, indent=2))


if __name__ == "__main__":
    main()
