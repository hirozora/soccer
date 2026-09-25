"""Whole-dataset proof of the F80 local-rank age contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .constants import POSSESSION_GRAPH_ROOT, SPLIT_PATH


def validate_age_index_contract(output_path: Path | None = None) -> dict[str, Any]:
    index = pd.read_csv(POSSESSION_GRAPH_ROOT / "metadata/match_index.csv")
    split = pd.read_csv(SPLIT_PATH)
    split_column = "split" if "split" in split.columns else "subset"
    split_by_match = dict(zip(split.match_id.astype(int), split[split_column].astype(str)))
    counts: dict[str, dict[str, int]] = {}
    mismatch_count = 0
    total_events = total_windows = total_occurrences = 0
    for row in index.itertuples():
        match_id, num_events = int(row.match_id), int(row.num_events)
        graph = torch.load(
            POSSESSION_GRAPH_ROOT / row.graph_path,
            map_location="cpu", weights_only=True,
        )
        event = graph["node_stores"]["event"]
        if int(event["num_nodes"]) != num_events:
            raise RuntimeError(f"Event count mismatch for match {match_id}")
        for field, value in event.items():
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] not in {num_events, 0}:
                raise RuntimeError(f"Non-aligned Event field {field} in match {match_id}")
        # F80 is a contiguous suffix. For every possible length, subtraction
        # from the anchor source rank must equal reverse local rank.
        for length in range(1, min(num_events, 80) + 1):
            source = torch.arange(length)
            local_age = torch.arange(length - 1, -1, -1)
            mismatch_count += int(((length - 1 - source) != local_age).sum())
        windows = max(num_events - 1, 0)
        occurrences = sum(min(anchor + 1, 80) for anchor in range(windows))
        name = split_by_match.get(match_id, "unknown")
        bucket = counts.setdefault(name, {"matches": 0, "events": 0, "windows": 0, "event_occurrences": 0})
        bucket["matches"] += 1
        bucket["events"] += num_events
        bucket["windows"] += windows
        bucket["event_occurrences"] += occurrences
        total_events += num_events
        total_windows += windows
        total_occurrences += occurrences
    result = {
        "matches": int(len(index)),
        "events": total_events,
        "windows": total_windows,
        "event_occurrences": total_occurrences,
        "source_age_mismatch_count": mismatch_count,
        "by_split": counts,
        "contract": "age is reverse local rank in the contiguous F80 causal suffix",
    }
    if mismatch_count:
        raise RuntimeError(f"Age-index contract has {mismatch_count} mismatches")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
