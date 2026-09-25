"""Pre-training structural statistics for deterministic context views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch

from football_benchmark.data import load_records
from football_benchmark.sampling import TargetSamplePlan

from .constants import POSSESSION_GRAPH_ROOT, SAMPLE_PLAN
from .subgraph_study import SUBGRAPH_EXPERIMENT_ROOT
from .subgraph_views import FIXED_VIEW_SPECS, select_event_indices


def _view_names() -> tuple[str, ...]:
    return tuple(FIXED_VIEW_SPECS)


def _rows_for_match(
    graph: dict[str, Any], split: str, currents: Iterable[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    event = graph["node_stores"]["event"]
    for anchor in currents:
        for name in _view_names():
            selection = select_event_indices(graph, int(anchor), name)
            indices = selection.event_indices
            consecutive = int(((indices[1:] - indices[:-1]) == 1).sum()) if indices.numel() > 1 else 0
            rows.append(
                {
                    "split": split,
                    "match_id": int(graph["match_id"]),
                    "sample_id": f"{int(graph['match_id'])}:{int(anchor)}",
                    "current_event_index": int(anchor),
                    "view": name,
                    "family": FIXED_VIEW_SPECS[name].family,
                    "event_count": selection.event_count,
                    "source_span": selection.source_span,
                    "time_span_seconds": selection.time_span_seconds,
                    "gap_rate": selection.gap_rate,
                    "marker_type": selection.marker_type,
                    "fallback_reason": selection.fallback_reason,
                    "next_edge_count": consecutive,
                    "time_edge_count": consecutive,
                    "possession_membership_count": int(
                        (event["possession_local_index"][indices] >= 0).sum()
                    ),
                    "tag_edge_count": int(event["retained_tag_count"][indices].sum()),
                    "start_zone_edge_count": int(event["start_position_mask"][indices].sum()),
                    "end_zone_edge_count": int(event["end_position_mask"][indices].sum()),
                }
            )
    return rows


def build_view_statistics(output_dir: Path | None = None) -> Path:
    output = output_dir or SUBGRAPH_EXPERIMENT_ROOT / "view_statistics"
    output.mkdir(parents=True, exist_ok=True)
    plan = TargetSamplePlan.load(SAMPLE_PLAN)
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        selections = plan.currents_by_match(split)
        for record in load_records(split, graph_root=POSSESSION_GRAPH_ROOT):
            graph = torch.load(record.graph_path, map_location="cpu", weights_only=True)
            rows.extend(_rows_for_match(graph, split, selections[record.match_id]))
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "per_sample_views.parquet", index=False)
    numeric = (
        frame.groupby(["split", "view", "family"])
        .agg(
            samples=("sample_id", "size"),
            event_count_mean=("event_count", "mean"),
            event_count_median=("event_count", "median"),
            event_count_p90=("event_count", lambda value: value.quantile(0.9)),
            source_span_mean=("source_span", "mean"),
            time_span_mean_seconds=("time_span_seconds", "mean"),
            gap_rate_mean=("gap_rate", "mean"),
            next_edges_mean=("next_edge_count", "mean"),
            possession_edges_mean=("possession_membership_count", "mean"),
            tag_edges_mean=("tag_edge_count", "mean"),
            start_zone_edges_mean=("start_zone_edge_count", "mean"),
            end_zone_edges_mean=("end_zone_edge_count", "mean"),
        )
        .reset_index()
    )
    numeric.to_csv(output / "summary.csv", index=False)
    frame.groupby(["split", "view", "marker_type"]).size().rename("samples").reset_index().to_csv(
        output / "transition_markers.csv", index=False
    )
    frame.groupby(["split", "view", "fallback_reason"]).size().rename("samples").reset_index().to_csv(
        output / "fallbacks.csv", index=False
    )
    manifest = {
        "views": list(_view_names()),
        "splits": sorted(frame.split.unique().tolist()),
        "rows": len(frame),
        "source_graph": str(POSSESSION_GRAPH_ROOT),
        "sample_plan": str(SAMPLE_PLAN),
        "test_accessed": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output
