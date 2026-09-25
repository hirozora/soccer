"""Strict task receptive fields over one batched F80 Semantic V3 graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from football_benchmark.data import CanonicalSample
from football_benchmark.possession_graph import extract_causal_subgraph
from football_benchmark.protocol import ProtocolArtifacts

from .five_task_data import collate_five_task_hgt
from .possession_data import _to_possession_heterodata
from .subgraph_views import DEFAULT_SELECTOR_SEED


RF_CONFIGURATIONS = {
    "rf_f80_equivalence": {"event": 80, "time": 80, "position": 80},
    "rf_core_u5": {"event": 5, "time": 5, "position": 5},
    "rf_core_u10": {"event": 10, "time": 10, "position": 10},
    "rf_task": {"event": 10, "time": 10, "position": 5},
}


@dataclass(frozen=True)
class ReceptiveFieldSpec:
    name: str
    core_event_counts: dict[str, int]

    @property
    def unique_counts(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.core_event_counts.values()), reverse=True))


def resolve_rf_spec(name: str) -> ReceptiveFieldSpec:
    try:
        values = RF_CONFIGURATIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown receptive-field configuration {name!r}") from exc
    return ReceptiveFieldSpec(name, dict(values))


def _short_indices(anchor: int, count: int) -> torch.Tensor:
    return torch.arange(max(0, anchor - count + 1), anchor + 1, dtype=torch.long)


def _node_source_indices(full: Any, compact: Any, node_type: str) -> torch.Tensor:
    if hasattr(compact[node_type], "source_index"):
        selected = compact[node_type].source_index.long()
        full_source = full[node_type].source_index.long()
        lookup = {int(value): index for index, value in enumerate(full_source.tolist())}
        return torch.tensor([lookup[int(value)] for value in selected.tolist()], dtype=torch.long)
    raw = compact[node_type].raw_id.long()
    full_raw = full[node_type].raw_id.long()
    lookup = {int(value): index for index, value in enumerate(full_raw.tolist())}
    return torch.tensor([lookup[int(value)] for value in raw.tolist()], dtype=torch.long)


def _single_rf_metadata(
    sample: CanonicalSample,
    full: Any,
    artifacts: ProtocolArtifacts,
    window_size: int,
    count: int,
    selector_seed: int,
) -> dict[str, Any]:
    indices = _short_indices(sample.current_event_index, count)
    compact, _ = _to_possession_heterodata(
        sample,
        artifacts,
        window_size,
        snapshot_scope="selected_events",
        feature_level="dynamic",
        topology="membership",
        context_view="f80" if count == 80 else f"s{count}",
        selector_seed=selector_seed,
        retain_full_player_roster=False,
    )
    node_masks: dict[str, torch.Tensor] = {}
    compact_to_full: dict[str, torch.Tensor] = {}
    for node_type in full.node_types:
        selected = _node_source_indices(full, compact, node_type)
        mask = torch.zeros(full[node_type].num_nodes, dtype=torch.bool)
        mask[selected] = True
        node_masks[node_type] = mask
        compact_to_full[node_type] = selected
    edge_masks: dict[tuple[str, str, str], torch.Tensor] = {}
    for edge_type in full.edge_types:
        edge = full[edge_type].edge_index
        edge_masks[edge_type] = node_masks[edge_type[0]][edge[0]] & node_masks[edge_type[2]][edge[1]]
    possession_dynamic = {}
    selected_possessions = compact_to_full["possession"]
    for field in (
        "duration_so_far_seconds", "event_count_so_far", "current_position",
        "current_position_mask", "is_closed_as_of_anchor", "is_current_active",
        "dynamic_feature_mask",
    ):
        target = torch.zeros_like(full["possession"][field])
        target[selected_possessions] = compact["possession"][field]
        possession_dynamic[field] = target
    return {
        "node_masks": node_masks,
        "edge_masks": edge_masks,
        "event_pool_mask": node_masks["event"],
        "possession_dynamic": possession_dynamic,
        "event_count": int(indices.numel()),
    }


def collate_receptive_field_hgt(
    samples: Sequence[CanonicalSample],
    artifacts: ProtocolArtifacts,
    window_size: int,
    *,
    configuration: str,
    selector_seed: int = DEFAULT_SELECTOR_SEED,
) -> dict[str, Any]:
    """Collate one F80 graph plus strict logical receptive-field metadata."""
    from torch_geometric.data import Batch

    spec = resolve_rf_spec(configuration)
    base = collate_five_task_hgt(
        samples, artifacts, window_size, context_views=("f80",), selector_seed=selector_seed
    )
    full_list = base["graphs"]["f80"].to_data_list()
    per_sample: dict[int, list[dict[str, Any]]] = {count: [] for count in spec.unique_counts}
    for sample, full in zip(samples, full_list):
        for count in spec.unique_counts:
            per_sample[count].append(
                _single_rf_metadata(sample, full, artifacts, window_size, count, selector_seed)
            )
    metadata = {}
    for count, entries in per_sample.items():
        node_masks = {
            node_type: torch.cat([entry["node_masks"][node_type] for entry in entries])
            for node_type in full_list[0].node_types
        }
        edge_masks = {
            edge_type: torch.cat([entry["edge_masks"][edge_type] for entry in entries])
            for edge_type in full_list[0].edge_types
        }
        possession_dynamic = {
            field: torch.cat([entry["possession_dynamic"][field] for entry in entries])
            for field in entries[0]["possession_dynamic"]
        }
        metadata[count] = {
            "node_masks": node_masks,
            "edge_masks": edge_masks,
            "event_pool_mask": node_masks["event"],
            "possession_dynamic": possession_dynamic,
            "event_counts": torch.tensor([entry["event_count"] for entry in entries]),
        }
    base["rf_spec"] = spec.core_event_counts
    base["rf_metadata"] = metadata
    return base
