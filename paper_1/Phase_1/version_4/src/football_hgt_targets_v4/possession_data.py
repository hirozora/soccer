"""Causal Semantic V3 collation for the Possession regression study."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import torch

from football_benchmark.data import CanonicalSample, _stack_targets, sample_to_semantic_heterodata
from football_benchmark.possession_graph import extract_causal_subgraph
from football_benchmark.protocol import ProtocolArtifacts

from .subgraph_views import DEFAULT_SELECTOR_SEED, ViewSelection, select_event_indices


POSSESSION_EDGE_TYPES = (
    ("event", "belongs_to", "possession"),
    ("possession", "contains", "event"),
    ("possession", "owned_by", "team"),
    ("team", "has_possession", "possession"),
    ("possession", "next", "possession"),
)


def _semantic_compatible_sample(sample: CanonicalSample) -> CanonicalSample:
    """Expose the unchanged Semantic V2 portion of a V3 graph."""

    graph = dict(sample.graph)
    graph["schema_version"] = "2.0.0"
    return replace(sample, graph=graph)


def _to_possession_heterodata(
    sample: CanonicalSample,
    artifacts: ProtocolArtifacts,
    window_size: int,
    *,
    snapshot_scope: str,
    feature_level: str,
    topology: str,
    context_view: str,
    selector_seed: int,
    retain_full_player_roster: bool = False,
) -> tuple[Any, ViewSelection]:
    selection = select_event_indices(
        sample.graph,
        sample.current_event_index,
        context_view,
        selector_seed=selector_seed,
    )
    indices = selection.event_indices
    causal = extract_causal_subgraph(
        sample.graph,
        indices,
        sample.current_event_index,
        snapshot_scope=snapshot_scope,
        include_dynamic_possession_features=feature_level == "dynamic",
    )
    if retain_full_player_roster:
        # Player prediction uses the match-local pre-match candidate universe for
        # every view. Only incidence edges from selected historical events are
        # retained, so this does not reintroduce unselected Event information.
        causal["node_stores"]["player"] = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in sample.graph["node_stores"]["player"].items()
        }
        local_events = torch.arange(indices.numel(), dtype=torch.long)
        player_indices = sample.graph["node_stores"]["event"][
            "player_local_index"
        ][indices].long()
        causal["edge_stores"]["event__performed_by__player"]["edge_index"] = (
            torch.stack((local_events, player_indices))
        )
        causal["edge_stores"]["player__performs__event"]["edge_index"] = (
            torch.stack((player_indices, local_events))
        )
    semantic = dict(causal)
    semantic["schema_version"] = "2.0.0"
    local_sample = replace(
        sample,
        graph=semantic,
        start=0,
        stop=int(indices.numel()),
        current_event_index=int(indices.numel() - 1),
    )
    data = sample_to_semantic_heterodata(local_sample, artifacts, window_size)
    source_indices = causal["node_stores"]["event"]["source_index"].long()
    denominator = float(
        max(window_size - 1, int(source_indices[-1] - source_indices[0]), 1)
    )
    data["event"].relative_features[:, 0] = (
        source_indices.float() - float(source_indices[-1])
    ) / denominator
    data["event"].source_index = source_indices
    for node_type in ("player", "team", "event_type", "tag", "zone"):
        source = causal["node_stores"][node_type].get("source_index")
        if source is None:
            source = torch.arange(
                int(causal["node_stores"][node_type]["num_nodes"]), dtype=torch.long
            )
        data[node_type].source_index = source.long()
    data["player"].raw_id = causal["node_stores"]["player"]["raw_id"].long()
    event = causal["node_stores"]["event"]
    data["event"].event_role_index = event["event_role_index"]
    data["event"].actor_relation_to_owner_index = event[
        "actor_relation_to_owner_index"
    ]
    data["event"].control_state_after_index = event["control_state_after_index"]
    data["event"].candidate_status_after_index = event[
        "candidate_status_after_index"
    ]
    data["event"].switch_confirmed = event["switch_confirmed"].bool()

    possession = causal["node_stores"]["possession"]
    count = int(possession["num_nodes"])
    data["possession"].num_nodes = count
    data["possession"].source_index = possession["source_index"].long()
    field_specs = {
        "duration_so_far_seconds": (torch.float32, (count,)),
        "event_count_so_far": (torch.float32, (count,)),
        "current_position": (torch.float32, (count, 2)),
        "current_position_mask": (torch.bool, (count,)),
        "is_closed_as_of_anchor": (torch.bool, (count,)),
        "is_current_active": (torch.bool, (count,)),
        "dynamic_feature_mask": (torch.bool, (count,)),
    }
    for field, (dtype, shape) in field_specs.items():
        data["possession"][field] = possession.get(
            field, torch.empty(shape, dtype=dtype)
        )
    for edge_type in POSSESSION_EDGE_TYPES:
        key = "__".join(edge_type)
        enabled = (
            edge_type[1] in {"belongs_to", "contains"}
            or topology in {"owner", "transition"}
            and edge_type[1] in {"owned_by", "has_possession"}
            or topology == "transition"
            and edge_type[1] == "next"
        )
        edge_index = causal["edge_stores"][key]["edge_index"]
        data[edge_type].edge_index = (
            edge_index if enabled else torch.empty((2, 0), dtype=torch.long)
        )
    return data, selection


def collate_possession_hgt(
    samples: Sequence[CanonicalSample],
    artifacts: ProtocolArtifacts,
    window_size: int,
    *,
    topology: str,
    feature_level: str,
    snapshot_scope: str,
    context_view: str = "f80",
    selector_seed: int = DEFAULT_SELECTOR_SEED,
    retain_full_player_roster: bool = False,
) -> dict[str, Any]:
    """Batch V3 causal windows; ``none`` is a strict Semantic V2 path."""

    try:
        from torch_geometric.data import Batch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch-geometric is required for semantic HGT") from exc
    if topology == "none":
        if context_view != "f80":
            raise ValueError("Sparse context views require an enabled Semantic V3 topology")
        values = [
            sample_to_semantic_heterodata(
                _semantic_compatible_sample(sample), artifacts, window_size
            )
            for sample in samples
        ]
    else:
        converted = [
            _to_possession_heterodata(
                sample,
                artifacts,
                window_size,
                snapshot_scope=snapshot_scope,
                feature_level=feature_level,
                topology=topology,
                context_view=context_view,
                selector_seed=selector_seed,
                retain_full_player_roster=retain_full_player_roster,
            )
            for sample in samples
        ]
        values = [value[0] for value in converted]
        selections = [value[1] for value in converted]
    result = {
        "graph": Batch.from_data_list(values),
        "targets": _stack_targets(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "match_ids": torch.tensor([sample.match_id for sample in samples]),
        "current_event_indices": torch.tensor(
            [sample.current_event_index for sample in samples]
        ),
    }
    if topology != "none":
        result["view_name"] = context_view
        result["view_marker_types"] = [value.marker_type for value in selections]
        result["view_fallback_reasons"] = [value.fallback_reason for value in selections]
        result["view_event_counts"] = torch.tensor(
            [value.event_count for value in selections], dtype=torch.long
        )
        result["view_source_spans"] = torch.tensor(
            [value.source_span for value in selections], dtype=torch.long
        )
        result["view_time_spans_seconds"] = torch.tensor(
            [value.time_span_seconds for value in selections], dtype=torch.float32
        )
        result["view_gap_rates"] = torch.tensor(
            [value.gap_rate for value in selections], dtype=torch.float32
        )
    return result


def collate_multiview_possession_hgt(
    samples: Sequence[CanonicalSample],
    artifacts: ProtocolArtifacts,
    window_size: int,
    *,
    topology: str,
    feature_level: str,
    snapshot_scope: str,
    context_views: Sequence[str],
    selector_seed: int = DEFAULT_SELECTOR_SEED,
    retain_full_player_roster: bool = False,
) -> dict[str, Any]:
    """Build aligned graph batches for several deterministic context views."""

    views = tuple(dict.fromkeys(context_views))
    if not views:
        raise ValueError("At least one context view is required")
    batches = {
        view: collate_possession_hgt(
            samples,
            artifacts,
            window_size,
            topology=topology,
            feature_level=feature_level,
            snapshot_scope=snapshot_scope,
            context_view=view,
            selector_seed=selector_seed,
            retain_full_player_roster=retain_full_player_roster,
        )
        for view in views
    }
    reference = batches[views[0]]
    for view in views[1:]:
        candidate = batches[view]
        if candidate["sample_ids"] != reference["sample_ids"]:
            raise RuntimeError(f"Multi-view sample IDs differ for {view}")
        if not torch.equal(candidate["match_ids"], reference["match_ids"]):
            raise RuntimeError(f"Multi-view match IDs differ for {view}")
        if not torch.equal(
            candidate["current_event_indices"], reference["current_event_indices"]
        ):
            raise RuntimeError(f"Multi-view anchors differ for {view}")
        for name, target in reference["targets"].items():
            other = candidate["targets"][name]
            if target.dtype.is_floating_point:
                equal = torch.allclose(target, other, equal_nan=True)
            else:
                equal = torch.equal(target, other)
            if not equal:
                raise RuntimeError(f"Multi-view target {name} differs for {view}")
    return {
        "graphs": {view: batches[view]["graph"] for view in views},
        "targets": reference["targets"],
        "sample_ids": reference["sample_ids"],
        "match_ids": reference["match_ids"],
        "current_event_indices": reference["current_event_indices"],
        "view_diagnostics": {
            view: {
                key: value
                for key, value in batches[view].items()
                if key.startswith("view_")
            }
            for view in views
        },
    }
