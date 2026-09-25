"""Deterministic causal Event selections for Semantic V3 scale studies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import torch

from football_benchmark.constants import PITCH_LENGTH_METERS, PITCH_WIDTH_METERS


ViewFamily = Literal[
    "full", "short", "possession", "transition", "spatial", "random",
    "matched_recency",
]

EVENT_ROLE_RESTART = 2
EVENT_ROLE_BOUNDARY = 4
DEFAULT_SELECTOR_SEED = 20260811


@dataclass(frozen=True)
class SubgraphViewSpec:
    name: str
    family: ViewFamily
    scale: float | int
    event_cap: int
    reference_view: str | None = None


@dataclass(frozen=True)
class ViewSelection:
    event_indices: torch.Tensor
    marker_type: str
    fallback_reason: str
    event_count: int
    source_span: int
    time_span_seconds: float
    gap_rate: float


FIXED_VIEW_SPECS: dict[str, SubgraphViewSpec] = {
    "f80": SubgraphViewSpec("f80", "full", 80, 80),
    "s5": SubgraphViewSpec("s5", "short", 5, 5),
    "s10": SubgraphViewSpec("s10", "short", 10, 10),
    "s20": SubgraphViewSpec("s20", "short", 20, 20),
    "s40": SubgraphViewSpec("s40", "short", 40, 40),
    "p1": SubgraphViewSpec("p1", "possession", 1, 80),
    "p2": SubgraphViewSpec("p2", "possession", 2, 80),
    "p3": SubgraphViewSpec("p3", "possession", 3, 80),
    "lp1": SubgraphViewSpec("lp1", "matched_recency", 1, 80, "p1"),
    "lp2": SubgraphViewSpec("lp2", "matched_recency", 2, 80, "p2"),
    "tr5": SubgraphViewSpec("tr5", "transition", 5, 40),
    "tr10": SubgraphViewSpec("tr10", "transition", 10, 40),
    "tr20": SubgraphViewSpec("tr20", "transition", 20, 40),
    "sp15": SubgraphViewSpec("sp15", "spatial", 15.0, 80),
    "sp30": SubgraphViewSpec("sp30", "spatial", 30.0, 80),
    "sp45": SubgraphViewSpec("sp45", "spatial", 45.0, 80),
}

ROUND_A_VIEWS = ("s20", "s40", "p1", "p2", "tr10", "sp30")
ROUND_B_BY_FAMILY = {
    "short": ("s10",),
    "possession": ("p3",),
    "transition": ("tr5", "tr20"),
    "spatial": ("sp15", "sp45"),
}


def resolve_view_spec(name: str) -> SubgraphViewSpec:
    normalized = name.lower()
    if normalized in FIXED_VIEW_SPECS:
        return FIXED_VIEW_SPECS[normalized]
    if normalized.startswith("random_"):
        reference = normalized.removeprefix("random_")
        base = FIXED_VIEW_SPECS.get(reference)
        if base is None or base.family not in {"possession", "transition", "spatial"}:
            raise ValueError(f"Random controls require a P/TR/SP reference view: {name!r}")
        return SubgraphViewSpec(normalized, "random", base.scale, base.event_cap, reference)
    raise ValueError(f"Unknown subgraph view {name!r}")


def representative_positions(event: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    start = event["start_position"]
    start_mask = event["start_position_mask"].bool()
    end = event["end_position"]
    end_mask = event["end_position_mask"].bool()
    return torch.where(end_mask.unsqueeze(-1), end, start), end_mask | start_mask


def align_to_anchor_team(
    positions: torch.Tensor,
    event_team: torch.Tensor,
    anchor_team: int,
) -> torch.Tensor:
    """Map opponent-relative Wyscout coordinates into the anchor-team frame."""

    return torch.where(
        (event_team == int(anchor_team)).unsqueeze(-1), positions, 1.0 - positions
    )


def metric_distances(positions: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    delta = positions - anchor.unsqueeze(0)
    return torch.sqrt(
        (delta[:, 0] * PITCH_LENGTH_METERS) ** 2
        + (delta[:, 1] * PITCH_WIDTH_METERS) ** 2
    )


def _period_indices(event: dict[str, Any], anchor: int) -> torch.Tensor:
    period = event["period_index"][anchor]
    prefix = torch.arange(anchor + 1, dtype=torch.long)
    return prefix[event["period_index"][: anchor + 1] == period]


def _short(anchor: int, count: int) -> tuple[torch.Tensor, str, str]:
    return torch.arange(max(0, anchor - count + 1), anchor + 1), "none", "none"


def _current_possession(graph: dict[str, Any], anchor: int, period_events: torch.Tensor) -> int:
    event = graph["node_stores"]["event"]
    current = int(event["possession_local_index"][anchor])
    if current >= 0:
        return current
    active = int(event["active_possession_after_local_index"][anchor])
    if active >= 0:
        return active
    possession = graph["node_stores"]["possession"]
    if not int(possession["num_nodes"]):
        return -1
    period = int(event["period_index"][anchor])
    visible = (
        (possession["period_index"] == period)
        & (possession["start_event_index"] <= anchor)
    )
    candidates = torch.nonzero(visible, as_tuple=False).flatten()
    return int(candidates[-1]) if candidates.numel() else -1


def _visible_predecessors(graph: dict[str, Any], anchor: int) -> dict[int, int]:
    store = graph["edge_stores"]["possession__next__possession"]
    keep = (store["transition_event_index"] <= anchor) & ~store["cross_period"].bool()
    edges = store["edge_index"][:, keep]
    return {int(destination): int(source) for source, destination in edges.t().tolist()}


def _possession(
    graph: dict[str, Any], anchor: int, count: int, cap: int
) -> tuple[torch.Tensor, str, str]:
    event = graph["node_stores"]["event"]
    period_events = _period_indices(event, anchor)
    current = _current_possession(graph, anchor, period_events)
    if current < 0:
        return torch.tensor([anchor]), "none", "no_visible_possession"
    retained = [current]
    predecessors = _visible_predecessors(graph, anchor)
    while len(retained) < count and retained[-1] in predecessors:
        retained.append(predecessors[retained[-1]])
    possession_ids = torch.tensor(retained, dtype=torch.long)
    memberships = event["possession_local_index"][period_events]
    selected = period_events[torch.isin(memberships, possession_ids)]
    selected = torch.unique(torch.cat((selected, torch.tensor([anchor]))), sorted=True)
    if selected.numel() > cap:
        selected = selected[-cap:]
    fallback = "none" if len(retained) == count else "fewer_visible_possessions"
    return selected, "none", fallback


def _transition(
    graph: dict[str, Any], anchor: int, before: int, cap: int
) -> tuple[torch.Tensor, str, str]:
    event = graph["node_stores"]["event"]
    period_events = _period_indices(event, anchor)
    roles = event["event_role_index"][period_events]
    confirmed = event["switch_confirmed"][period_events].bool() | (
        roles == EVENT_ROLE_RESTART
    )
    boundary = roles == EVENT_ROLE_BOUNDARY
    markers = period_events[confirmed | boundary]
    if not markers.numel():
        return period_events[-cap:], "fallback", "no_transition_marker"
    marker = int(markers[-1])
    marker_type = "confirmed_transition" if bool(
        event["switch_confirmed"][marker]
        or event["event_role_index"][marker] == EVENT_ROLE_RESTART
    ) else "boundary_context"
    period_start = int(period_events[0])
    pre = torch.arange(max(period_start, marker - before), marker + 1)
    post = torch.arange(marker + 1, anchor + 1)
    if pre.numel() + post.numel() <= cap:
        selected = torch.cat((pre, post))
    else:
        remaining = max(cap - int(pre.numel()), 0)
        selected = torch.cat((pre[-cap:], post[-remaining:] if remaining else post[:0]))
    selected = torch.unique(torch.cat((selected, torch.tensor([anchor]))), sorted=True)
    if selected.numel() > cap:
        keep = torch.unique(torch.cat((pre, torch.tensor([anchor]))), sorted=True)
        remaining = max(cap - int(keep.numel()), 0)
        selected = torch.unique(torch.cat((keep[-cap:], post[-remaining:] if remaining else post[:0])), sorted=True)
    return selected, marker_type, "none"


def _spatial(
    graph: dict[str, Any], anchor: int, radius_m: float, cap: int
) -> tuple[torch.Tensor, str, str]:
    event = graph["node_stores"]["event"]
    period_events = _period_indices(event, anchor)
    last_five = period_events[-5:]
    positions, valid = representative_positions(event)
    if not bool(valid[anchor]):
        return last_five, "none", "missing_anchor_position"
    anchor_team = int(event["team_local_index"][anchor])
    candidates = period_events[valid[period_events]]
    aligned = align_to_anchor_team(
        positions[candidates], event["team_local_index"][candidates], anchor_team
    )
    anchor_position = positions[anchor]
    nearby = candidates[metric_distances(aligned, anchor_position) <= float(radius_m) + 1e-6]
    selected = torch.unique(torch.cat((nearby, last_five, torch.tensor([anchor]))), sorted=True)
    if selected.numel() > cap:
        selected = selected[-cap:]
    return selected, "none", "none"


def _stable_seed(match_id: int, anchor: int, reference: str, seed: int) -> int:
    payload = f"{seed}:{match_id}:{anchor}:{reference}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _random_control(
    graph: dict[str, Any], anchor: int, spec: SubgraphViewSpec, selector_seed: int
) -> tuple[torch.Tensor, str, str]:
    reference = resolve_view_spec(spec.reference_view or "")
    target = select_event_indices(graph, anchor, reference, selector_seed=selector_seed)
    event = graph["node_stores"]["event"]
    pool = _period_indices(event, anchor)
    count = min(target.event_count, int(pool.numel()))
    if count <= 1:
        return torch.tensor([anchor]), "none", f"size_matched:{reference.name}"
    candidates = pool[pool != anchor]
    generator = torch.Generator().manual_seed(
        _stable_seed(int(graph["match_id"]), anchor, reference.name, selector_seed)
    )
    chosen = candidates[torch.randperm(candidates.numel(), generator=generator)[: count - 1]]
    selected = torch.sort(torch.cat((chosen, torch.tensor([anchor])))).values
    return selected, "none", f"size_matched:{reference.name}"


def _matched_recency(
    graph: dict[str, Any], anchor: int, spec: SubgraphViewSpec
) -> tuple[torch.Tensor, str, str]:
    """Take the latest contiguous same-period events matching P1/P2 size."""

    reference = resolve_view_spec(spec.reference_view or "")
    if reference.family != "possession":
        raise ValueError("Matched-recency views require a Possession reference")
    target = select_event_indices(graph, anchor, reference)
    event = graph["node_stores"]["event"]
    period_events = _period_indices(event, anchor)
    count = target.event_count
    if count > int(period_events.numel()):
        raise RuntimeError("Possession reference exceeds visible same-period events")
    selected = period_events[-count:]
    return selected, "none", f"event_count_matched:{reference.name}"


def _selection_result(
    graph: dict[str, Any], indices: torch.Tensor, marker: str, fallback: str
) -> ViewSelection:
    indices = torch.unique(indices.long(), sorted=True)
    event = graph["node_stores"]["event"]
    span = int(indices[-1] - indices[0] + 1)
    time_span = float(event["absolute_seconds"][indices[-1]] - event["absolute_seconds"][indices[0]])
    gap_rate = 0.0 if span <= 1 else 1.0 - float(indices.numel() - 1) / float(span - 1)
    return ViewSelection(
        event_indices=indices,
        marker_type=marker,
        fallback_reason=fallback,
        event_count=int(indices.numel()),
        source_span=span,
        time_span_seconds=max(time_span, 0.0),
        gap_rate=max(0.0, min(gap_rate, 1.0)),
    )


def select_event_indices(
    graph: dict[str, Any],
    anchor_event_index: int,
    spec: SubgraphViewSpec | str,
    *,
    selector_seed: int = DEFAULT_SELECTOR_SEED,
) -> ViewSelection:
    """Select an anchor-causal Event view from a complete match graph."""

    view = resolve_view_spec(spec) if isinstance(spec, str) else spec
    anchor = int(anchor_event_index)
    num_events = int(graph["node_stores"]["event"]["num_nodes"])
    if anchor < 0 or anchor >= num_events:
        raise ValueError("anchor_event_index is out of range")
    if view.family in {"full", "short"}:
        indices, marker, fallback = _short(anchor, int(view.scale))
    elif view.family == "possession":
        indices, marker, fallback = _possession(graph, anchor, int(view.scale), view.event_cap)
    elif view.family == "transition":
        indices, marker, fallback = _transition(graph, anchor, int(view.scale), view.event_cap)
    elif view.family == "spatial":
        indices, marker, fallback = _spatial(graph, anchor, float(view.scale), view.event_cap)
    elif view.family == "random":
        indices, marker, fallback = _random_control(graph, anchor, view, selector_seed)
    elif view.family == "matched_recency":
        indices, marker, fallback = _matched_recency(graph, anchor, view)
    else:  # pragma: no cover - Literal plus validated registry
        raise ValueError(f"Unsupported view family {view.family!r}")
    result = _selection_result(graph, indices, marker, fallback)
    if int(result.event_indices[-1]) != anchor:
        raise RuntimeError(f"View {view.name} did not retain its anchor")
    if int(result.event_indices.max()) > anchor:
        raise RuntimeError(f"View {view.name} selected a future Event")
    return result
