"""Diagnostics for sample-state-conditioned relation gates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .model import StateAwarePartialL2FiveTaskHGT
from .training import _move_batch_to_device


LAYERS = ("shared_l1", "main_l2", "player_l2")
STATE_GROUPS = {
    "all": lambda state: torch.ones(state.shape[0], dtype=torch.bool, device=state.device),
    "CONTROL": lambda state: state[:, 0].bool(),
    "CONTESTED": lambda state: state[:, 1].bool(),
    "restart": lambda state: state[:, 2].bool(),
    "boundary": lambda state: state[:, 3].bool(),
    "switch": lambda state: state[:, 4].bool(),
    "snapshot_valid": lambda state: state[:, 7].bool(),
}


def _edge_counts(model: StateAwarePartialL2FiveTaskHGT, batch: dict[str, Any]) -> torch.Tensor:
    graph = batch["graphs"]["f80"]
    convolution = model.convolutions[0]
    count = torch.zeros(
        (batch["anchor_state"].shape[0], len(convolution.edge_types)),
        dtype=torch.float64,
        device=batch["anchor_state"].device,
    )
    for relation_index, edge_type in enumerate(convolution.edge_types):
        edge_index = graph[edge_type].edge_index
        if edge_index.numel() == 0:
            continue
        destination_batch = graph[edge_type[-1]].batch[edge_index[1]]
        count[:, relation_index] = torch.bincount(
            destination_batch, minlength=count.shape[0]
        ).to(torch.float64)
    return count


def gate_snapshot(
    model: StateAwarePartialL2FiveTaskHGT,
    batch: dict[str, Any],
) -> dict[str, Any]:
    """Small deterministic record for one diagnostic batch."""

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(batch)
    relation_names = ["|".join(value) for value in model.convolutions[0].edge_types]
    output = {
        layer: {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "min": float(values.min()),
            "max": float(values.max()),
            "by_relation": {
                name: float(values[:, index].mean())
                for index, name in enumerate(relation_names)
            },
        }
        for layer, values in model.gate_matrices().items()
    }
    model.train(was_training)
    return output


def collect_gate_diagnostics(
    model: StateAwarePartialL2FiveTaskHGT,
    loader: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Aggregate gates by layer, relation, and causal anchor state."""

    sums: dict[tuple[str, str, str], float] = defaultdict(float)
    squares: dict[tuple[str, str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    edge_sums: dict[tuple[str, str], float] = defaultdict(float)
    edge_counts: dict[tuple[str, str], float] = defaultdict(float)
    minima: dict[tuple[str, str, str], float] = {}
    maxima: dict[tuple[str, str, str], float] = {}
    state_counts: dict[str, int] = defaultdict(int)
    relation_names: list[str] | None = None
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch_to_device(raw_batch, device)
            model(batch)
            state = batch["anchor_state"]
            matrices = model.gate_matrices()
            relations = ["|".join(value) for value in model.convolutions[0].edge_types]
            relation_names = relations
            per_sample_edges = _edge_counts(model, batch)
            masks = {name: function(state) for name, function in STATE_GROUPS.items()}
            duration = torch.expm1(state[:, 5] * 5.0)
            event_count = state[:, 6] * 80.0
            masks.update({
                "duration_0_5s": state[:, 7].bool() & (duration < 5),
                "duration_5_15s": state[:, 7].bool() & (duration >= 5) & (duration < 15),
                "duration_15plus": state[:, 7].bool() & (duration >= 15),
                "events_1_2": state[:, 7].bool() & (event_count <= 2),
                "events_3_5": state[:, 7].bool() & (event_count > 2) & (event_count <= 5),
                "events_6plus": state[:, 7].bool() & (event_count > 5),
            })
            for group, mask in masks.items():
                state_counts[group] += int(mask.sum())
            for layer, matrix in matrices.items():
                values = matrix.to(torch.float64)
                for relation_index, relation in enumerate(relations):
                    relation_values = values[:, relation_index]
                    relation_edges = per_sample_edges[:, relation_index]
                    total_edges = float(relation_edges.sum())
                    if total_edges > 0:
                        edge_sums[(layer, relation)] += float(
                            (relation_values * relation_edges).sum()
                        )
                        edge_counts[(layer, relation)] += total_edges
                    for group, mask in masks.items():
                        active = mask & (relation_edges > 0)
                        if not active.any():
                            continue
                        selected = relation_values[active]
                        key = (layer, relation, group)
                        sums[key] += float(selected.sum())
                        squares[key] += float((selected * selected).sum())
                        counts[key] += int(selected.numel())
                        current_min = float(selected.min())
                        current_max = float(selected.max())
                        minima[key] = min(minima.get(key, current_min), current_min)
                        maxima[key] = max(maxima.get(key, current_max), current_max)
    model.train(was_training)
    if relation_names is None:
        raise RuntimeError("Gate diagnostic loader was empty")

    by_layer: dict[str, Any] = {}
    for layer in LAYERS:
        by_relation = {}
        for relation in relation_names:
            groups = {}
            for group in state_counts:
                key = (layer, relation, group)
                count = counts[key]
                if not count:
                    continue
                mean = sums[key] / count
                variance = max(0.0, squares[key] / count - mean * mean)
                groups[group] = {
                    "samples_with_relation": count,
                    "mean": mean,
                    "std": variance**0.5,
                    "min": minima[key],
                    "max": maxima[key],
                }
            by_relation[relation] = {
                "groups": groups,
                "edge_weighted_mean": (
                    edge_sums[(layer, relation)] / edge_counts[(layer, relation)]
                    if edge_counts[(layer, relation)] else None
                ),
                "edge_count": edge_counts[(layer, relation)],
            }
        by_layer[layer] = by_relation
    return {
        "gate_definition": "sample x layer x concrete edge type scalar",
        "state_dimensions": [
            "CONTROL", "CONTESTED", "restart", "boundary", "switch_confirmed",
            "log1p_duration_div_5", "event_count_div_80", "snapshot_mask",
        ],
        "state_group_counts": dict(state_counts),
        "layers": by_layer,
    }
