"""Low-rank task/age-conditioned propagation residuals."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


PROPAGATION_TASKS = ("event", "time", "position")
PROPAGATION_MODES = ("constant", "shared", "task")


def edge_type_name(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)


class AgePropagationGate(nn.Module):
    """Layer-specific age gates with shared or task-specific queries."""

    def __init__(self, mode: str, layers: int = 2) -> None:
        super().__init__()
        if mode not in {"shared", "task"}:
            raise ValueError(f"Age gate does not support mode {mode!r}")
        self.mode = mode
        self.age_embedding = nn.Embedding(80, 16)
        q0 = torch.empty(16)
        nn.init.normal_(q0, std=0.02)
        rows = 1 if mode == "shared" else len(PROPAGATION_TASKS)
        self.pooling_embeddings = nn.Parameter(q0.repeat(rows, 1))
        self.scorers = nn.ModuleList()
        for _ in range(layers):
            scorer = nn.Sequential(
                nn.Linear(32, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(scorer[-1].weight)
            nn.init.zeros_(scorer[-1].bias)
            self.scorers.append(scorer)

    def _query(self, task: str, count: int) -> torch.Tensor:
        index = 0 if self.mode == "shared" else PROPAGATION_TASKS.index(task)
        return self.pooling_embeddings[index].unsqueeze(0).expand(count, -1)

    def raw_scores(
        self, layer: int, task: str, ages: torch.Tensor
    ) -> torch.Tensor:
        age_state = self.age_embedding(ages.long())
        query = self._query(task, int(ages.numel()))
        return self.scorers[layer](torch.cat((age_state, query), dim=-1)).squeeze(-1)

    def forward(
        self, layer: int, task: str, ages: torch.Tensor
    ) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.raw_scores(layer, task, ages))


class LowRankRelationResidual(nn.Module):
    """Global incoming-mean relation adapter for one propagation layer."""

    def __init__(
        self,
        node_types: tuple[str, ...],
        edge_types: tuple[tuple[str, str, str], ...],
        rank: int = 8,
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.rank = rank
        self.down = nn.ModuleList(
            nn.Linear(64, rank, bias=False) for _ in edge_types
        )
        self.up = nn.ModuleDict(
            {node_type: nn.Linear(rank, 64) for node_type in node_types}
        )
        for layer in self.down:
            nn.init.xavier_uniform_(layer.weight)
        for layer in self.up.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.forward_calls = 0
        self.capture_diagnostics = False
        self.last_relation_norms: dict[str, float] = {}
        self.last_degree_by_type: dict[str, torch.Tensor] = {}

    @staticmethod
    def _edge_ages(
        edge_type: tuple[str, str, str],
        edge_index: torch.Tensor,
        event_ages: torch.Tensor,
    ) -> torch.Tensor | None:
        source_type, _, destination_type = edge_type
        if source_type == "event":
            return event_ages[edge_index[0]]
        if destination_type == "event":
            return event_ages[edge_index[1]]
        return None

    def forward(
        self,
        states: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        event_ages: torch.Tensor,
        gate: AgePropagationGate | None,
        layer_index: int,
        task: str,
    ) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        aggregates = {
            node_type: states[node_type].new_zeros(
                (states[node_type].shape[0], self.rank)
            )
            for node_type in self.node_types
        }
        degrees = {
            node_type: states[node_type].new_zeros(states[node_type].shape[0])
            for node_type in self.node_types
        }
        relation_norms: dict[str, float] = {}
        for relation_index, edge_type in enumerate(self.edge_types):
            edge_index = edge_index_dict.get(edge_type)
            if edge_index is None or edge_index.numel() == 0:
                relation_norms[edge_type_name(edge_type)] = 0.0
                continue
            source_type, _, destination_type = edge_type
            messages = self.down[relation_index](states[source_type][edge_index[0]])
            ages = self._edge_ages(edge_type, edge_index, event_ages)
            if gate is not None and ages is not None:
                messages = messages * gate(layer_index, task, ages).unsqueeze(-1)
            aggregates[destination_type] = aggregates[destination_type].index_add(
                0, edge_index[1], messages
            )
            degrees[destination_type] = degrees[destination_type].index_add(
                0,
                edge_index[1],
                torch.ones(
                    edge_index.shape[1],
                    device=edge_index.device,
                    dtype=degrees[destination_type].dtype,
                ),
            )
            if self.capture_diagnostics:
                relation_norms[edge_type_name(edge_type)] = float(
                    messages.detach().norm().cpu()
                )
        if self.capture_diagnostics:
            self.last_relation_norms = relation_norms
            self.last_degree_by_type = {
                node_type: degree.detach() for node_type, degree in degrees.items()
            }
        return {
            node_type: self.up[node_type](
                aggregates[node_type]
                / degrees[node_type].clamp_min(1.0).unsqueeze(-1)
            )
            for node_type in self.node_types
        }


def add_state_residual(
    base: dict[str, torch.Tensor], residual: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        node_type: state + residual[node_type]
        for node_type, state in base.items()
    }


def local_event_ages(graph: Any) -> torch.Tensor:
    ptr = graph["event"].ptr
    counts = ptr[1:] - ptr[:-1]
    local_position = torch.arange(int(ptr[-1]), device=ptr.device) - torch.repeat_interleave(
        ptr[:-1], counts
    )
    return torch.repeat_interleave(counts - 1, counts) - local_position


def source_age_mismatch_count(graph: Any) -> int:
    event = graph["event"]
    if not hasattr(event, "source_index"):
        raise ValueError("F80 Event nodes must expose source_index")
    ptr = event.ptr
    counts = ptr[1:] - ptr[:-1]
    anchor_source = event.source_index[ptr[1:] - 1]
    source_age = torch.repeat_interleave(anchor_source, counts) - event.source_index
    return int((source_age != local_event_ages(graph)).sum().item())
