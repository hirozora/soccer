"""Sample-state-conditioned relation gating for PyG HGT propagation."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.nn.conv.hgt_conv import construct_bipartite_edge_index
from torch_geometric.utils import softmax


class RelationGateController(nn.Module):
    """One scalar gate per sample and concrete edge type."""

    def __init__(self, edge_types: tuple[tuple[str, str, str], ...]) -> None:
        super().__init__()
        self.edge_types = edge_types
        self.relation_embedding = nn.Embedding(len(edge_types), 16)
        self.network = nn.Sequential(
            nn.Linear(24, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, anchor_state: torch.Tensor) -> torch.Tensor:
        if anchor_state.ndim != 2 or anchor_state.shape[1] != 8:
            raise ValueError("anchor_state must have shape [batch, 8]")
        batch_size = anchor_state.shape[0]
        relation = self.relation_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        state = anchor_state.unsqueeze(1).expand(-1, len(self.edge_types), -1)
        return 2.0 * torch.sigmoid(
            self.network(torch.cat((state, relation), dim=-1)).squeeze(-1)
        )


class StateAwareHGTConv(HGTConv):
    """HGTConv with a sample-level scalar on each relation's messages."""

    def __init__(
        self,
        in_channels: int | dict[str, int],
        out_channels: int,
        metadata: tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]],
        heads: int = 1,
    ) -> None:
        super().__init__(in_channels, out_channels, metadata, heads=heads)
        self.gate_controller = RelationGateController(tuple(metadata[1]))
        self.last_gate_matrix: torch.Tensor | None = None

    def edge_gate_vector(
        self,
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        node_batch_dict: dict[str, torch.Tensor],
        gate_matrix: torch.Tensor,
        edge_type_order: tuple[tuple[str, str, str], ...] | None = None,
    ) -> torch.Tensor:
        """Broadcast sample/relation gates in PyG's concrete-edge order."""

        edge_gates = []
        for edge_type in edge_type_order or tuple(edge_index_dict):
            local_edges = edge_index_dict[edge_type]
            destination_batch = node_batch_dict[edge_type[-1]][local_edges[1]]
            relation_index = self.edge_types_map[edge_type]
            edge_gates.append(gate_matrix[destination_batch, relation_index])
        return torch.cat(edge_gates, dim=0)

    @classmethod
    def from_hgt(
        cls,
        source: HGTConv,
        metadata: tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]],
        *,
        controller_source: "StateAwareHGTConv | None" = None,
    ) -> "StateAwareHGTConv":
        target = cls(source.in_channels, source.out_channels, metadata, source.heads)
        target.load_state_dict(source.state_dict(), strict=False)
        if controller_source is not None:
            target.gate_controller.load_state_dict(
                controller_source.gate_controller.state_dict()
            )
        return target

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        anchor_state: torch.Tensor,
        node_batch_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        heads = self.heads
        width = self.out_channels // heads
        k_dict: dict[str, torch.Tensor] = {}
        q_dict: dict[str, torch.Tensor] = {}
        v_dict: dict[str, torch.Tensor] = {}
        out_dict: dict[str, torch.Tensor] = {}

        kqv_dict = self.kqv_lin(x_dict)
        for key, values in kqv_dict.items():
            key_state, query, value = torch.tensor_split(values, 3, dim=1)
            k_dict[key] = key_state.view(-1, heads, width)
            q_dict[key] = query.view(-1, heads, width)
            v_dict[key] = value.view(-1, heads, width)

        query, destination_offsets = self._cat(q_dict)
        key, value, source_offsets = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict
        )
        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict,
            source_offsets,
            destination_offsets,
            edge_attr_dict=self.p_rel,
            num_nodes=key.size(0),
        )
        gate_matrix = self.gate_controller(anchor_state)
        self.last_gate_matrix = gate_matrix.detach()
        edge_gate = self.edge_gate_vector(
            edge_index_dict,
            node_batch_dict,
            gate_matrix,
            tuple(source_offsets),
        )

        output = self.propagate(
            edge_index,
            k=key,
            q=query,
            v=value,
            edge_attr=edge_attr,
            edge_gate=edge_gate,
        )
        for node_type, start in destination_offsets.items():
            end = start + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = output[start:end]
        transformed = self.out_lin(
            {name: F.gelu(values) for name, values in out_dict.items()}
        )
        for node_type, values in transformed.items():
            if values.size(-1) == x_dict[node_type].size(-1):
                skip = self.skip[node_type].sigmoid()
                values = skip * values + (1.0 - skip) * x_dict[node_type]
            out_dict[node_type] = values
        return out_dict

    def message(
        self,
        k_j: torch.Tensor,
        q_i: torch.Tensor,
        v_j: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor | None,
        size_i: int | None,
    ) -> torch.Tensor:
        attention = (q_i * k_j).sum(dim=-1) * edge_attr
        attention = attention / math.sqrt(q_i.size(-1))
        attention = softmax(attention, index, ptr, size_i)
        output = v_j * attention.unsqueeze(-1) * edge_gate.view(-1, 1, 1)
        return output.view(-1, self.out_channels)


def base_hgt_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return parameters shared with Standard HGT, excluding gate controllers."""

    return {
        name: value
        for name, value in module.state_dict().items()
        if ".gate_controller." not in name
        and not name.startswith("gate_controller.")
    }
