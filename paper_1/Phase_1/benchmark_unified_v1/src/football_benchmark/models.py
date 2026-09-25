"""Feature-aligned HGT and published baseline architecture families."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .protocol import ProtocolArtifacts
from .semantic_graph import SEMANTIC_EDGE_TYPES, SEMANTIC_NODE_TYPES


@dataclass(frozen=True)
class ModelSpec:
    family: str
    contract: str
    window_size: int
    dropout: float = 0.1
    graph_variant: str = "legacy"


class Projection(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class EventTokenEncoder(nn.Module):
    """Encode exactly the raw causal fields available to the HGT graph."""

    def __init__(
        self, artifacts: ProtocolArtifacts, output_dim: int, dropout: float
    ) -> None:
        super().__init__()
        cardinalities = artifacts.cardinalities
        self.event = nn.Embedding(cardinalities["event"], 16)
        self.subevent = nn.Embedding(cardinalities["subevent"], 16)
        self.period = nn.Embedding(cardinalities["period"], 4)
        self.result = nn.Embedding(cardinalities["result"], 4)
        self.player = nn.Embedding(artifacts.num_players, 16)
        self.role = nn.Embedding(cardinalities["role"], 4)
        self.foot = nn.Embedding(cardinalities["foot"], 4)
        self.team = nn.Embedding(artifacts.num_teams, 8)
        self.team_side = nn.Embedding(cardinalities["team_side"], 2)
        self.team_type = nn.Embedding(cardinalities["team_type"], 4)
        self.tag_embedding = nn.Parameter(torch.empty(artifacts.num_tags, 8))
        nn.init.normal_(self.tag_embedding, std=0.02)
        self.numeric = Projection(12, 16, dropout)
        self.output = Projection(102, output_dim, dropout)

    def forward(
        self, sequence: dict[str, torch.Tensor], valid_mask: torch.Tensor
    ) -> torch.Tensor:
        tags = sequence["tags"]
        tag_count = tags.sum(dim=-1, keepdim=True).clamp_min(1.0)
        tag_state = tags @ self.tag_embedding / tag_count
        values = torch.cat(
            (
                self.event(sequence["event"]),
                self.subevent(sequence["subevent"]),
                self.period(sequence["period"]),
                self.result(sequence["result"]),
                self.player(sequence["player"]),
                self.role(sequence["role"]),
                self.foot(sequence["foot"]),
                self.team(sequence["team"]),
                self.team_side(sequence["team_side"]),
                self.team_type(sequence["team_type"]),
                tag_state,
                self.numeric(sequence["numeric"]),
            ),
            dim=-1,
        )
        return self.output(values) * valid_mask.unsqueeze(-1)


def sinusoidal_encoding(length: int, width: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    dimensions = torch.arange(width, device=device, dtype=torch.float32).unsqueeze(0)
    rates = torch.pow(10000.0, -2.0 * torch.floor(dimensions / 2.0) / width)
    angles = positions * rates
    return torch.where((dimensions.long() % 2) == 0, torch.sin(angles), torch.cos(angles))


class Seq2EventModel(nn.Module):
    """Official compact single-layer Transformer with feature-aligned tokens."""

    family = "seq2event"

    def __init__(self, artifacts: ProtocolArtifacts, dropout: float = 0.1) -> None:
        super().__init__()
        self.token_encoder = EventTokenEncoder(artifacts, 17, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=17,
            nhead=1,
            dim_feedforward=8,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.final = nn.Sequential(nn.Linear(17, 64), nn.ReLU())
        self.event_head = nn.Linear(64, 4)
        self.position_head = nn.Linear(64, 2)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        valid = batch["valid_mask"]
        values = self.token_encoder(batch["sequence"], valid)
        values = values + sinusoidal_encoding(
            values.shape[1], values.shape[2], values.device
        ).unsqueeze(0)
        encoded = self.encoder(values, src_key_padding_mask=~valid)
        context = self.final(encoded[:, -1])
        return {
            "event_logits": self.event_head(context),
            "position_xy": torch.sigmoid(self.position_head(context)),
        }


class UnifiedLEMModel(nn.Module):
    """Feature-aligned MLP LEM with a shared 101-token autoregressive output."""

    family = "unified_lem"

    def __init__(
        self,
        artifacts: ProtocolArtifacts,
        window_size: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.register_buffer("fine_active_mask", artifacts.fine_active_mask.bool())
        self.token_encoder = EventTokenEncoder(artifacts, 32, dropout)
        self.context = nn.Sequential(
            nn.Linear(window_size * 32, 196),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(196, 196),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(196, 196),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.task_embedding = nn.Embedding(4, 16)
        self.value_embedding = nn.Embedding(101, 16)
        self.condition = nn.Sequential(
            nn.Linear(196 + 16 + 3 * 16, 196),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shared_output = nn.Linear(196, 101)

    def _task_logits(
        self,
        context: torch.Tensor,
        task_index: int,
        previous: list[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        task = self.task_embedding(
            torch.full((batch_size,), task_index, dtype=torch.long, device=context.device)
        )
        prior_states = [self.value_embedding(value.clamp(0, 100)) for value in previous]
        while len(prior_states) < 3:
            prior_states.append(torch.zeros_like(task))
        conditioned = self.condition(torch.cat((context, task, *prior_states), dim=-1))
        return self.shared_output(conditioned)

    @staticmethod
    def _mask_width(logits: torch.Tensor, width: int) -> torch.Tensor:
        result = logits.clone()
        result[:, width:] = torch.finfo(result.dtype).min
        return result

    def forward(
        self, batch: dict[str, Any], teacher_forcing: bool | None = None
    ) -> dict[str, torch.Tensor]:
        if teacher_forcing is None:
            teacher_forcing = self.training
        valid = batch["valid_mask"]
        encoded = self.token_encoder(batch["sequence"], valid)
        if encoded.shape[1] != self.window_size:
            raise ValueError(
                f"Unified LEM expected width {self.window_size}, got {encoded.shape[1]}"
            )
        context = self.context(encoded.flatten(start_dim=1))
        targets = batch["targets"]

        event_logits = self._mask_width(self._task_logits(context, 0, []), 32)
        inactive = ~self.fine_active_mask
        event_logits[:, :32][:, inactive] = torch.finfo(event_logits.dtype).min
        event_value = (
            targets["fine_event_32"]
            if teacher_forcing
            else event_logits[:, :32].argmax(dim=-1)
        )

        x_logits = self._task_logits(context, 1, [event_value])
        x_target = torch.floor(targets["position_xy"][:, 0] * 100).long().clamp(0, 100)
        x_value = x_target if teacher_forcing else x_logits.argmax(dim=-1)

        y_logits = self._task_logits(context, 2, [event_value, x_value])
        y_target = torch.floor(targets["position_xy"][:, 1] * 100).long().clamp(0, 100)
        y_value = y_target if teacher_forcing else y_logits.argmax(dim=-1)

        time_logits = self._mask_width(
            self._task_logits(context, 3, [event_value, x_value, y_value]), 61
        )
        values_101 = torch.arange(101, device=context.device, dtype=context.dtype)
        position = torch.stack(
            (
                (x_logits.softmax(dim=-1) * values_101).sum(dim=-1) / 100.0,
                (y_logits.softmax(dim=-1) * values_101).sum(dim=-1) / 100.0,
            ),
            dim=-1,
        )
        time_values = values_101[:61]
        time_seconds = (
            time_logits[:, :61].softmax(dim=-1) * time_values
        ).sum(dim=-1)
        return {
            "fine_event_logits": event_logits[:, :32],
            "x_logits": x_logits,
            "y_logits": y_logits,
            "time_logits": time_logits[:, :61],
            "position_xy": position,
            "time_seconds": time_seconds,
        }


class NMSTPPModel(nn.Module):
    """Dependent time -> zone -> action Transformer point-process baseline."""

    family = "nmstpp"

    def __init__(self, artifacts: ProtocolArtifacts, dropout: float = 0.1) -> None:
        super().__init__()
        self.token_encoder = EventTokenEncoder(artifacts, 31, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=31,
            nhead=1,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.time_hidden = nn.Linear(31, 31)
        self.time_head = nn.Linear(31, 1)
        self.zone_hidden = nn.Linear(32, 32)
        self.zone_head = nn.Linear(32, 20)
        self.action_hidden = nn.Linear(52, 52)
        self.action_head = nn.Linear(52, 4)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        valid = batch["valid_mask"]
        values = self.token_encoder(batch["sequence"], valid)
        values = values + sinusoidal_encoding(
            values.shape[1], values.shape[2], values.device
        ).unsqueeze(0)
        encoded = self.encoder(values, src_key_padding_mask=~valid)[:, -1]
        time_state = F.relu(self.time_hidden(encoded))
        time_seconds = torch.sigmoid(self.time_head(time_state).squeeze(-1)) * 60.0
        zone_state = F.relu(
            self.zone_hidden(torch.cat((encoded, time_seconds.unsqueeze(-1) / 60.0), dim=-1))
        )
        zone_logits = self.zone_head(zone_state)
        action_state = F.relu(
            self.action_hidden(
                torch.cat((encoded, time_seconds.unsqueeze(-1) / 60.0, zone_logits), dim=-1)
            )
        )
        return {
            "time_seconds": time_seconds,
            "zone_logits": zone_logits,
            "event_logits": self.action_head(action_state),
        }


HGT_METADATA = (
    ("event", "player", "team", "tag"),
    (
        ("event", "next", "event"),
        ("player", "performs", "event"),
        ("team", "performs", "event"),
        ("tag", "describes", "event"),
    ),
)


class BenchmarkHGT(nn.Module):
    """The Version 1 two-layer HGT trunk with pair-specific target heads."""

    family = "hgt"

    def __init__(
        self,
        artifacts: ProtocolArtifacts,
        contract: str,
        hidden_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import HGTConv
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch-geometric is required for HGT") from exc
        self.contract = contract
        c = artifacts.cardinalities
        self.event_type_embedding = nn.Embedding(c["event"], 16)
        self.subevent_embedding = nn.Embedding(c["subevent"], 16)
        self.period_embedding = nn.Embedding(c["period"], 4)
        self.result_embedding = nn.Embedding(c["result"], 4)
        self.event_projection = Projection(49, hidden_channels, dropout)
        self.player_embedding = nn.Embedding(artifacts.num_players, 24)
        self.player_role_embedding = nn.Embedding(c["role"], 4)
        self.player_foot_embedding = nn.Embedding(c["foot"], 4)
        self.player_projection = Projection(35, hidden_channels, dropout)
        self.team_embedding = nn.Embedding(artifacts.num_teams, 16)
        self.team_side_embedding = nn.Embedding(c["team_side"], 4)
        self.team_type_embedding = nn.Embedding(c["team_type"], 4)
        self.team_projection = Projection(24, hidden_channels, dropout)
        self.tag_embedding = nn.Embedding(artifacts.num_tags, 16)
        self.tag_category_embedding = nn.Embedding(c["tag_category"], 4)
        self.tag_projection = Projection(20, hidden_channels, dropout)
        self.convolutions = nn.ModuleList(
            HGTConv(hidden_channels, hidden_channels, HGT_METADATA, heads=4)
            for _ in range(2)
        )
        self.event_norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(2))
        self.dropout = nn.Dropout(dropout)
        self.context_projection = Projection(2 * hidden_channels, hidden_channels, dropout)
        event_classes = 10 if contract == "unified_lem" else 4
        self.event_head = nn.Linear(hidden_channels, event_classes)
        self.time_head = nn.Linear(hidden_channels, 1) if contract != "seq2event" else None
        self.position_head = nn.Linear(hidden_channels, 2)

    def _encode_nodes(self, graph: Any) -> dict[str, torch.Tensor]:
        event = graph["event"]
        event_numeric = torch.cat(
            (
                (event.period_seconds / 3600.0).unsqueeze(-1),
                (event.absolute_seconds / 7200.0).unsqueeze(-1),
                torch.log1p(event.delta_from_previous.clamp_min(0)).unsqueeze(-1),
                event.start_position,
                event.start_position_mask.float().unsqueeze(-1),
                event.end_position,
                event.end_position_mask.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        event_state = self.event_projection(
            torch.cat(
                (
                    self.event_type_embedding(event.event_type_index),
                    self.subevent_embedding(event.subevent_type_index),
                    self.period_embedding(event.period_index),
                    self.result_embedding(event.result_direction),
                    event_numeric,
                ),
                dim=-1,
            )
        )
        player = graph["player"]
        player_numeric = torch.cat(
            (
                player.height_weight[:, :1] / 200.0,
                player.height_weight[:, 1:] / 100.0,
                player.metadata_mask.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        player_state = self.player_projection(
            torch.cat(
                (
                    self.player_embedding(player.vocab_index),
                    self.player_role_embedding(player.role_index),
                    self.player_foot_embedding(player.foot_index),
                    player_numeric,
                ),
                dim=-1,
            )
        )
        team = graph["team"]
        team_state = self.team_projection(
            torch.cat(
                (
                    self.team_embedding(team.vocab_index),
                    self.team_side_embedding(team.side_index),
                    self.team_type_embedding(team.type_index),
                ),
                dim=-1,
            )
        )
        tag = graph["tag"]
        tag_state = self.tag_projection(
            torch.cat(
                (
                    self.tag_embedding(tag.vocab_index),
                    self.tag_category_embedding(tag.category_index),
                ),
                dim=-1,
            )
        )
        return {"event": event_state, "player": player_state, "team": team_state, "tag": tag_state}

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        try:
            from torch_geometric.nn import global_mean_pool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch-geometric is required for HGT") from exc
        graph = batch["graph"]
        states = self._encode_nodes(graph)
        for convolution, norm in zip(self.convolutions, self.event_norms):
            updated = convolution(states, graph.edge_index_dict)
            event_update = updated.get("event")
            if event_update is None:
                raise RuntimeError("HGT produced no Event representation")
            states["event"] = norm(self.dropout(event_update))
        event_ptr = graph["event"].ptr
        anchors = states["event"][event_ptr[1:] - 1]
        means = global_mean_pool(states["event"], graph["event"].batch)
        context = self.context_projection(torch.cat((anchors, means), dim=-1))
        result = {
            "event_logits": self.event_head(context),
            "position_xy": torch.sigmoid(self.position_head(context)),
        }
        if self.time_head is not None:
            log_delta = F.softplus(self.time_head(context).squeeze(-1))
            result["time_seconds"] = torch.expm1(log_delta).clamp(max=60.0)
        return result


SEMANTIC_HGT_METADATA = (SEMANTIC_NODE_TYPES, SEMANTIC_EDGE_TYPES)

SEMANTIC_RELATION_FAMILIES = {
    "player": frozenset(
        {("player", "performs", "event"), ("event", "performed_by", "player")}
    ),
    "team": frozenset(
        {
            ("team", "performs", "event"),
            ("event", "performed_by_team", "team"),
        }
    ),
    "event_type": frozenset(
        {
            ("event", "has_type", "event_type"),
            ("event_type", "describes", "event"),
        }
    ),
    "tag": frozenset(
        {("event", "has_tag", "tag"), ("tag", "describes", "event")}
    ),
    "zone": frozenset(
        {
            ("event", "starts_in", "zone"),
            ("zone", "start_of", "event"),
            ("event", "ends_in", "zone"),
            ("zone", "end_of", "event"),
        }
    ),
    "temporal_bucket": frozenset(
        edge_type for edge_type in SEMANTIC_EDGE_TYPES if edge_type[1].startswith("gap_")
    )
    | frozenset({("event", "period_break", "event")}),
    "next": frozenset({("event", "next", "event")}),
}


class SemanticBenchmarkHGT(nn.Module):
    """Two-layer event-entity HGT with explicit temporal and spatial structure."""

    family = "hgt"

    def __init__(
        self,
        artifacts: ProtocolArtifacts,
        contract: str,
        hidden_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import HGTConv
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch-geometric is required for HGT") from exc
        self.contract = contract
        c = artifacts.cardinalities
        self.event_type_embedding = nn.Embedding(c["event"], 16)
        self.subevent_embedding = nn.Embedding(c["subevent"], 16)
        self.period_embedding = nn.Embedding(c["period"], 4)
        self.result_embedding = nn.Embedding(c["result"], 4)
        self.event_projection = Projection(56, hidden_channels, dropout)

        self.player_embedding = nn.Embedding(artifacts.num_players, 24)
        self.player_role_embedding = nn.Embedding(c["role"], 4)
        self.player_foot_embedding = nn.Embedding(c["foot"], 4)
        self.player_projection = Projection(35, hidden_channels, dropout)

        self.team_embedding = nn.Embedding(artifacts.num_teams, 16)
        self.team_side_embedding = nn.Embedding(c["team_side"], 4)
        self.team_type_embedding = nn.Embedding(c["team_type"], 4)
        self.team_projection = Projection(24, hidden_channels, dropout)

        self.type_node_embedding = nn.Embedding(c["event"], 24)
        self.type_node_projection = Projection(24, hidden_channels, dropout)
        self.tag_embedding = nn.Embedding(artifacts.num_tags, 16)
        self.tag_category_embedding = nn.Embedding(c["tag_category"], 4)
        self.tag_projection = Projection(20, hidden_channels, dropout)
        self.zone_embedding = nn.Embedding(20, 24)
        self.zone_projection = Projection(26, hidden_channels, dropout)

        self.convolutions = nn.ModuleList(
            HGTConv(hidden_channels, hidden_channels, SEMANTIC_HGT_METADATA, heads=4)
            for _ in range(2)
        )
        self.norms = nn.ModuleList(
            nn.ModuleDict(
                {
                    node_type: nn.LayerNorm(hidden_channels)
                    for node_type in SEMANTIC_NODE_TYPES
                }
            )
            for _ in range(2)
        )
        self.dropout = nn.Dropout(dropout)
        self.context_projection = Projection(2 * hidden_channels, hidden_channels, dropout)
        event_classes = 10 if contract == "unified_lem" else 4
        self.event_head = nn.Linear(hidden_channels, event_classes)
        self.time_head = nn.Linear(hidden_channels, 1) if contract != "seq2event" else None
        self.position_head = nn.Linear(hidden_channels, 2)

    def _encode_nodes(self, graph: Any) -> dict[str, torch.Tensor]:
        event = graph["event"]
        event_numeric = torch.cat(
            (
                (event.period_seconds / 3600.0).unsqueeze(-1),
                (event.absolute_seconds / 7200.0).unsqueeze(-1),
                torch.log1p(event.delta_from_previous.clamp_min(0)).unsqueeze(-1),
                event.start_position,
                event.start_position_mask.float().unsqueeze(-1),
                event.end_position,
                event.end_position_mask.float().unsqueeze(-1),
                event.relative_features,
            ),
            dim=-1,
        )
        event_state = self.event_projection(
            torch.cat(
                (
                    self.event_type_embedding(event.event_type_index),
                    self.subevent_embedding(event.subevent_type_index),
                    self.period_embedding(event.period_index),
                    self.result_embedding(event.result_direction),
                    event_numeric,
                ),
                dim=-1,
            )
        )

        player = graph["player"]
        player_numeric = torch.cat(
            (
                player.height_weight[:, :1] / 200.0,
                player.height_weight[:, 1:] / 100.0,
                player.metadata_mask.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        player_state = self.player_projection(
            torch.cat(
                (
                    self.player_embedding(player.vocab_index),
                    self.player_role_embedding(player.role_index),
                    self.player_foot_embedding(player.foot_index),
                    player_numeric,
                ),
                dim=-1,
            )
        )
        team = graph["team"]
        team_state = self.team_projection(
            torch.cat(
                (
                    self.team_embedding(team.vocab_index),
                    self.team_side_embedding(team.side_index),
                    self.team_type_embedding(team.type_index),
                ),
                dim=-1,
            )
        )
        event_type = graph["event_type"]
        event_type_state = self.type_node_projection(
            self.type_node_embedding(event_type.vocab_index)
        )
        tag = graph["tag"]
        tag_state = self.tag_projection(
            torch.cat(
                (
                    self.tag_embedding(tag.vocab_index),
                    self.tag_category_embedding(tag.category_index),
                ),
                dim=-1,
            )
        )
        zone = graph["zone"]
        zone_state = self.zone_projection(
            torch.cat((self.zone_embedding(zone.vocab_index), zone.center_xy), dim=-1)
        )
        return {
            "event": event_state,
            "player": player_state,
            "team": team_state,
            "event_type": event_type_state,
            "tag": tag_state,
            "zone": zone_state,
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        try:
            from torch_geometric.nn import global_mean_pool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch-geometric is required for HGT") from exc
        graph = batch["graph"]
        states = self._encode_nodes(graph)
        disabled = set(batch.get("disabled_relation_families", ()))
        unknown = disabled - set(SEMANTIC_RELATION_FAMILIES)
        if unknown:
            raise ValueError(f"Unknown semantic relation families: {sorted(unknown)}")
        disabled_edge_types = set().union(
            *(SEMANTIC_RELATION_FAMILIES[name] for name in disabled)
        ) if disabled else set()
        edge_index_dict = {
            edge_type: (
                torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)
                if edge_type in disabled_edge_types
                else edge_index
            )
            for edge_type, edge_index in graph.edge_index_dict.items()
        }
        for convolution, norms in zip(self.convolutions, self.norms):
            updated = convolution(states, edge_index_dict)
            states = {
                node_type: (
                    norms[node_type](state + self.dropout(updated[node_type]))
                    if node_type in updated
                    else state
                )
                for node_type, state in states.items()
            }
        event_ptr = graph["event"].ptr
        anchors = states["event"][event_ptr[1:] - 1]
        means = global_mean_pool(states["event"], graph["event"].batch)
        context = self.context_projection(torch.cat((anchors, means), dim=-1))
        result = {
            "event_logits": self.event_head(context),
            "position_xy": torch.sigmoid(self.position_head(context)),
        }
        if self.time_head is not None:
            log_delta = F.softplus(self.time_head(context).squeeze(-1))
            result["time_seconds"] = torch.expm1(log_delta).clamp(max=60.0)
        return result


def build_model(spec: ModelSpec, artifacts: ProtocolArtifacts) -> nn.Module:
    if spec.family == "hgt":
        if spec.graph_variant == "semantic_v2":
            return SemanticBenchmarkHGT(artifacts, spec.contract, dropout=spec.dropout)
        if spec.graph_variant != "legacy":
            raise ValueError(f"Unknown HGT graph variant {spec.graph_variant!r}")
        return BenchmarkHGT(artifacts, spec.contract, dropout=spec.dropout)
    if spec.graph_variant != "legacy":
        raise ValueError("Graph variants are only available for HGT")
    if spec.family == "seq2event":
        if spec.contract != "seq2event":
            raise ValueError("Seq2Event only supports the seq2event contract")
        return Seq2EventModel(artifacts, dropout=spec.dropout)
    if spec.family == "unified_lem":
        if spec.contract != "unified_lem":
            raise ValueError("Unified LEM only supports the unified_lem contract")
        return UnifiedLEMModel(artifacts, spec.window_size, dropout=0.3)
    if spec.family == "nmstpp":
        if spec.contract != "nmstpp":
            raise ValueError("NMSTPP only supports the nmstpp contract")
        return NMSTPPModel(artifacts, dropout=spec.dropout)
    raise ValueError(f"Unknown model family {spec.family!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
