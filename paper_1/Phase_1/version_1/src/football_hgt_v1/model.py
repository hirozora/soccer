"""Heterogeneous Graph Transformer for six next-event prediction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HGTConv, global_mean_pool


NODE_TYPES = ["event", "player", "team", "tag"]
EDGE_TYPES = [
    ("event", "next", "event"),
    ("player", "performs", "event"),
    ("team", "performs", "event"),
    ("tag", "describes", "event"),
]
METADATA = (NODE_TYPES, EDGE_TYPES)


@dataclass(frozen=True)
class ModelConfig:
    num_event_types: int
    num_subevent_types: int
    num_players: int
    num_teams: int
    num_tags: int
    hidden_channels: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


class Projection(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class FootballHGT(nn.Module):
    """Encode causal event windows and emit six prediction heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.event_type_embedding = nn.Embedding(config.num_event_types, 16)
        self.subevent_embedding = nn.Embedding(config.num_subevent_types, 16)
        self.period_embedding = nn.Embedding(5, 4)
        self.result_embedding = nn.Embedding(3, 4)
        self.event_projection = Projection(49, config.hidden_channels, config.dropout)

        self.player_embedding = nn.Embedding(config.num_players, 24)
        self.player_role_embedding = nn.Embedding(5, 4)
        self.player_foot_embedding = nn.Embedding(4, 4)
        self.player_projection = Projection(35, config.hidden_channels, config.dropout)

        self.team_embedding = nn.Embedding(config.num_teams + 1, 16)
        self.team_side_embedding = nn.Embedding(3, 4)
        self.team_type_embedding = nn.Embedding(3, 4)
        self.team_projection = Projection(24, config.hidden_channels, config.dropout)

        self.tag_embedding = nn.Embedding(config.num_tags, 16)
        self.tag_category_embedding = nn.Embedding(3, 4)
        self.tag_projection = Projection(20, config.hidden_channels, config.dropout)

        self.convolutions = nn.ModuleList(
            HGTConv(
                in_channels=config.hidden_channels,
                out_channels=config.hidden_channels,
                metadata=METADATA,
                heads=config.num_heads,
            )
            for _ in range(config.num_layers)
        )
        self.event_norms = nn.ModuleList(
            nn.LayerNorm(config.hidden_channels) for _ in range(config.num_layers)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.context_projection = Projection(
            2 * config.hidden_channels, config.hidden_channels, config.dropout
        )

        self.event_head = nn.Linear(config.hidden_channels, config.num_event_types)
        self.time_head = nn.Linear(config.hidden_channels, 1)
        self.position_head = nn.Linear(config.hidden_channels, 2)
        self.side_head = nn.Linear(config.hidden_channels, 2)
        self.advantage_head = nn.Linear(config.hidden_channels, 2)
        self.player_scorer = nn.Sequential(
            nn.Linear(2 * config.hidden_channels, config.hidden_channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, 1),
        )

    def encode_nodes(self, batch: Any) -> dict[str, torch.Tensor]:
        event = batch["event"]
        event_numeric = torch.cat(
            (
                (event.period_seconds / 3600.0).unsqueeze(-1),
                (event.absolute_seconds / 7200.0).unsqueeze(-1),
                torch.log1p(event.delta_from_previous.clamp_min(0.0)).unsqueeze(-1),
                event.start_position,
                event.start_position_mask.float().unsqueeze(-1),
                event.end_position,
                event.end_position_mask.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        event_features = torch.cat(
            (
                self.event_type_embedding(event.event_type_index),
                self.subevent_embedding(event.subevent_type_index),
                self.period_embedding(event.period_index),
                self.result_embedding(event.result_direction),
                event_numeric,
            ),
            dim=-1,
        )

        player = batch["player"]
        player_numeric = torch.cat(
            (
                player.height_weight[:, :1] / 200.0,
                player.height_weight[:, 1:] / 100.0,
                player.metadata_mask.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        player_features = torch.cat(
            (
                self.player_embedding(player.vocab_index),
                self.player_role_embedding(player.role_index),
                self.player_foot_embedding(player.foot_index),
                player_numeric,
            ),
            dim=-1,
        )

        team = batch["team"]
        team_vocab = (team.vocab_index + 1).clamp_min(0)
        team_features = torch.cat(
            (
                self.team_embedding(team_vocab),
                self.team_side_embedding(team.side_index),
                self.team_type_embedding(team.type_index),
            ),
            dim=-1,
        )

        tag = batch["tag"]
        tag_features = torch.cat(
            (
                self.tag_embedding(tag.vocab_index),
                self.tag_category_embedding(tag.category_index),
            ),
            dim=-1,
        )
        return {
            "event": self.event_projection(event_features),
            "player": self.player_projection(player_features),
            "team": self.team_projection(team_features),
            "tag": self.tag_projection(tag_features),
        }

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        representations = self.encode_nodes(batch)
        for convolution, event_norm in zip(self.convolutions, self.event_norms):
            updated = convolution(representations, batch.edge_index_dict)
            event_update = updated.get("event")
            if event_update is None:
                raise RuntimeError("HGT produced no Event representation")
            # HGTConv already applies its learned skip gate internally.
            representations["event"] = event_norm(self.dropout(event_update))

        event_ptr = batch["event"].ptr
        query_indices = event_ptr[1:] - 1
        last_event = representations["event"][query_indices]
        pooled_events = global_mean_pool(
            representations["event"], batch["event"].batch
        )
        context = self.context_projection(torch.cat((last_event, pooled_events), dim=-1))

        player_batch = batch["player"].batch
        player_scores = self.player_scorer(
            torch.cat((context[player_batch], representations["player"]), dim=-1)
        ).squeeze(-1)
        return {
            "context": context,
            "event_logits": self.event_head(context),
            "log_delta": F.softplus(self.time_head(context).squeeze(-1)),
            "position": torch.sigmoid(self.position_head(context)),
            "side_logits": self.side_head(context),
            "advantage_logits": self.advantage_head(context),
            "player_scores": player_scores,
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return values.sum() * 0.0


def player_cross_entropy(
    scores: torch.Tensor,
    player_ptr: torch.Tensor,
    local_targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        start = int(player_ptr[index])
        stop = int(player_ptr[index + 1])
        target = start + int(local_targets[index])
        losses.append(-(scores[target] - torch.logsumexp(scores[start:stop], dim=0)))
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def compute_multitask_loss(
    predictions: dict[str, torch.Tensor], batch: Any
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the Version 1 six-task objective and masking policy."""

    event_loss = F.cross_entropy(predictions["event_logits"], batch.y_event_type)
    time_loss = F.huber_loss(predictions["log_delta"], batch.y_log_delta)

    position_values = F.huber_loss(
        predictions["position"], batch.y_position, reduction="none"
    ).mean(dim=-1)
    position_loss = _masked_mean(position_values, batch.y_position_mask.bool())
    side_loss = F.cross_entropy(predictions["side_logits"], batch.y_side)
    player_loss = player_cross_entropy(
        predictions["player_scores"],
        batch["player"].ptr,
        batch.y_player_local,
        batch.y_player_mask.bool(),
    )
    advantage_values = F.cross_entropy(
        predictions["advantage_logits"],
        batch.y_advantage.clamp_min(0),
        reduction="none",
    )
    advantage_loss = _masked_mean(
        advantage_values, batch.y_advantage_mask.bool()
    )
    losses = {
        "event": event_loss,
        "time": time_loss,
        "position": position_loss,
        "side": side_loss,
        "player": player_loss,
        "advantage": advantage_loss,
    }
    total = (
        losses["event"]
        + 0.3 * losses["time"]
        + 0.5 * losses["position"]
        + 0.3 * losses["side"]
        + 0.5 * losses["player"]
        + 0.5 * losses["advantage"]
    )
    return total, losses
