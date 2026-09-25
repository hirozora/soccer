"""Single-HGT data-aware residual mixture of experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.nn import global_mean_pool

from football_hgt_v1.model import FootballHGT, ModelConfig

from .data import CANDIDATE_FEATURE_NAMES
from .experts import EXPERT_NAMES, STAT_NAMES


TASK_NAMES = ("event", "time", "position", "side", "player", "advantage")
TASK_INDEX = {name: index for index, name in enumerate(TASK_NAMES)}


@dataclass(frozen=True)
class DataAwareMoEConfig:
    base: ModelConfig
    expert_mode: str = "homogeneous"
    residual_mode: str = "task_specific"
    routing_mode: str = "dense"
    conditioned_router: bool = True
    position_correction: bool = False
    player_correction: bool = False
    router_hidden_channels: int = 128
    task_embedding_channels: int = 16
    expert_embedding_channels: int = 8
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.expert_mode not in {"homogeneous", "structural"}:
            raise ValueError("expert_mode must be homogeneous or structural")
        if self.residual_mode not in {"shared", "task_specific"}:
            raise ValueError("residual_mode must be shared or task_specific")
        if self.routing_mode not in {"uniform", "dense"}:
            raise ValueError("routing_mode must be uniform or dense")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _last_selected(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    index = mask.long().sum(dim=1).clamp_min(1) - 1
    return values[torch.arange(values.shape[0], device=values.device), index]


def _group_mean(
    values: torch.Tensor, mask: torch.Tensor, group: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = mask & group
    return _masked_mean(values, selected), selected.any(dim=1, keepdim=True).float()


class HomogeneousReadout(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, states: torch.Tensor, selection: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = selection["selection_mask"]
        return self.output(torch.cat((_last_selected(states, mask), _masked_mean(states, mask)), dim=-1))


class ShortTemporalReadout(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.time_projection = nn.Linear(2, hidden)
        self.attention = nn.MultiheadAttention(
            hidden, num_heads=4, dropout=dropout, batch_first=True
        )
        self.output = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, states: torch.Tensor, selection: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = selection["selection_mask"]
        auxiliary = torch.stack(
            (selection["relative_index"], selection["relative_time"]), dim=-1
        )
        values = states + self.time_projection(auxiliary)
        length = values.shape[1]
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=values.device), diagonal=1
        )
        attended, _ = self.attention(
            values,
            values,
            values,
            attn_mask=causal,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return self.output(
            torch.cat((_last_selected(attended, mask), _masked_mean(attended, mask)), dim=-1)
        )


class TeamSequenceReadout(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(3 * hidden + 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, states: torch.Tensor, selection: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = selection["selection_mask"]
        relation = selection["team_relation"]
        same, same_present = _group_mean(states, mask, relation == 0)
        other, other_present = _group_mean(states, mask, relation == 1)
        boundary, boundary_present = _group_mean(states, mask, selection["boundary"])
        return self.output(
            torch.cat(
                (same, other, boundary, same_present, other_present, boundary_present), dim=-1
            )
        )


class SpatialReadout(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))
        self.output = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, states: torch.Tensor, selection: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = selection["selection_mask"]
        logits = self.score(selection["spatial_features"]).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bk,bkh->bh", weights, states)
        return self.output(torch.cat((_last_selected(states, mask), pooled), dim=-1))


class ActorRelationReadout(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(3 * hidden + 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, states: torch.Tensor, selection: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = selection["selection_mask"]
        relation = selection["actor_relation"]
        current, current_present = _group_mean(states, mask, relation == 0)
        same, same_present = _group_mean(states, mask, relation == 1)
        opponent, opponent_present = _group_mean(states, mask, relation == 2)
        return self.output(
            torch.cat(
                (current, same, opponent, current_present, same_present, opponent_present),
                dim=-1,
            )
        )


class TransitionReadout(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.auxiliary = nn.Linear(4, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, states: torch.Tensor, selection: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = selection["selection_mask"]
        values = states + self.auxiliary(selection["transition_features"])
        lengths = mask.long().sum(dim=1).cpu()
        packed = pack_padded_sequence(values, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.output(torch.cat((_last_selected(states, mask), hidden[-1]), dim=-1))


class ResidualStem(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        input_channels = 4 * hidden + len(STAT_NAMES)
        self.norm = nn.LayerNorm(input_channels)
        self.linear = nn.Linear(input_channels, hidden)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(
        self, evidence: torch.Tensor, context: torch.Tensor, statistics: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat(
            (evidence, context, evidence - context, evidence * context, statistics), dim=-1
        )
        return self.dropout(F.gelu(self.linear(self.norm(features))))


def _zero_linear(input_channels: int, output_channels: int) -> nn.Linear:
    layer = nn.Linear(input_channels, output_channels)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class DataAwareResidualMoE(nn.Module):
    """Frozen V1 K=80 backbone with post-HGT structural residual experts."""

    def __init__(self, config: DataAwareMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.base = FootballHGT(config.base)
        hidden = config.base.hidden_channels
        self.hgt_forward_calls = 0

        if config.expert_mode == "homogeneous":
            readouts = {name: HomogeneousReadout(hidden, config.dropout) for name in EXPERT_NAMES}
        else:
            readouts = {
                "short_temporal": ShortTemporalReadout(hidden, config.dropout),
                "team_sequence": TeamSequenceReadout(hidden, config.dropout),
                "spatial": SpatialReadout(hidden, config.dropout),
                "actor_relation": ActorRelationReadout(hidden, config.dropout),
                "transition": TransitionReadout(hidden, config.dropout),
            }
        self.readouts = nn.ModuleDict(readouts)
        self.residual_stems = nn.ModuleList(
            ResidualStem(hidden, config.dropout) for _ in EXPERT_NAMES
        )
        if config.residual_mode == "shared":
            self.shared_residual_outputs = nn.ModuleList(
                _zero_linear(hidden, hidden) for _ in EXPERT_NAMES
            )
            self.task_residual_outputs = None
        else:
            self.shared_residual_outputs = None
            self.task_residual_outputs = nn.ModuleList(
                _zero_linear(hidden, hidden)
                for _ in range(len(TASK_NAMES) * len(EXPERT_NAMES))
            )

        self.expert_embedding = nn.Embedding(
            len(EXPERT_NAMES), config.expert_embedding_channels
        )
        self.task_embedding = nn.Embedding(
            len(TASK_NAMES), config.task_embedding_channels
        )
        router_input = 3 * hidden + len(STAT_NAMES) + config.expert_embedding_channels
        self.router_evidence = nn.Sequential(
            nn.Linear(router_input, config.router_hidden_channels),
            nn.LayerNorm(config.router_hidden_channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        task_router_input = (
            config.router_hidden_channels + config.task_embedding_channels + 12
        )
        self.router_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(task_router_input, 64),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(64, 1),
            )
            for _ in TASK_NAMES
        )

        if config.position_correction:
            position_input = hidden + 2 + 2 + 2 + 10 + 2
            self.position_transport_scale = nn.Parameter(torch.zeros(2))
            self.position_delta = nn.Sequential(
                nn.Linear(position_input, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                _zero_linear(hidden, 2),
            )
        else:
            self.register_parameter("position_transport_scale", None)
            self.position_delta = None

        if config.player_correction:
            player_input = 2 * hidden + len(CANDIDATE_FEATURE_NAMES) + 10 + 2
            self.player_delta = nn.Sequential(
                nn.Linear(player_input, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                _zero_linear(hidden, 1),
            )
        else:
            self.player_delta = None
        self.freeze_base()

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def train(self, mode: bool = True) -> "DataAwareResidualMoE":
        super().train(mode)
        self.base.eval()
        return self

    def _encode_full_history(
        self, graph: Any
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        self.hgt_forward_calls += 1
        with torch.no_grad():
            representations = self.base.encode_nodes(graph)
            for convolution, event_norm in zip(
                self.base.convolutions, self.base.event_norms
            ):
                updated = convolution(representations, graph.edge_index_dict)
                event_update = updated.get("event")
                if event_update is None:
                    raise RuntimeError("HGT produced no Event representation")
                representations["event"] = event_norm(self.base.dropout(event_update))
            event_ptr = graph["event"].ptr
            query = event_ptr[1:] - 1
            last = representations["event"][query]
            pooled = global_mean_pool(representations["event"], graph["event"].batch)
            context = self.base.context_projection(torch.cat((last, pooled), dim=-1))
        return representations, context

    def _expert_evidence(
        self,
        event_states: torch.Tensor,
        selections: dict[str, dict[str, torch.Tensor]],
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evidence_rows = []
        stem_rows = []
        statistics_rows = []
        for index, name in enumerate(EXPERT_NAMES):
            selection = selections[name]
            states = event_states[selection["event_indices"]]
            evidence = self.readouts[name](states, selection)
            statistics = selection["structural_features"]
            stem = self.residual_stems[index](evidence, context, statistics)
            evidence_rows.append(evidence)
            stem_rows.append(stem)
            statistics_rows.append(statistics)
        return (
            torch.stack(evidence_rows, dim=1),
            torch.stack(stem_rows, dim=1),
            torch.stack(statistics_rows, dim=1),
        )

    def _task_residuals(self, stems: torch.Tensor) -> torch.Tensor:
        batch_size, num_experts, hidden = stems.shape
        if self.shared_residual_outputs is not None:
            shared = torch.stack(
                [layer(stems[:, index]) for index, layer in enumerate(self.shared_residual_outputs)],
                dim=1,
            )
            return shared[:, None].expand(batch_size, len(TASK_NAMES), num_experts, hidden)
        assert self.task_residual_outputs is not None
        rows = []
        for task_index in range(len(TASK_NAMES)):
            task_rows = []
            for expert_index in range(num_experts):
                layer = self.task_residual_outputs[
                    task_index * num_experts + expert_index
                ]
                task_rows.append(layer(stems[:, expert_index]))
            rows.append(torch.stack(task_rows, dim=1))
        return torch.stack(rows, dim=1)

    def _router_features(
        self,
        context: torch.Tensor,
        evidence: torch.Tensor,
        statistics: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_experts, _ = evidence.shape
        context_rows = context[:, None].expand(-1, num_experts, -1)
        expert_ids = torch.arange(num_experts, device=context.device)
        expert_embedding = self.expert_embedding(expert_ids)[None].expand(batch_size, -1, -1)
        values = torch.cat(
            (context_rows, evidence, evidence - context_rows, statistics, expert_embedding),
            dim=-1,
        )
        return self.router_evidence(values)

    def _route(
        self,
        task_index: int,
        router_features: torch.Tensor,
        condition: torch.Tensor,
        disabled_expert: int | None,
    ) -> torch.Tensor:
        batch_size, num_experts, _ = router_features.shape
        if self.config.routing_mode == "uniform":
            weights = torch.ones(
                (batch_size, num_experts), device=router_features.device
            )
        else:
            task_ids = torch.full(
                (batch_size, num_experts), task_index, dtype=torch.long, device=router_features.device
            )
            task = self.task_embedding(task_ids)
            condition_rows = condition[:, None].expand(-1, num_experts, -1)
            logits = self.router_heads[task_index](
                torch.cat((router_features, task, condition_rows), dim=-1)
            ).squeeze(-1)
            weights = torch.softmax(logits, dim=-1)
        if disabled_expert is not None:
            weights = weights.clone()
            weights[:, disabled_expert] = 0.0
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _condition(
        event_probability: torch.Tensor | None,
        side_probability: torch.Tensor | None,
        enabled: bool,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        result = torch.zeros((batch_size, 12), device=device)
        if enabled and event_probability is not None:
            result[:, :10] = event_probability.detach()
        if enabled and side_probability is not None:
            result[:, 10:] = side_probability.detach()
        return result

    def forward(
        self,
        graph: Any,
        selections: dict[str, dict[str, torch.Tensor]],
        candidate_features: torch.Tensor,
        disabled_expert: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if int(graph.num_graphs) != int(graph.y_event_type.numel()):
            raise ValueError("Version 3 expects exactly one graph per sample")
        batch_size = int(graph.num_graphs)
        representations, context = self._encode_full_history(graph)
        evidence, stems, statistics = self._expert_evidence(
            representations["event"], selections, context
        )
        residuals = self._task_residuals(stems)
        router_features = self._router_features(context, evidence, statistics)
        zero_condition = self._condition(None, None, False, batch_size, context.device)

        task_contexts: list[torch.Tensor | None] = [None] * len(TASK_NAMES)
        routing: list[torch.Tensor | None] = [None] * len(TASK_NAMES)

        def build_task(task_name: str, condition: torch.Tensor) -> torch.Tensor:
            task_index = TASK_INDEX[task_name]
            weights = self._route(task_index, router_features, condition, disabled_expert)
            routing[task_index] = weights
            mixed = torch.einsum("bm,bmh->bh", weights, residuals[:, task_index])
            task_contexts[task_index] = context + mixed
            return task_contexts[task_index]  # type: ignore[return-value]

        event_context = build_task("event", zero_condition)
        event_logits = self.base.event_head(event_context)
        event_probability = torch.softmax(event_logits, dim=-1)

        event_condition = self._condition(
            event_probability,
            None,
            self.config.conditioned_router,
            batch_size,
            context.device,
        )
        side_context = build_task("side", event_condition)
        side_logits = self.base.side_head(side_context)
        side_probability = torch.softmax(side_logits, dim=-1)
        full_condition = self._condition(
            event_probability,
            side_probability,
            self.config.conditioned_router,
            batch_size,
            context.device,
        )

        time_context = build_task("time", event_condition)
        position_context = build_task("position", full_condition)
        player_context = build_task("player", full_condition)
        advantage_context = build_task("advantage", event_condition)

        log_delta = F.softplus(self.base.time_head(time_context).squeeze(-1))
        base_position = torch.sigmoid(self.base.position_head(position_context))
        position = base_position
        event_ptr = graph["event"].ptr
        query = event_ptr[1:] - 1
        event = graph["event"]
        anchor_position = torch.where(
            event.end_position_mask[query, None],
            event.end_position[query],
            event.start_position[query],
        )
        transport = (
            side_probability[:, 1:2].detach() * anchor_position
            + side_probability[:, 0:1].detach() * (1.0 - anchor_position)
        )
        if self.position_delta is not None:
            position_features = torch.cat(
                (
                    position_context,
                    transport,
                    base_position,
                    transport - base_position,
                    event_probability.detach(),
                    side_probability.detach(),
                ),
                dim=-1,
            )
            correction = self.position_delta(position_features)
            position = torch.clamp(
                base_position
                + torch.tanh(self.position_transport_scale) * (transport - base_position)
                + correction,
                0.0,
                1.0,
            )

        player_batch = graph["player"].batch
        player_representation = representations["player"]
        player_scores = self.base.player_scorer(
            torch.cat((player_context[player_batch], player_representation), dim=-1)
        ).squeeze(-1)
        if self.player_delta is not None:
            player_features = torch.cat(
                (
                    player_context[player_batch],
                    player_representation,
                    candidate_features,
                    event_probability.detach()[player_batch],
                    side_probability.detach()[player_batch],
                ),
                dim=-1,
            )
            player_scores = player_scores + self.player_delta(player_features).squeeze(-1)

        advantage_logits = self.base.advantage_head(advantage_context)
        base_player_scores = self.base.player_scorer(
            torch.cat((context[player_batch], player_representation), dim=-1)
        ).squeeze(-1)
        resolved_contexts = torch.stack(
            [value for value in task_contexts if value is not None], dim=1
        )
        resolved_routing = torch.stack(
            [value for value in routing if value is not None], dim=1
        )
        return {
            "event_logits": event_logits,
            "log_delta": log_delta,
            "position": position,
            "side_logits": side_logits,
            "player_scores": player_scores,
            "advantage_logits": advantage_logits,
            "base_event_logits": self.base.event_head(context),
            "base_log_delta": F.softplus(self.base.time_head(context).squeeze(-1)),
            "base_position": torch.sigmoid(self.base.position_head(context)),
            "base_side_logits": self.base.side_head(context),
            "base_player_scores": base_player_scores,
            "base_advantage_logits": self.base.advantage_head(context),
            "base_context": context,
            "expert_evidence": evidence,
            "expert_stems": stems,
            "task_residuals": residuals,
            "task_contexts": resolved_contexts,
            "routing_weights": resolved_routing,
            "transport_position": transport,
        }


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
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def _masked_mean_loss(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].mean() if bool(mask.any()) else values.sum() * 0.0


def compute_multitask_loss(
    predictions: dict[str, torch.Tensor], graph: Any
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    event_loss = F.cross_entropy(predictions["event_logits"], graph.y_event_type)
    time_loss = F.huber_loss(predictions["log_delta"], graph.y_log_delta)
    position_values = F.huber_loss(
        predictions["position"], graph.y_position, reduction="none"
    ).mean(dim=-1)
    position_loss = _masked_mean_loss(position_values, graph.y_position_mask.bool())
    side_loss = F.cross_entropy(predictions["side_logits"], graph.y_side)
    player_loss = player_cross_entropy(
        predictions["player_scores"],
        graph["player"].ptr,
        graph.y_player_local,
        graph.y_player_mask.bool(),
    )
    advantage_values = F.cross_entropy(
        predictions["advantage_logits"], graph.y_advantage.clamp_min(0), reduction="none"
    )
    advantage_loss = _masked_mean_loss(
        advantage_values, graph.y_advantage_mask.bool()
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
        event_loss
        + 0.3 * time_loss
        + 0.5 * position_loss
        + 0.3 * side_loss
        + 0.5 * player_loss
        + 0.5 * advantage_loss
    )
    return total, losses
