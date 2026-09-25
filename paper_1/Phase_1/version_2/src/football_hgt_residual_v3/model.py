"""Version 3 full-history HGT with dense structural residual experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool

from football_hgt_v1.model import FootballHGT, ModelConfig

from .experts import BRANCH_NAMES, EXPERT_NAMES


TASK_NAMES = (
    "event",
    "time",
    "position",
    "side",
    "player",
    "advantage",
)


@dataclass(frozen=True)
class ResidualMoEConfig:
    base: ModelConfig
    adapter_channels: int = 16
    task_embedding_channels: int = 16
    router_hidden_channels: int = 128
    router_temperature: float = 1.0


class StructuralResidualAdapter(nn.Module):
    """Map full/expert contrast to a zero-initialized structural residual."""

    def __init__(self, hidden_channels: int, adapter_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(3 * hidden_channels)
        self.down = nn.Linear(3 * hidden_channels, adapter_channels)
        self.up = nn.Linear(adapter_channels, hidden_channels)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self, base_context: torch.Tensor, expert_context: torch.Tensor
    ) -> torch.Tensor:
        difference = expert_context - base_context
        features = self.norm(
            torch.cat((base_context, expert_context, difference), dim=-1)
        )
        return self.up(F.gelu(self.down(features)))


class FullHistoryResidualMoE(nn.Module):
    """Keep K=80 context intact and add dense task-conditioned residuals."""

    def __init__(self, config: ResidualMoEConfig) -> None:
        super().__init__()
        if config.router_temperature <= 0.0:
            raise ValueError("router_temperature must be positive")
        self.config = config
        self.base = FootballHGT(config.base)
        hidden = config.base.hidden_channels
        self.residual_adapters = nn.ModuleList(
            StructuralResidualAdapter(hidden, config.adapter_channels)
            for _ in EXPERT_NAMES
        )
        self.task_embedding = nn.Embedding(
            len(TASK_NAMES), config.task_embedding_channels
        )
        self.router = nn.Sequential(
            nn.Linear(
                (1 + len(EXPERT_NAMES)) * hidden
                + config.task_embedding_channels,
                config.router_hidden_channels,
            ),
            nn.LayerNorm(config.router_hidden_channels),
            nn.GELU(),
            nn.Linear(config.router_hidden_channels, len(EXPERT_NAMES)),
        )

    def set_base_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the complete K=80 HGT and its prediction heads."""

        for parameter in self.base.parameters():
            parameter.requires_grad_(trainable)

    def _encode_branch_contexts(
        self, graph_batch: Any, batch_size: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        representations = self.base.encode_nodes(graph_batch)
        for convolution, event_norm in zip(
            self.base.convolutions, self.base.event_norms
        ):
            updated = convolution(representations, graph_batch.edge_index_dict)
            event_update = updated.get("event")
            if event_update is None:
                raise RuntimeError("HGT produced no Event representation")
            representations["event"] = event_norm(self.base.dropout(event_update))

        event_ptr = graph_batch["event"].ptr
        query = event_ptr[1:] - 1
        last = representations["event"][query]
        pooled = global_mean_pool(
            representations["event"], graph_batch["event"].batch
        )
        contexts = self.base.context_projection(torch.cat((last, pooled), dim=-1))
        contexts = contexts.reshape(len(BRANCH_NAMES), batch_size, -1).transpose(0, 1)
        return representations, contexts

    def _dense_routing_weights(
        self,
        base_context: torch.Tensor,
        expert_contexts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_experts, hidden = expert_contexts.shape
        num_tasks = len(TASK_NAMES)
        task_ids = torch.arange(num_tasks, device=base_context.device)
        task = self.task_embedding(task_ids)

        branch_context = torch.cat(
            (base_context, expert_contexts.reshape(batch_size, num_experts * hidden)),
            dim=-1,
        )
        branch_context = branch_context[:, None, :].expand(
            batch_size, num_tasks, -1
        )
        task = task[None, :, :].expand(batch_size, num_tasks, -1)
        logits = self.router(torch.cat((branch_context, task), dim=-1))
        return torch.softmax(logits / self.config.router_temperature, dim=-1)

    def forward(self, graph_batch: Any, batch_size: int) -> dict[str, torch.Tensor]:
        expected_graphs = len(BRANCH_NAMES) * batch_size
        if int(graph_batch.num_graphs) != expected_graphs:
            raise ValueError(
                f"Expected {expected_graphs} graph branches, got {graph_batch.num_graphs}"
            )

        representations, contexts = self._encode_branch_contexts(
            graph_batch, batch_size
        )
        base_context = contexts[:, 0]
        expert_contexts = contexts[:, 1:]
        expert_residuals = torch.stack(
            [
                adapter(base_context, expert_contexts[:, expert_index])
                for expert_index, adapter in enumerate(self.residual_adapters)
            ],
            dim=1,
        )
        routing = self._dense_routing_weights(base_context, expert_contexts)
        task_contexts = base_context[:, None, :] + torch.einsum(
            "bqm,bmh->bqh", routing, expert_residuals
        )
        task_index = {name: index for index, name in enumerate(TASK_NAMES)}

        player_ptr = graph_batch["player"].ptr
        base_player_stop = int(player_ptr[batch_size])
        player_representation = representations["player"][:base_player_stop]
        player_batch = graph_batch["player"].batch[:base_player_stop]
        player_context = task_contexts[:, task_index["player"]]
        player_scores = self.base.player_scorer(
            torch.cat(
                (player_context[player_batch], player_representation), dim=-1
            )
        ).squeeze(-1)
        base_player_scores = self.base.player_scorer(
            torch.cat(
                (base_context[player_batch], player_representation), dim=-1
            )
        ).squeeze(-1)

        event_context = task_contexts[:, task_index["event"]]
        time_context = task_contexts[:, task_index["time"]]
        position_context = task_contexts[:, task_index["position"]]
        side_context = task_contexts[:, task_index["side"]]
        advantage_context = task_contexts[:, task_index["advantage"]]
        return {
            "event_logits": self.base.event_head(event_context),
            "log_delta": F.softplus(self.base.time_head(time_context).squeeze(-1)),
            "position": torch.sigmoid(self.base.position_head(position_context)),
            "side_logits": self.base.side_head(side_context),
            "advantage_logits": self.base.advantage_head(advantage_context),
            "player_scores": player_scores,
            "base_event_logits": self.base.event_head(base_context),
            "base_log_delta": F.softplus(
                self.base.time_head(base_context).squeeze(-1)
            ),
            "base_position": torch.sigmoid(self.base.position_head(base_context)),
            "base_side_logits": self.base.side_head(base_context),
            "base_advantage_logits": self.base.advantage_head(base_context),
            "base_player_scores": base_player_scores,
            "base_context": base_context,
            "expert_contexts": expert_contexts,
            "expert_residuals": expert_residuals,
            "task_contexts": task_contexts,
            "routing_weights": routing,
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


def compute_residual_moe_loss(
    predictions: dict[str, torch.Tensor],
    graph_batch: Any,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the six-task objective to the dense residual prediction."""

    target = slice(0, batch_size)
    event_loss = F.cross_entropy(
        predictions["event_logits"], graph_batch.y_event_type[target]
    )
    time_loss = F.huber_loss(
        predictions["log_delta"], graph_batch.y_log_delta[target]
    )
    position_values = F.huber_loss(
        predictions["position"],
        graph_batch.y_position[target],
        reduction="none",
    ).mean(dim=-1)
    position_mask = graph_batch.y_position_mask[target].bool()
    position_loss = (
        position_values[position_mask].mean()
        if bool(position_mask.any())
        else position_values.sum() * 0.0
    )
    side_loss = F.cross_entropy(
        predictions["side_logits"], graph_batch.y_side[target]
    )
    player_loss = player_cross_entropy(
        predictions["player_scores"],
        graph_batch["player"].ptr[: batch_size + 1],
        graph_batch.y_player_local[target],
        graph_batch.y_player_mask[target].bool(),
    )
    advantage_values = F.cross_entropy(
        predictions["advantage_logits"],
        graph_batch.y_advantage[target].clamp_min(0),
        reduction="none",
    )
    advantage_mask = graph_batch.y_advantage_mask[target].bool()
    advantage_loss = (
        advantage_values[advantage_mask].mean()
        if bool(advantage_mask.any())
        else advantage_values.sum() * 0.0
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
