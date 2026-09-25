"""Adapters for baseline architectures using the common Wyscout graph split."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from itertools import accumulate
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from football_hgt.dataset import MatchGraphRecord, load_match_graph


class SequenceBaselineDataset(Dataset[dict[str, torch.Tensor]]):
    """Return left-padded event sequences without constructing graph windows."""

    def __init__(
        self,
        records: Sequence[MatchGraphRecord],
        sequence_length: int,
        num_event_types: int,
    ) -> None:
        if not records or sequence_length < 1:
            raise ValueError("records and a positive sequence length are required")
        self.records = tuple(records)
        self.sequence_length = sequence_length
        self.num_event_types = num_event_types
        self._sample_ends = tuple(
            accumulate(record.num_events - 1 for record in self.records)
        )
        self._cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return self._sample_ends[-1]

    def _load(self, record: MatchGraphRecord) -> dict[str, Any]:
        if record.graph_path in self._cache:
            graph = self._cache.pop(record.graph_path)
            self._cache[record.graph_path] = graph
            return graph
        graph = load_match_graph(record.graph_path, validate=False)
        self._cache[record.graph_path] = graph
        while len(self._cache) > len(self.records):
            self._cache.popitem(last=False)
        return graph

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = bisect_right(self._sample_ends, index)
        previous_end = self._sample_ends[record_index - 1] if record_index else 0
        current = index - previous_end
        graph = self._load(self.records[record_index])
        event = graph["node_stores"]["event"]
        targets = graph["targets"]

        start = max(0, current - self.sequence_length + 1)
        stop = current + 1
        length = stop - start
        padding = self.sequence_length - length
        event_types = torch.full(
            (self.sequence_length,), self.num_event_types, dtype=torch.long
        )
        event_types[padding:] = event["event_type_index"][start:stop]

        continuous = torch.zeros((self.sequence_length, 9), dtype=torch.float32)
        selected = torch.cat(
            (
                event["start_position"][start:stop],
                event["end_position"][start:stop],
                torch.log1p(
                    event["delta_from_previous"][start:stop].clamp_min(0.0)
                ).unsqueeze(-1),
                (event["period_seconds"][start:stop] / 3600.0).unsqueeze(-1),
                (event["absolute_seconds"][start:stop] / 7200.0).unsqueeze(-1),
                event["team_local_index"][start:stop].float().unsqueeze(-1),
                (event["result_direction"][start:stop].float() / 2.0).unsqueeze(-1),
            ),
            dim=-1,
        )
        continuous[padding:] = selected
        valid_mask = torch.zeros(self.sequence_length, dtype=torch.bool)
        valid_mask[padding:] = True
        return {
            "event_type": event_types,
            "continuous": continuous,
            "valid_mask": valid_mask,
            "target_event_type": targets["event_type_index"][current].clone(),
            "target_log_delta": torch.log1p(
                targets["delta_seconds"][current]
            ).clone(),
            "target_delta_seconds": targets["delta_seconds"][current].clone(),
            "target_position": targets["start_position"][current].clone(),
            "target_position_mask": targets["start_position_mask"][current].clone(),
        }


def sinusoidal_encoding(length: int, channels: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    dimensions = torch.arange(channels, dtype=torch.float32).unsqueeze(0)
    denominator = torch.pow(100.0, 2.0 * dimensions / channels)
    values = positions / denominator
    return torch.where((dimensions.long() % 2) == 0, values.sin(), values.cos())


class SoccerSeq2EventAdapter(nn.Module):
    """Official one-layer Transformer shape with the common 10-class target."""

    supports_continuous_targets = True

    def __init__(self, num_event_types: int, sequence_length: int = 40) -> None:
        super().__init__()
        self.event_embedding = nn.Embedding(
            num_event_types + 1, 7, padding_idx=num_event_types
        )
        self.continuous_projection = nn.Linear(9, 10)
        layer = nn.TransformerEncoderLayer(
            d_model=17,
            nhead=1,
            dim_feedforward=8,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.hidden = nn.Linear(17, 64)
        self.event_head = nn.Linear(64, num_event_types)
        self.time_head = nn.Linear(64, 1)
        self.position_head = nn.Linear(64, 2)
        self.register_buffer(
            "position_encoding", sinusoidal_encoding(sequence_length, 17)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values = torch.cat(
            (
                self.event_embedding(batch["event_type"]),
                self.continuous_projection(batch["continuous"]),
            ),
            dim=-1,
        )
        values = values + self.position_encoding.unsqueeze(0)
        encoded = self.transformer(
            values, src_key_padding_mask=~batch["valid_mask"]
        )
        hidden = F.relu(self.hidden(encoded[:, -1]))
        return {
            "event_logits": self.event_head(hidden),
            "log_delta": F.softplus(self.time_head(hidden).squeeze(-1)),
            "position": torch.sigmoid(self.position_head(hidden)),
        }


class UnifiedLEMAdapter(nn.Module):
    """Unified LEM tabular MLP using its largest published context length."""

    supports_continuous_targets = False

    def __init__(
        self, num_event_types: int, sequence_length: int = 9, hidden_size: int = 196
    ) -> None:
        super().__init__()
        input_size = sequence_length * 11
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_event_types),
        )
        self.num_event_types = num_event_types

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        normalized_event = batch["event_type"].float() / self.num_event_types
        values = torch.cat(
            (
                normalized_event.unsqueeze(-1),
                batch["continuous"],
                batch["valid_mask"].float().unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"event_logits": self.layers(values.flatten(start_dim=1))}


class OGLEMExtendedAdapter(nn.Module):
    """Original LEM multi-layer event classifier with sigmoid outputs."""

    supports_continuous_targets = False

    def __init__(self, num_event_types: int, sequence_length: int = 1) -> None:
        super().__init__()
        input_size = sequence_length * 11
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_event_types),
            nn.Sigmoid(),
        )
        self.num_event_types = num_event_types

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        normalized_event = batch["event_type"].float() / self.num_event_types
        values = torch.cat(
            (
                normalized_event.unsqueeze(-1),
                batch["continuous"],
                batch["valid_mask"].float().unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"event_probabilities": self.layers(values.flatten(start_dim=1))}


def baseline_loss(
    model_name: str,
    model: nn.Module,
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    if model_name == "og_lem_extended":
        target = F.one_hot(
            batch["target_event_type"], num_classes=predictions["event_probabilities"].shape[1]
        ).float()
        return F.binary_cross_entropy(predictions["event_probabilities"], target)

    event_loss = F.cross_entropy(
        predictions["event_logits"], batch["target_event_type"]
    )
    if not model.supports_continuous_targets:
        return event_loss
    time_loss = F.huber_loss(
        predictions["log_delta"], batch["target_log_delta"]
    )
    position_values = F.huber_loss(
        predictions["position"], batch["target_position"], reduction="none"
    ).mean(dim=-1)
    mask = batch["target_position_mask"].bool()
    position_loss = position_values[mask].mean() if bool(mask.any()) else position_values.sum() * 0.0
    return event_loss + 0.3 * time_loss + 0.5 * position_loss
