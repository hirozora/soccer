"""Causal Team/Player targets and fair match-local candidate collation."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from football_benchmark.data import CanonicalSample
from football_benchmark.protocol import ProtocolArtifacts

from .possession_data import collate_possession_hgt
from .subgraph_views import DEFAULT_SELECTOR_SEED


def actor_targets(samples: Sequence[CanonicalSample]) -> dict[str, torch.Tensor]:
    """Build next-actor labels without depending on the selected context view."""

    team_targets: list[int] = []
    player_targets: list[int] = []
    player_masks: list[bool] = []
    player_raw_ids: list[int] = []
    for sample in samples:
        anchor = sample.current_event_index
        event = sample.graph["node_stores"]["event"]
        players = sample.graph["node_stores"]["player"]
        target_team = int(sample.graph["targets"]["team_local_index"][anchor])
        anchor_team = int(event["team_local_index"][anchor])
        target_player = int(event["player_local_index"][anchor + 1])
        known = bool(sample.graph["targets"]["player_known_mask"][anchor])
        team_targets.append(int(target_team == anchor_team))
        player_targets.append(target_player)
        player_masks.append(known)
        player_raw_ids.append(int(players["raw_id"][target_player]))
    return {
        "team_actor": torch.tensor(team_targets, dtype=torch.long),
        "player_local": torch.tensor(player_targets, dtype=torch.long),
        "player_mask": torch.tensor(player_masks, dtype=torch.bool),
        "player_raw": torch.tensor(player_raw_ids, dtype=torch.long),
    }


def collate_actor_hgt(
    samples: Sequence[CanonicalSample],
    artifacts: ProtocolArtifacts,
    window_size: int,
    *,
    context_view: str,
    selector_seed: int = DEFAULT_SELECTOR_SEED,
) -> dict[str, Any]:
    """Collate one view while retaining an identical Player roster universe."""

    batch = collate_possession_hgt(
        samples,
        artifacts,
        window_size,
        topology="membership",
        feature_level="dynamic",
        snapshot_scope="selected_events",
        context_view=context_view,
        selector_seed=selector_seed,
        retain_full_player_roster=True,
    )
    batch["targets"].update(actor_targets(samples))
    return batch

