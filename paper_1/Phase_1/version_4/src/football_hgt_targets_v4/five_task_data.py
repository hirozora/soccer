"""Aligned Semantic V3 views and five immediate-next-event targets."""

from __future__ import annotations

from typing import Any, Sequence

from football_benchmark.data import CanonicalSample
from football_benchmark.protocol import ProtocolArtifacts

from .actor_data import actor_targets
from .possession_data import collate_multiview_possession_hgt
from .subgraph_views import DEFAULT_SELECTOR_SEED


def anchor_global_state(graph: Any) -> Any:
    """Build the causal 8D sample state used by relation gating."""

    import torch

    event = graph["event"]
    anchor_indices = event.ptr[1:] - 1
    batch_size = int(anchor_indices.numel())
    state = torch.zeros((batch_size, 8), dtype=torch.float32)
    control = event.control_state_after_index[anchor_indices]
    role = event.event_role_index[anchor_indices]
    state[:, 0] = (control == 2).float()
    state[:, 1] = (control == 3).float()
    state[:, 2] = (role == 2).float()
    state[:, 3] = (role == 4).float()
    state[:, 4] = event.switch_confirmed[anchor_indices].float()

    membership = graph[("event", "belongs_to", "possession")].edge_index
    if membership.numel():
        anchor_to_sample = torch.full(
            (event.num_nodes,), -1, dtype=torch.long
        )
        anchor_to_sample[anchor_indices] = torch.arange(batch_size)
        samples = anchor_to_sample[membership[0]]
        keep = samples >= 0
        if keep.any():
            sample_indices = samples[keep]
            possession_indices = membership[1, keep]
            possession = graph["possession"]
            dynamic = possession.dynamic_feature_mask[possession_indices].bool()
            state[sample_indices, 5] = (
                torch.log1p(possession.duration_so_far_seconds[possession_indices])
                / 5.0
            ) * dynamic
            state[sample_indices, 6] = (
                possession.event_count_so_far[possession_indices] / 80.0
            ) * dynamic
            state[sample_indices, 7] = dynamic.float()
    return state


def collate_five_task_hgt(
    samples: Sequence[CanonicalSample],
    artifacts: ProtocolArtifacts,
    window_size: int,
    *,
    context_views: Sequence[str],
    selector_seed: int = DEFAULT_SELECTOR_SEED,
) -> dict[str, Any]:
    """Collate aligned views with a view-invariant match-local roster."""

    batch = collate_multiview_possession_hgt(
        samples,
        artifacts,
        window_size,
        topology="membership",
        feature_level="dynamic",
        snapshot_scope="selected_events",
        context_views=context_views,
        selector_seed=selector_seed,
        retain_full_player_roster=True,
    )
    batch["targets"].update(actor_targets(samples))
    batch["anchor_state"] = anchor_global_state(batch["graphs"]["f80"])
    return batch
