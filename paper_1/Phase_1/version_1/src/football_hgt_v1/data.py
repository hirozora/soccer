"""Convert portable fixed-window dictionaries to PyG batches."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch_geometric.data import Batch, HeteroData


NODE_FIELDS = {
    "event": (
        "event_type_index",
        "subevent_type_index",
        "period_index",
        "period_seconds",
        "absolute_seconds",
        "delta_from_previous",
        "start_position",
        "start_position_mask",
        "end_position",
        "end_position_mask",
        "result_direction",
    ),
    "player": (
        "vocab_index",
        "role_index",
        "foot_index",
        "height_weight",
        "metadata_mask",
    ),
    "team": ("vocab_index", "side_index", "type_index"),
    "tag": ("vocab_index", "category_index"),
}


def window_to_heterodata(sample: dict[str, Any]) -> HeteroData:
    """Convert one validated portable window to :class:`HeteroData`."""

    data = HeteroData()
    for node_type, fields in NODE_FIELDS.items():
        source = sample["node_stores"][node_type]
        data[node_type].num_nodes = int(source["num_nodes"])
        for field in fields:
            data[node_type][field] = source[field]

    for source, relation, destination in sample["edge_types"]:
        key = f"{source}__{relation}__{destination}"
        data[(source, relation, destination)].edge_index = sample["edge_stores"][
            key
        ]["edge_index"]

    targets = sample["targets"]
    data.y_event_type = targets["event_type_index"].reshape(1)
    data.y_log_delta = targets["log_delta_seconds"].reshape(1)
    data.y_delta_seconds = targets["delta_seconds"].reshape(1)
    data.y_position = targets["start_position"].reshape(1, 2)
    data.y_position_mask = targets["start_position_mask"].reshape(1)
    data.y_side = targets["acting_side_index"].reshape(1)
    data.y_player_local = targets["player_local_index"].reshape(1)
    data.y_player_mask = targets["player_mask"].reshape(1)
    data.y_advantage = targets["advantage_index"].reshape(1)
    data.y_advantage_mask = targets["advantage_mask"].reshape(1)
    data.match_id = torch.tensor([sample["match_id"]], dtype=torch.long)
    return data


def collate_fixed_windows(samples: Sequence[dict[str, Any]]) -> Batch:
    """Batch portable fixed windows using PyG's heterogeneous collation."""

    return Batch.from_data_list([window_to_heterodata(sample) for sample in samples])
