"""Version 3 dataset and batching for full-history residual graph views."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from itertools import accumulate
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from football_hgt.dataset import (
    MatchGraphRecord,
    load_match_graph,
    sample_fixed_event_window,
)
from football_hgt_v1.data import window_to_heterodata

from .experts import (
    BRANCH_NAMES,
    EXPERT_NAMES,
    GraphViewConfig,
    StructuralExpertGenerator,
    build_structural_expert_view,
    pairwise_jaccard,
)


OVERLAP_KEYS = tuple(
    f"{left}__{right}"
    for left_index, left in enumerate(BRANCH_NAMES)
    for right in BRANCH_NAMES[left_index + 1 :]
)


class FullHistoryResidualDataset(Dataset[dict[str, Any]]):
    """Generate one K=80 base branch and five residual expert views per step."""

    def __init__(
        self,
        records: Sequence[MatchGraphRecord],
        view_config: GraphViewConfig | None = None,
        cache_size: int | None = None,
    ) -> None:
        if not records:
            raise ValueError("At least one match record is required")
        self.records = tuple(records)
        self.view_config = view_config or GraphViewConfig()
        self.generator = StructuralExpertGenerator(self.view_config)
        self.cache_size = cache_size or min(8, len(records))
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
        if int(graph["node_stores"]["event"]["num_nodes"]) != record.num_events:
            raise ValueError(f"Event count mismatch for {record.graph_path}")
        self._cache[record.graph_path] = graph
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return graph

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = bisect_right(self._sample_ends, index)
        previous_end = self._sample_ends[record_index - 1] if record_index else 0
        current = index - previous_end
        graph = self._load(self.records[record_index])
        full_history = sample_fixed_event_window(
            graph,
            current_event_index=current,
            window_size=self.view_config.full_history,
        )
        selected = self.generator.generate(graph, current)
        experts = {
            name: build_structural_expert_view(
                graph,
                selected[name],
                current,
                name,
                full_history["targets"],
            )
            for name in EXPERT_NAMES
        }
        full_start = int(full_history["window"]["start_event_index"])
        all_selected = {
            "full_history": list(range(full_start, current + 1)),
            **selected,
        }
        return {
            "match_id": graph["match_id"],
            "current_event_index": current,
            "full_history": full_history,
            "experts": experts,
            "overlap": pairwise_jaccard(all_selected),
            "view_sizes": {
                name: len(indices) for name, indices in all_selected.items()
            },
        }


def collate_full_history_residuals(
    samples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Batch the full branch first, followed by five expert-major blocks."""

    data_list = [
        window_to_heterodata(
            sample["full_history"]
            if branch_name == "full_history"
            else sample["experts"][branch_name]
        )
        for branch_name in BRANCH_NAMES
        for sample in samples
    ]
    overlap = torch.tensor(
        [
            [sample["overlap"][key] for key in OVERLAP_KEYS]
            for sample in samples
        ],
        dtype=torch.float32,
    )
    view_sizes = torch.tensor(
        [
            [sample["view_sizes"][name] for name in BRANCH_NAMES]
            for sample in samples
        ],
        dtype=torch.long,
    )
    return {
        "graph": Batch.from_data_list(data_list),
        "batch_size": len(samples),
        "overlap": overlap,
        "view_sizes": view_sizes,
    }


def _target_signature(targets: dict[str, torch.Tensor]) -> tuple[Any, ...]:
    return (
        int(targets["event_type_index"]),
        float(targets["delta_seconds"]),
        tuple(float(value) for value in targets["start_position"]),
        int(targets["acting_side_index"]),
        int(targets["player_vocab_index"]),
        int(targets["advantage_index"]),
    )


def validate_residual_sample(sample: dict[str, Any]) -> list[str]:
    """Check the base branch, expert causality, edge bounds, and shared target."""

    errors = []
    current = int(sample["current_event_index"])
    full_history = sample.get("full_history")
    if full_history is None:
        return ["missing full_history branch"]
    if full_history["window"]["current_event_index"] != current:
        errors.append("full_history does not end at the shared anchor")
    if full_history["window"]["target_event_index"] != current + 1:
        errors.append("full_history target index is not current+1")
    reference_target = _target_signature(full_history["targets"])

    for name in EXPERT_NAMES:
        view = sample["experts"].get(name)
        if view is None:
            errors.append(f"missing expert {name}")
            continue
        selected = view["selected_event_indices"]
        if not selected or selected[-1] != current:
            errors.append(f"{name} does not end at the shared anchor")
        if any(index > current for index in selected):
            errors.append(f"{name} contains a future event")
        if view["window"]["target_event_index"] != current + 1:
            errors.append(f"{name} target index is not current+1")
        num_events = view["node_stores"]["event"]["num_nodes"]
        for edge_name, edge_store in view["edge_stores"].items():
            edge_index = edge_store["edge_index"]
            if edge_name.endswith("__event") and edge_index.numel():
                if int(edge_index[1].max()) >= num_events:
                    errors.append(f"{name}.{edge_name} has an invalid event destination")
            if edge_name.startswith("event__") and edge_index.numel():
                if int(edge_index[0].max()) >= num_events:
                    errors.append(f"{name}.{edge_name} has an invalid event source")
        if _target_signature(view["targets"]) != reference_target:
            errors.append(f"{name} target differs from the full-history target")
    return errors
