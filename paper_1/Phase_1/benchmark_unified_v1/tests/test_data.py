from __future__ import annotations

import torch

from football_benchmark.data import (
    CanonicalEventDataset,
    collate_sequence,
    load_records,
)


def test_canonical_sample_is_immediate_and_causal(artifacts) -> None:
    dataset = CanonicalEventDataset(load_records("train")[:1], artifacts, window_size=80)
    sample = dataset[100]
    assert sample.target_event_index == sample.current_event_index + 1
    assert sample.stop == sample.current_event_index + 1
    assert sample.stop - sample.start <= 80
    event = sample.graph["node_stores"]["event"]
    assert sample.target.raw_event_10.item() == event["event_type_index"][101].item()
    assert sample.sample_id == f"{sample.match_id}:100"


def test_short_histories_are_left_padded_without_future(artifacts) -> None:
    dataset = CanonicalEventDataset(load_records("train")[:1], artifacts, window_size=80)
    samples = [dataset[0], dataset[1], dataset[79]]
    batch = collate_sequence(samples, artifacts, padded_width=80)
    assert batch["valid_mask"].sum(dim=1).tolist() == [1, 2, 80]
    assert batch["sequence"]["event"].shape == (3, 80)
    assert batch["sequence"]["numeric"].shape[-1] == 12
    assert batch["targets"]["raw_event_10"].dtype == torch.long
    assert batch["targets"]["action_4_mask"].dtype == torch.bool


def test_period_boundary_masks_only_time(artifacts) -> None:
    dataset = CanonicalEventDataset(load_records("train")[:1], artifacts, window_size=80)
    graph = dataset[0].graph
    periods = graph["node_stores"]["event"]["period_index"]
    boundary = int(torch.nonzero(periods[1:] != periods[:-1])[0])
    sample = dataset[boundary]
    assert not bool(sample.target.time_mask)
    assert bool(sample.target.position_mask)

