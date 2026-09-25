from __future__ import annotations

from functools import partial

import pytest
import torch

from football_benchmark.data import (
    CanonicalEventDataset,
    collate_hgt,
    collate_sequence,
    load_records,
)
from football_benchmark.losses import compute_benchmark_loss
from football_benchmark.models import ModelSpec, build_model


@pytest.mark.parametrize(
    ("contract", "family", "window"),
    [
        ("seq2event", "seq2event", 40),
        ("unified_lem", "unified_lem", 3),
        ("nmstpp", "nmstpp", 40),
    ],
)
def test_sequence_models_forward_backward(artifacts, contract, family, window) -> None:
    dataset = CanonicalEventDataset(load_records("train")[:1], artifacts, window_size=window)
    samples = [dataset[index] for index in (50, 51, 52, 53)]
    batch = collate_sequence(samples, artifacts, padded_width=window)
    model = build_model(ModelSpec(family, contract, window), artifacts)
    predictions = model(batch)
    loss, components = compute_benchmark_loss(
        predictions, batch, family, contract, artifacts
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert components
    assert any(parameter.grad is not None for parameter in model.parameters())
    if family == "unified_lem":
        inactive = ~artifacts.fine_active_mask
        assert torch.all(
            predictions["fine_event_logits"][:, inactive]
            == torch.finfo(predictions["fine_event_logits"].dtype).min
        )


def test_hgt_forward_backward_and_single_graph_batch(artifacts) -> None:
    pytest.importorskip("torch_geometric")
    dataset = CanonicalEventDataset(load_records("train")[:1], artifacts, window_size=8)
    samples = [dataset[index] for index in (20, 21)]
    batch = collate_hgt(samples, artifacts)
    model = build_model(ModelSpec("hgt", "unified_lem", 8), artifacts)
    predictions = model(batch)
    loss, _ = compute_benchmark_loss(
        predictions, batch, "hgt", "unified_lem", artifacts
    )
    loss.backward()
    assert predictions["event_logits"].shape == (2, 10)
    assert predictions["position_xy"].shape == (2, 2)
    assert predictions["time_seconds"].shape == (2,)
