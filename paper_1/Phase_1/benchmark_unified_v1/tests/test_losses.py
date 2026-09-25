from __future__ import annotations

import math

import torch

from football_benchmark.losses import (
    _normalized_classification_loss,
    stable_sqrt_class_weights,
)


def test_stable_sqrt_weights_are_bounded_and_sample_normalized(
    feasibility_artifacts,
) -> None:
    counts = feasibility_artifacts.class_counts["fine32"]
    weights = stable_sqrt_class_weights(counts)
    active = counts > 0
    assert torch.equal(weights[~active], torch.zeros_like(weights[~active]))
    assert float(weights[active].max() / weights[active].min()) <= 5.0 + 1e-6
    sample_mean = (counts.float() * weights).sum() / counts.sum()
    assert torch.allclose(sample_mean, torch.tensor(1.0), atol=1e-6)


def test_normalized_weighted_ce_has_unit_uniform_baseline(
    feasibility_artifacts,
) -> None:
    counts = feasibility_artifacts.class_counts["fine32"]
    active = counts > 0
    active_indices = torch.nonzero(active).flatten()
    targets = active_indices.repeat_interleave(3)
    logits = torch.full((len(targets), len(counts)), torch.finfo(torch.float32).min)
    logits[:, active] = 0.0
    mask = torch.ones(len(targets), dtype=torch.bool)
    weights = stable_sqrt_class_weights(counts)
    loss = _normalized_classification_loss(logits, targets, mask, weights)
    assert math.isclose(float(loss), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_normalized_weighted_ce_respects_empty_mask(feasibility_artifacts) -> None:
    counts = feasibility_artifacts.class_counts["fine32"]
    logits = torch.zeros((2, len(counts)), requires_grad=True)
    targets = torch.zeros(2, dtype=torch.long)
    loss = _normalized_classification_loss(
        logits,
        targets,
        torch.zeros(2, dtype=torch.bool),
        stable_sqrt_class_weights(counts),
    )
    loss.backward()
    assert float(loss) == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))
