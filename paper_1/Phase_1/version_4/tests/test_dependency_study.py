from __future__ import annotations

import numpy as np
import pytest
import torch

from football_hgt_targets_v4.dependency_reporting import _bootstrap_improvement
from football_hgt_targets_v4.dependency_study import (
    POSITION_CONFIGS,
    TIME_CONFIGS,
    ResidualProbe,
    _permutation,
    build_conditions,
)
from football_hgt_targets_v4.position_attribution import _group_summary


def _cache(length: int = 12) -> dict:
    return {
        "sample_ids": [f"1:{index}" for index in range(length)],
        "seed": 20260715,
        "split": "train",
        "event_true": torch.arange(length) % 10,
        "event_probabilities": torch.softmax(torch.arange(length * 10).reshape(length, 10).float(), -1),
        "time_true": torch.linspace(0, 60, length),
        "time_mask": torch.tensor([True] * (length - 1) + [False]),
        "base_time_seconds": torch.linspace(1, 30, length),
    }


@pytest.mark.parametrize("name", POSITION_CONFIGS)
def test_position_conditions_are_fixed_width_and_finite(name: str) -> None:
    values = build_conditions(_cache(), "position", name)
    assert values.shape == (12, 16)
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("name", TIME_CONFIGS)
def test_time_conditions_are_fixed_width_and_finite(name: str) -> None:
    values = build_conditions(_cache(), "time", name)
    assert values.shape == (12, 10)
    assert torch.isfinite(values).all()


def test_shuffling_is_deterministic_and_preserves_marginal() -> None:
    cache = _cache()
    first = build_conditions(cache, "position", "p7_event_shuffled")[:, :10]
    second = build_conditions(cache, "position", "p7_event_shuffled")[:, :10]
    oracle = build_conditions(cache, "position", "p1_event_oracle")[:, :10]
    assert torch.equal(first, second)
    assert not torch.equal(first, oracle)
    assert torch.equal(first.sum(0), oracle.sum(0))
    assert torch.equal(
        _permutation(12, 20260715, "train", "x"),
        _permutation(12, 20260715, "train", "x"),
    )


@pytest.mark.parametrize("family", ["position", "time"])
def test_zero_initialized_probe_exactly_preserves_base(family: str) -> None:
    torch.manual_seed(4)
    model = ResidualProbe(family).eval()
    context = torch.randn(8, 64)
    condition = torch.randn(8, 16 if family == "position" else 10)
    base = torch.rand(8, 2) if family == "position" else torch.rand(8) * 60
    observed = model(context, condition, base)
    assert torch.allclose(observed, base, atol=1e-6, rtol=0)
    assert torch.count_nonzero(model.output.weight) == 0
    assert torch.count_nonzero(model.output.bias) == 0


def test_paired_bootstrap_reports_improvement_direction() -> None:
    baseline = np.array([[[30.0, 10.0], [20.0, 10.0]]] * 3)
    candidate = np.array([[[20.0, 10.0], [10.0, 10.0]]] * 3)
    result = _bootstrap_improvement(baseline, candidate, iterations=100, seed=3)
    assert result["improvement"] == pytest.approx(1.0)
    assert result["ci95"][0] > 0


def test_attribution_group_contributions_reproduce_both_gaps() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "match_id": [1, 1, 2, 2],
            "group": ["a", "a", "b", "b"],
            "hgt_error_m": [2.0, 4.0, 8.0, 10.0],
            "nmstpp_error_m": [1.0, 3.0, 5.0, 7.0],
            "gap_m": [1.0, 1.0, 3.0, 3.0],
            "hgt_equal_error_m": [3.0, 4.0, 5.0, 6.0],
            "nmstpp_equal_error_m": [2.0, 2.0, 4.0, 4.0],
            "equal_gap_m": [1.0, 2.0, 1.0, 2.0],
        }
    )
    summary = _group_summary(frame, "group", ["a", "b"])
    assert summary.gap_contribution_m.sum() == pytest.approx(frame.gap_m.mean())
    assert summary.equal_gap_contribution_m.sum() == pytest.approx(
        frame.equal_gap_m.mean()
    )
