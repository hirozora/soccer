from __future__ import annotations

import numpy as np
import torch

from football_benchmark.protocol import ProtocolArtifacts
from football_hgt_targets_v4.constants import FEASIBILITY_ARTIFACT
from football_hgt_targets_v4.model import TargetStudyHGT
from football_hgt_targets_v4.reporting import _paired_hierarchical_bootstrap
from football_hgt_targets_v4.selection import select_confirmed_methods
from football_hgt_targets_v4.training import backbone_state_hash


def test_selection_applies_event_accuracy_guard() -> None:
    means = {
        "event": {
            "inverse_ce": {"accuracy": 0.62, "macro_f1": 0.51},
            "ce": {"accuracy": 0.75, "macro_f1": 0.59},
            "sqrt_capped_ce": {"accuracy": 0.60, "macro_f1": 0.61},
        },
        "time": {
            "current_huber": {"mae_seconds": 1.43},
            "log1p_huber": {"mae_seconds": 1.47},
        },
        "position": {
            "xy": {"distance_mae_m": 16.2},
            "zone_residual": {"distance_mae_m": 16.4},
        },
    }
    winners, floor = select_confirmed_methods(means)
    assert floor == 0.61
    assert winners == {
        "event": "ce",
        "time": "current_huber",
        "position": "xy",
    }


def test_joint_variants_share_identical_backbone_initialization() -> None:
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    torch.manual_seed(20260715)
    original = TargetStudyHGT(
        artifacts,
        "joint",
        "joint_original",
        {"event": "inverse_ce", "time": "current_huber", "position": "xy"},
    )
    torch.manual_seed(20260715)
    optimized = TargetStudyHGT(
        artifacts,
        "joint",
        "joint_optimized",
        {"event": "ce", "time": "log1p_huber", "position": "zone_residual"},
    )
    assert backbone_state_hash(original) == backbone_state_hash(optimized)


def test_hierarchical_bootstrap_preserves_paired_improvement_direction() -> None:
    # Three seeds, two matches, and 104 sufficient-statistic columns.
    original = np.zeros((3, 2, 104), dtype=np.float64)
    optimized = np.zeros_like(original)
    for seed in range(3):
        for match in range(2):
            # Original predicts 8/10 correctly; optimized predicts all 10 correctly.
            original[seed, match, 0] = 8
            original[seed, match, 1] = 2
            optimized[seed, match, 0] = 10
            original[seed, match, 100:104] = [20, 10, 30, 10]
            optimized[seed, match, 100:104] = [10, 10, 20, 10]
    result = _paired_hierarchical_bootstrap(
        original, optimized, iterations=100, seed=5
    )
    assert result["event_accuracy"]["difference_optimized_minus_original"] > 0
    assert result["time_mae_seconds"]["difference_optimized_minus_original"] < 0
    assert result["position_distance_mae_m"]["difference_optimized_minus_original"] < 0

