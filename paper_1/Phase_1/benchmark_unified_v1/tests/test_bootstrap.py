from __future__ import annotations

import pandas as pd

from football_benchmark.bootstrap import hierarchical_paired_bootstrap


def test_paired_bootstrap_is_zero_for_identical_predictions() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["1:0", "1:1", "2:0", "2:1"],
            "match_id": [1, 1, 2, 2],
            "event_true": [0, 1, 2, 3],
            "event_pred": [0, 1, 2, 3],
            "event_mask": [True] * 4,
            "position_true_x": [0.1, 0.2, 0.3, 0.4],
            "position_true_y": [0.5, 0.6, 0.7, 0.8],
            "position_pred_x": [0.1, 0.2, 0.3, 0.4],
            "position_pred_y": [0.5, 0.6, 0.7, 0.8],
            "position_mask": [True] * 4,
            "zone_true": [0, 1, 2, 3],
            "zone_pred": [0, 1, 2, 3],
        }
    )
    result = hierarchical_paired_bootstrap(
        [frame], [frame.copy()], "seq2event", replicates=20, seed=1
    )
    assert result
    for values in result.values():
        assert values["delta_hgt_minus_baseline"] == 0
        assert values["ci95_low"] == 0
        assert values["ci95_high"] == 0

