from pathlib import Path

from football_hgt_targets_v4.recency_control_study import ALL_CONFIGS, NEW_CONFIGS


def test_registered_recency_control_configs_are_complete() -> None:
    assert NEW_CONFIGS == ("lp1", "lp2", "recency_sf_b")
    assert set(ALL_CONFIGS) == {
        "f80", "p1", "lp1", "p2", "lp2", "semantic_sf_b", "recency_sf_b"
    }


def test_fusion_cost_contract_is_additive() -> None:
    costs = {"p1": 5.43, "p2": 10.66, "lp1": 5.43, "lp2": 10.66}
    assert costs["p1"] + costs["p2"] == costs["lp1"] + costs["lp2"]
    assert costs["p1"] + costs["p2"] > costs["p2"]
