from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_eca_anchor_attention import age_bucket, pair_mass, AttentionCapture
from test_partial_l2 import artifacts, batch
from football_hgt_targets_v4.model import build_partial_l2_model
from football_hgt_targets_v4.spatiotemporal_edge import edge_bundle


def test_age_boundaries():
    assert torch.equal(age_bucket(torch.tensor([0, 1, 5, 6, 20, 21, 79])),
                       torch.tensor([0, 1, 1, 2, 2, 3, 3]))
    with pytest.raises(ValueError):
        age_bucket(torch.tensor([-1]))


def test_next_gap_sum_keeps_full_neighbor_denominator():
    next_edge = ('event', 'next', 'event')
    gap_edge = ('event', 'gap_short', 'event')
    hub_edge = ('team', 'performs', 'event')
    bundle = {'pairs': torch.tensor([[0], [1]]), 'mappings': {
        next_edge: torch.tensor([0]), gap_edge: torch.tensor([0]), hub_edge: torch.tensor([-1])}}
    attention = torch.tensor([[.2]*4, [.3]*4, [.5]*4])
    result = pair_mass(bundle, [next_edge, gap_edge, hub_edge], attention)
    assert (result == .5).all()


def test_observation_is_nonmutating(artifacts, batch):
    model = build_partial_l2_model(artifacts).eval()
    with torch.no_grad():
        expected = model(batch)
        capture = AttentionCapture(model.convolutions[1])
        actual = model(batch)
        capture.close()
    assert all(torch.equal(actual[k], expected[k]) for k in expected)
    assert capture.calls == 1
    bundle = edge_bundle(batch['graphs']['f80'])
    mass = pair_mass(bundle, batch['graphs']['f80'].edge_index_dict, capture.attention)
    assert mass.shape == (bundle['pairs'].shape[1], 4)
