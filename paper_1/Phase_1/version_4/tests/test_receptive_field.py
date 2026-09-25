from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from football_benchmark.data import CanonicalEventDataset, load_records
from football_benchmark.protocol import ProtocolArtifacts
from football_benchmark.sampling import TargetSamplePlan
from football_hgt_targets_v4.constants import FEASIBILITY_ARTIFACT, POSSESSION_GRAPH_ROOT, SAMPLE_PLAN, WINDOW_SIZE
from football_hgt_targets_v4.fixed_budget_study import CONFIGURATIONS
from football_hgt_targets_v4.fixed_budget_training import _common_hash
from football_hgt_targets_v4.model import build_partial_l2_model, build_receptive_field_model
from football_hgt_targets_v4.possession_data import collate_multiview_possession_hgt
from football_hgt_targets_v4.receptive_field import collate_receptive_field_hgt
from football_hgt_targets_v4.training import _move_batch_to_device


@pytest.fixture(scope="module")
def artifacts():
    return ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)


@pytest.fixture(scope="module")
def samples(artifacts):
    selected = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("validation")
    dataset = CanonicalEventDataset(
        load_records("validation", graph_root=POSSESSION_GRAPH_ROOT), artifacts,
        WINDOW_SIZE, max_samples=64, selected_currents=selected,
    )
    return [value for value in dataset if value.current_event_index >= 20][:1]


def make_batch(samples, artifacts, configuration):
    return next(iter(DataLoader(samples, batch_size=1, collate_fn=partial(
        collate_receptive_field_hgt, artifacts=artifacts, window_size=WINDOW_SIZE,
        configuration=configuration,
    ))))


def cuda_batch(samples, artifacts, configuration):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for HGT receptive-field forward tests")
    return _move_batch_to_device(
        make_batch(samples, artifacts, configuration), torch.device("cuda:0")
    )


def test_rf_configurations_are_guarded_partial_l2():
    for name in ("rf_core_u5", "rf_core_u10", "rf_task"):
        assert CONFIGURATIONS[name]["partial_l2"] is True
        assert CONFIGURATIONS[name]["checkpoint_metric"] == "guarded_core"
        assert CONFIGURATIONS[name]["receptive_field"] is True


def test_one_f80_graph_and_strict_counts(samples, artifacts):
    batch = make_batch(samples, artifacts, "rf_task")
    assert tuple(batch["graphs"]) == ("f80",)
    for count in (5, 10):
        metadata = batch["rf_metadata"][count]
        assert metadata["event_counts"].tolist() == [count]
        assert int(metadata["event_pool_mask"].sum()) == count
        for edge_type, mask in metadata["edge_masks"].items():
            edge = batch["graphs"]["f80"][edge_type].edge_index
            assert torch.all(metadata["node_masks"][edge_type[0]][edge[0, mask]])
            assert torch.all(metadata["node_masks"][edge_type[2]][edge[1, mask]])


def test_rf_has_no_extra_parameters_and_expected_calls(samples, artifacts):
    torch.manual_seed(17)
    baseline = build_partial_l2_model(artifacts, dropout=0.0)
    baseline_hash = _common_hash(baseline)
    count = sum(value.numel() for value in baseline.parameters())
    del baseline
    torch.manual_seed(17)
    model = build_receptive_field_model(artifacts, "rf_task", dropout=0.0).cuda().eval()
    assert sum(value.numel() for value in model.parameters()) == count
    assert _common_hash(model) == baseline_hash
    with torch.no_grad():
        model(cuda_batch(samples, artifacts, "rf_task"))
    assert model._rf_forward_counts == {10: 1, 5: 1}


def test_n80_is_exact_partial_l2_regression(samples, artifacts):
    batch = cuda_batch(samples, artifacts, "rf_f80_equivalence")
    torch.manual_seed(23)
    baseline = build_partial_l2_model(artifacts, dropout=0.0).cuda().eval()
    with torch.no_grad():
        expected = baseline(batch)
    del baseline
    torch.manual_seed(23)
    candidate = build_receptive_field_model(artifacts, "rf_f80_equivalence", dropout=0.0).cuda().eval()
    with torch.no_grad():
        actual = candidate(batch)
    assert max(float((expected[name] - actual[name]).abs().max()) for name in expected) < 1e-6


def test_edge_filtering_precedes_hgt_attention(samples, artifacts):
    batch = cuda_batch(samples, artifacts, "rf_task")
    model = build_receptive_field_model(artifacts, "rf_task", dropout=0.0).cuda().eval()
    observed = []
    hook = model.convolutions[0].register_forward_pre_hook(
        lambda _module, args: observed.append({key: value.shape[1] for key, value in args[1].items()})
    )
    with torch.no_grad():
        model(batch)
    hook.remove()
    # First call is F80; later calls must receive already-filtered edge_index tensors.
    assert len(observed) == 3
    assert any(observed[1][key] < observed[0][key] for key in observed[0])
    assert any(observed[2][key] < observed[1][key] for key in observed[0])


@pytest.mark.parametrize("count", [5, 10])
def test_masked_f80_matches_physical_short_graph(samples, artifacts, count):
    rf_batch = cuda_batch(samples, artifacts, "rf_task")
    physical = collate_multiview_possession_hgt(
        samples, artifacts, WINDOW_SIZE, topology="membership",
        feature_level="dynamic", snapshot_scope="selected_events",
        context_views=(f"s{count}",), retain_full_player_roster=True,
    )
    physical = _move_batch_to_device(physical, torch.device("cuda:0"))
    model = build_receptive_field_model(artifacts, "rf_task", dropout=0.0).cuda().eval()
    with torch.no_grad():
        _, contexts = model.forward_with_contexts(rf_batch)
        graph = physical["graphs"][f"s{count}"]
        states = model._encode_nodes(graph)
        edges = dict(graph.edge_index_dict)
        states = model._apply_layer(states, edges, model.convolutions[0], model.norms[0], model.dropout)
        states = model._apply_layer(states, edges, model.convolutions[1], model.norms[1], model.dropout)
        expected = model._pool_context(states, graph, model.context_projection)
    assert float((contexts[f"n{count}"] - expected).abs().max()) < 1e-6


def test_out_of_range_event_features_do_not_change_core_outputs(samples, artifacts):
    batch = cuda_batch(samples, artifacts, "rf_task")
    model = build_receptive_field_model(artifacts, "rf_task", dropout=0.0).cuda().eval()
    with torch.no_grad():
        expected = model(batch)
        graph = batch["graphs"]["f80"]
        old = ~batch["rf_metadata"][10]["event_pool_mask"]
        graph["event"].start_position[old] = 1.0 - graph["event"].start_position[old]
        graph["event"].end_position[old] = 1.0 - graph["event"].end_position[old]
        graph["event"].relative_features[old] += 3.0
        actual = model(batch)
    for key in ("event_logits", "time_seconds", "position_xy"):
        assert float((expected[key] - actual[key]).abs().max()) < 1e-6
