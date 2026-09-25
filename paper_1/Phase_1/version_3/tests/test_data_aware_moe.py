from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
PHASE_ROOT = VERSION_ROOT.parent
GRAPH_ROOT = PHASE_ROOT / "data/whyscout/processed/heterogeneous_graphs/v1"
sys.path.insert(0, str(PHASE_ROOT / "src"))
sys.path.insert(0, str(PHASE_ROOT / "version_1/src"))
sys.path.insert(0, str(VERSION_ROOT / "src"))

from football_hgt.dataset import MatchGraphRecord, load_match_graph  # noqa: E402
from football_hgt_v1.model import ModelConfig  # noqa: E402
from football_hgt_v3.data import (  # noqa: E402
    DataAwareWindowDataset,
    candidate_player_features,
    collate_data_aware_windows,
    validate_data_aware_sample,
)
from football_hgt_v3.experts import (  # noqa: E402
    EXPERT_NAMES,
    SelectionConfig,
    oriented_event_points,
)
from football_hgt_v3.model import (  # noqa: E402
    TASK_NAMES,
    DataAwareMoEConfig,
    DataAwareResidualMoE,
    compute_multitask_loss,
    player_cross_entropy,
)


@pytest.fixture(scope="module")
def dataset() -> DataAwareWindowDataset:
    path = GRAPH_ROOT / "graphs/England/2499719.pt"
    graph = load_match_graph(path, validate=False)
    record = MatchGraphRecord(
        competition_slug="England",
        match_id=2499719,
        graph_path=path,
        num_events=int(graph["node_stores"]["event"]["num_nodes"]),
    )
    return DataAwareWindowDataset([record])


@pytest.fixture(scope="module")
def model_config() -> ModelConfig:
    vocab = json.loads((GRAPH_ROOT / "metadata/vocabularies.json").read_text(encoding="utf-8"))
    return ModelConfig(
        num_event_types=len(vocab["event_type_ids"]),
        num_subevent_types=len(vocab["subevent_type_ids"]),
        num_players=len(vocab["player_ids"]),
        num_teams=len(vocab["team_ids"]),
        num_tags=len(vocab["tag_ids"]),
        hidden_channels=32,
    )


def test_selections_are_causal_normalized_and_single_graph(dataset):
    samples = [dataset[index] for index in (0, 20, 100)]
    assert [validate_data_aware_sample(sample) for sample in samples] == [[], [], []]
    payload = collate_data_aware_windows(samples)
    assert payload["graph"].num_graphs == len(samples)
    assert tuple(payload["overlap"].shape) == (3, 10)
    config = SelectionConfig()
    for name in EXPERT_NAMES:
        selection = payload["selections"][name]
        assert tuple(selection["event_indices"].shape) == (3, config.budget(name))
        assert tuple(selection["structural_features"].shape) == (3, 12)
        assert selection["event_indices"].dtype == torch.long
        assert selection["selection_mask"].dtype == torch.bool
        assert torch.isfinite(selection["structural_features"]).all()
        assert torch.all((selection["structural_features"] >= 0) & (selection["structural_features"] <= 1))

    next_edges = payload["graph"][("event", "next", "event")].edge_index
    event_batch = payload["graph"]["event"].batch
    assert torch.equal(event_batch[next_edges[0]], event_batch[next_edges[1]])
    assert torch.all(next_edges[1] - next_edges[0] == 1)


def test_spatial_orientation_and_end_fallback():
    event = {
        "start_position": torch.tensor([[0.2, 0.3], [0.4, 0.6], [0.7, 0.8]]),
        "start_position_mask": torch.tensor([True, True, True]),
        "end_position": torch.tensor([[0.1, 0.4], [0.0, 0.0], [0.9, 0.2]]),
        "end_position_mask": torch.tensor([True, False, True]),
        "team_local_index": torch.tensor([0, 1, 0]),
    }
    points, valid = oriented_event_points(event, anchor_team=0)
    assert torch.allclose(points[0], torch.tensor([0.1, 0.4]))
    assert torch.allclose(points[1], torch.tensor([0.6, 0.4]))
    assert torch.allclose(points[2], torch.tensor([0.9, 0.2]))
    assert valid.tolist() == [True, True, True]


def test_candidate_features_are_prefix_aligned(dataset):
    sample = dataset[100]
    features = candidate_player_features(sample["window"])
    player_count = sample["window"]["node_stores"]["player"]["num_nodes"]
    assert tuple(features.shape) == (player_count, 10)
    assert torch.isfinite(features).all()
    assert torch.all((features >= 0) & (features <= 1))
    unseen = features[:, 6] == 0
    assert torch.all(features[unseen, 9] == 1)
    assert torch.all(features[unseen, 3:5] == 1)


@pytest.mark.parametrize("expert_mode", ["homogeneous", "structural"])
def test_single_hgt_forward_is_dense_and_exact_at_initialization(
    dataset, model_config, expert_mode
):
    payload = collate_data_aware_windows([dataset[20], dataset[100]])
    model = DataAwareResidualMoE(
        DataAwareMoEConfig(
            base=model_config,
            expert_mode=expert_mode,
            residual_mode="task_specific",
            routing_mode="dense",
            conditioned_router=True,
            position_correction=True,
            player_correction=True,
        )
    )
    model.eval()
    predictions = model(
        payload["graph"], payload["selections"], payload["candidate_features"]
    )
    assert model.hgt_forward_calls == 1
    assert tuple(predictions["routing_weights"].shape) == (2, len(TASK_NAMES), len(EXPERT_NAMES))
    assert tuple(predictions["task_residuals"].shape) == (
        2,
        len(TASK_NAMES),
        len(EXPERT_NAMES),
        model_config.hidden_channels,
    )
    assert torch.all(predictions["routing_weights"] > 0)
    assert torch.allclose(
        predictions["routing_weights"].sum(dim=-1), torch.ones(2, len(TASK_NAMES))
    )
    for candidate, base in (
        ("event_logits", "base_event_logits"),
        ("log_delta", "base_log_delta"),
        ("position", "base_position"),
        ("side_logits", "base_side_logits"),
        ("player_scores", "base_player_scores"),
        ("advantage_logits", "base_advantage_logits"),
    ):
        assert torch.allclose(predictions[candidate], predictions[base], atol=1e-7)
    model.train()
    assert not model.base.training
    assert not any(parameter.requires_grad for parameter in model.base.parameters())


def test_zero_initialized_path_learns_without_double_zero_gate(dataset, model_config):
    payload = collate_data_aware_windows([dataset[20], dataset[100]])
    model = DataAwareResidualMoE(
        DataAwareMoEConfig(
            base=model_config,
            expert_mode="homogeneous",
            residual_mode="task_specific",
            routing_mode="dense",
        )
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=3e-4
    )

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        predictions = model(
            payload["graph"], payload["selections"], payload["candidate_features"]
        )
        loss, _ = compute_multitask_loss(predictions, payload["graph"])
        loss.backward()
        optimizer.step()

    model.train()
    step()
    output_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "task_residual_outputs" in name and name.endswith("weight")
    ]
    assert output_gradients and any(
        gradient is not None and float(gradient.norm()) > 0 for gradient in output_gradients
    )
    step()
    stem_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "residual_stems" in name
    ]
    router_gradients = [
        parameter.grad for name, parameter in model.named_parameters() if "router_heads" in name
    ]
    assert any(gradient is not None and float(gradient.norm()) > 0 for gradient in stem_gradients)
    assert any(gradient is not None and float(gradient.norm()) > 0 for gradient in router_gradients)


def test_unknown_players_produce_zero_player_loss():
    scores = torch.tensor([0.2, 0.8, -0.1], requires_grad=True)
    loss = player_cross_entropy(
        scores,
        torch.tensor([0, 3]),
        torch.tensor([-1]),
        torch.tensor([False]),
    )
    assert float(loss) == 0.0
    loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))
