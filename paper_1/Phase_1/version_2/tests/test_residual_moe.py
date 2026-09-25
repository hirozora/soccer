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
from football_hgt_residual_v3.data import (  # noqa: E402
    FullHistoryResidualDataset,
    collate_full_history_residuals,
    validate_residual_sample,
)
from football_hgt_residual_v3.experts import (  # noqa: E402
    BRANCH_NAMES,
    EXPERT_NAMES,
    GraphViewConfig,
    StructuralExpertGenerator,
)
from football_hgt_residual_v3.model import (  # noqa: E402
    FullHistoryResidualMoE,
    ModelConfig,
    ResidualMoEConfig,
    TASK_NAMES,
    compute_residual_moe_loss,
)
from football_hgt_v1.data import window_to_heterodata  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402


@pytest.fixture(scope="module")
def graph_and_dataset():
    path = GRAPH_ROOT / "graphs/England/2499719.pt"
    graph = load_match_graph(path, validate=False)
    record = MatchGraphRecord(
        competition_slug="England",
        match_id=2499719,
        graph_path=path,
        num_events=int(graph["node_stores"]["event"]["num_nodes"]),
    )
    return graph, FullHistoryResidualDataset([record])


def test_full_history_and_experts_are_causal_and_share_target(graph_and_dataset):
    _, dataset = graph_and_dataset
    config = GraphViewConfig()
    for current in (0, 1, 20, 100, len(dataset) - 1):
        sample = dataset[current]
        assert validate_residual_sample(sample) == []
        full = sample["full_history"]
        assert full["window"]["current_event_index"] == current
        assert full["window"]["target_event_index"] == current + 1
        assert full["window"]["num_events"] == min(current + 1, config.full_history)
        assert tuple(sample["experts"]) == EXPERT_NAMES
        for name, view in sample["experts"].items():
            selected = view["selected_event_indices"]
            assert selected == sorted(set(selected))
            assert selected[-1] == current
            assert max(selected) <= current
            assert len(selected) <= getattr(config, name)
        assert all(0.0 <= value <= 1.0 for value in sample["overlap"].values())


def test_structural_experts_follow_candidate_horizons(graph_and_dataset):
    graph, _ = graph_and_dataset
    current = 100
    config = GraphViewConfig()
    views = StructuralExpertGenerator(config).generate(graph, current)
    assert min(views["spatial"]) >= current - config.spatial_horizon + 1
    assert min(views["actor_relation"]) >= current - config.actor_horizon + 1


def test_dense_residual_forward_and_backward(graph_and_dataset):
    _, dataset = graph_and_dataset
    payload = collate_full_history_residuals([dataset[20], dataset[100]])
    graph_batch = payload["graph"]
    assert graph_batch.num_graphs == 2 * len(BRANCH_NAMES)
    assert tuple(payload["overlap"].shape) == (2, 15)
    assert tuple(payload["view_sizes"].shape) == (2, len(BRANCH_NAMES))
    assert graph_batch["event"].ptr[:3].tolist() == [0, 21, 101]

    vocab = json.loads(
        (GRAPH_ROOT / "metadata/vocabularies.json").read_text(encoding="utf-8")
    )
    base = ModelConfig(
        num_event_types=len(vocab["event_type_ids"]),
        num_subevent_types=len(vocab["subevent_type_ids"]),
        num_players=len(vocab["player_ids"]),
        num_teams=len(vocab["team_ids"]),
        num_tags=len(vocab["tag_ids"]),
        hidden_channels=32,
    )
    model = FullHistoryResidualMoE(
        ResidualMoEConfig(
            base=base,
            adapter_channels=8,
            router_hidden_channels=32,
        )
    )
    model.eval()
    predictions = model(graph_batch, batch_size=2)
    routing = predictions["routing_weights"]
    assert tuple(predictions["event_logits"].shape) == (2, base.num_event_types)
    assert tuple(routing.shape) == (2, len(TASK_NAMES), len(EXPERT_NAMES))
    assert torch.allclose(routing.sum(dim=-1), torch.ones(2, len(TASK_NAMES)))
    assert torch.all(routing > 0.0)
    assert torch.count_nonzero(predictions["expert_residuals"]) == 0
    expected_base = predictions["base_context"][:, None, :].expand(
        -1, len(TASK_NAMES), -1
    )
    assert torch.equal(predictions["task_contexts"], expected_base)

    full_batch = Batch.from_data_list(
        [
            window_to_heterodata(dataset[index]["full_history"])
            for index in (20, 100)
        ]
    )
    base_predictions = model.base(full_batch)
    for residual_name, base_name in (
        ("event_logits", "event_logits"),
        ("log_delta", "log_delta"),
        ("position", "position"),
        ("side_logits", "side_logits"),
        ("advantage_logits", "advantage_logits"),
        ("player_scores", "player_scores"),
    ):
        assert torch.allclose(
            predictions[residual_name], base_predictions[base_name], atol=1e-6
        )

    model.set_base_trainable(False)
    assert not any(parameter.requires_grad for parameter in model.base.parameters())
    model.train()

    with torch.no_grad():
        for adapter in model.residual_adapters:
            adapter.up.weight.normal_(std=1e-3)
    predictions = model(graph_batch, batch_size=2)
    loss, parts = compute_residual_moe_loss(predictions, graph_batch, 2)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    loss.backward()
    router_gradient = model.router[-1].weight.grad
    assert router_gradient is not None
    assert float(router_gradient.norm()) > 0.0
