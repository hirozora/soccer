from __future__ import annotations

import torch

from football_benchmark.models import ModelSpec, build_model
from football_benchmark.training import (
    TrainingConfig,
    evaluate_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_round_trip(tmp_path, artifacts) -> None:
    model = build_model(ModelSpec("seq2event", "seq2event", 40), artifacts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    config = TrainingConfig(
        contract="seq2event",
        family="seq2event",
        window_size=40,
        learning_rate=1e-3,
        seed=1,
        output_dir=tmp_path,
        device="cpu",
    )
    save_checkpoint(path, model, optimizer, 3, 0.4, config)
    clone = build_model(ModelSpec("seq2event", "seq2event", 40), artifacts)
    state = load_checkpoint(path, clone)
    assert state["epoch"] == 3
    for original, restored in zip(model.parameters(), clone.parameters()):
        assert torch.equal(original, restored)


def test_checkpoint_evaluation_rejects_mismatched_contract(tmp_path, artifacts) -> None:
    model = build_model(ModelSpec("seq2event", "seq2event", 40), artifacts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    source = TrainingConfig(
        contract="seq2event",
        family="seq2event",
        window_size=40,
        learning_rate=1e-3,
        seed=1,
        output_dir=tmp_path,
        device="cpu",
    )
    save_checkpoint(path, model, optimizer, 1, 0.5, source)
    mismatched = TrainingConfig(
        contract="unified_lem",
        family="seq2event",
        window_size=40,
        learning_rate=1e-3,
        seed=1,
        output_dir=tmp_path / "evaluation",
        device="cpu",
    )
    try:
        evaluate_checkpoint(mismatched, artifacts, path)
    except ValueError as exc:
        assert "Checkpoint contract" in str(exc)
    else:
        raise AssertionError("mismatched checkpoint contract was accepted")


def test_checkpoint_records_graph_variant(tmp_path, artifacts) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("torch_geometric")
    model = build_model(
        ModelSpec("hgt", "unified_lem", 8, graph_variant="semantic_v2"), artifacts
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "semantic.pt"
    config = TrainingConfig(
        contract="unified_lem",
        family="hgt",
        window_size=8,
        learning_rate=1e-3,
        seed=1,
        output_dir=tmp_path,
        device="cpu",
        graph_variant="semantic_v2",
    )
    save_checkpoint(path, model, optimizer, 1, 0.5, config)
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["config"]["graph_variant"] == "semantic_v2"
