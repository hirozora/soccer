from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from football_hgt_targets_v4.position_head_refit import (
    RefitConfig, fit_cached, head_from_state, load_cache, load_predictions, metrics,
    paired_bootstrap, position_loss, predict, require_test_lock, selection_value, tensor_hash,
)


def source_state():
    generator = torch.Generator().manual_seed(31)
    return {"position_head.weight": torch.randn(2, 64, generator=generator) * 0.01,
            "position_head.bias": torch.zeros(2), "other_head": torch.ones(2)}


def cache(n=64, seed=18):
    generator = torch.Generator().manual_seed(seed)
    return {"main_context": torch.randn(n, 64, generator=generator),
            "position_true": torch.rand(n, 2, generator=generator),
            "position_mask": torch.ones(n, dtype=torch.bool),
            "player_mask": torch.zeros(n, dtype=torch.bool)}


def test_original_architecture_initialization_rng_and_parameter_isolation():
    state = source_state()
    before = tensor_hash(state)
    rng = torch.get_rng_state().clone()
    head = head_from_state(state)
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in head.parameters()) == 130
    x = cache()["main_context"]
    torch.testing.assert_close(predict(head, x), torch.sigmoid(x @ state["position_head.weight"].T + state["position_head.bias"]), rtol=0, atol=1e-6)
    position_loss(head(x).sigmoid(), torch.zeros(len(x), 2), torch.ones(len(x), dtype=torch.bool)).backward()
    assert all(p.grad.abs().sum() > 0 for p in head.parameters())
    with torch.no_grad():
        head.weight.add_(1)
    assert tensor_hash(state) == before


def test_invalid_positions_are_excluded_and_unknown_players_are_included():
    prediction = torch.tensor([[0.0, 0.0], [0.5, 0.5]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0], [float("nan"), float("nan")]])
    mask = torch.tensor([True, False])
    loss = position_loss(prediction, target, mask)
    assert float(loss) == pytest.approx(0.25)
    loss.backward()
    assert prediction.grad[0, 0] != 0 and prediction.grad[1].abs().sum() == 0
    m = metrics(prediction.detach(), target, mask)
    assert m["samples"] == 1 and m["mean_distance_m"] == 105.0
    m = metrics(torch.tensor([[0.0, 1.0]]), torch.zeros(1, 2), torch.tensor([True]))
    assert m["mean_distance_m"] == 68.0


def test_epoch_zero_fallback_and_patience(tmp_path):
    head = head_from_state(source_state())
    train, validation = cache(seed=1), cache(seed=2)
    validation["position_true"] = predict(head, validation["main_context"])
    result = fit_cached(head, train, validation, RefitConfig(31, patience=2, max_epochs=8), tmp_path, {})
    assert result["best_epoch"] == 0
    assert result["completed_epoch"] == 2 and result["stopped_early"]
    assert selection_value({"mean_distance_m": 1.0, "loss": 2.0}, 0) < selection_value({"mean_distance_m": 1.0, "loss": 2.0}, 1)


def test_resume_matches_uninterrupted_training_and_rng(tmp_path):
    train, validation = cache(seed=3), cache(seed=4)
    cfg = RefitConfig(31, batch_size=16, max_epochs=5, patience=10)
    fit_cached(head_from_state(source_state()), train, validation, cfg, tmp_path / "whole", {})
    fit_cached(head_from_state(source_state()), train, validation, cfg, tmp_path / "resume", {}, stop_after_epoch=2)
    result = fit_cached(head_from_state(source_state()), train, validation, cfg, tmp_path / "resume", {}, resume_from=tmp_path / "resume/last.pt")
    whole = torch.load(tmp_path / "whole/last.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resume/last.pt", weights_only=False)
    assert whole["history"] == resumed["history"]
    assert tensor_hash(whole["head"]) == tensor_hash(resumed["head"])
    assert tensor_hash(whole["best"]["head"]) == tensor_hash(resumed["best"]["head"])
    assert torch.equal(whole["rng"]["sampler"], resumed["rng"]["sampler"])
    assert torch.equal(whole["rng"]["cpu"], resumed["rng"]["cpu"])
    assert result["completed_epoch"] == 5
    with pytest.raises(RuntimeError, match="changed"):
        fit_cached(head_from_state(source_state()), train, validation, cfg, tmp_path / "resume", {"changed": True}, resume_from=tmp_path / "resume/last.pt")


def bootstrap_frames():
    rows = []
    for seed in (1, 2, 3):
        for match, count, error in ((10, 2, 2.0), (20, 5, 4.0), (30, 3, 8.0)):
            for i in range(count):
                rows.append(dict(seed=seed, sample_id=f"{match}:{i}", match_id=match, current_event_index=i,
                                 position_mask=True, player_mask=False, position_true_x=0.0, position_true_y=0.0,
                                 position_pred_x=error / 105.0, position_pred_y=0.0))
    ref = pd.DataFrame(rows)
    candidate = ref.copy()
    candidate.position_pred_x = 0.0
    return ref, candidate


def test_bootstrap_is_match_clustered_shared_across_seeds_and_has_explicit_population():
    ref, cand = bootstrap_frames()
    mask = ref.position_mask.to_numpy(dtype=bool)
    actual = paired_bootstrap(ref, cand, mask, replicates=100, selector_seed=1)
    assert actual["effective"] and actual["improving_seeds"] == 3
    assert actual["improvement_m"] == pytest.approx(4.8)
    draws = np.random.default_rng(1).integers(3, size=(100, 3))
    counts = np.array([2, 5, 3])
    sums = np.array([4, 20, 24])
    expected = sums[draws].sum(1) / counts[draws].sum(1)
    assert actual["ci95"] == pytest.approx(np.quantile(expected, [0.025, 0.975]))
    repeated = paired_bootstrap(ref, cand, mask, replicates=100, selector_seed=1)
    assert actual == repeated
    cand.loc[0, "position_true_x"] = 0.1
    with pytest.raises(RuntimeError, match="differ"):
        paired_bootstrap(ref, cand, mask)


def test_test_gate_precedes_cache_or_prediction_reads(tmp_path):
    with pytest.raises(RuntimeError, match="lock required"):
        load_cache(20260715, "test", tmp_path)
    with pytest.raises(RuntimeError, match="lock required"):
        load_predictions("test", "original", tmp_path)
    (tmp_path / "selection").mkdir()
    (tmp_path / "selection/position_refit_lock.json").write_text(json.dumps({"passed": False}))
    with pytest.raises(RuntimeError, match="did not authorize"):
        require_test_lock(tmp_path)
