import json
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from football_hgt_targets_v4.event_triad import (
    TRIAD, GRID, TriadHead, HeadConfig, refine, state_features, match_folds,
    choose_bias, grid_predictions, fit, require_lock, load_cache, compare,
)
from football_hgt_targets_v4.event_posterior_integration import prediction_frame, selection_value, eligible
from football_hgt_targets_v4.position_head_refit import tensor_hash


@pytest.fixture(autouse=True)
def native_cpu():
    torch.set_num_threads(1)
    with torch.backends.mkldnn.flags(enabled=False):
        yield


def fake_cache(n=40):
    gen = torch.Generator().manual_seed(12)
    return {"main_context": torch.randn(n, 64, generator=gen), "state": state_features(torch.arange(n) % 10, torch.arange(n) % 4),
        "base_event_logits": torch.randn(n, 10, generator=gen), "event_true": torch.arange(n) % 10,
        "player_mask": torch.arange(n) % 3 != 0, "match_ids": torch.arange(n) // 10,
        "current_event_indices": torch.arange(n) % 10, "sample_ids": [f"{i//10}:{i%10}" for i in range(n)],
        "zone_true": torch.arange(n) % 20, "event_role": torch.zeros(n, dtype=torch.long),
        "control_state": torch.arange(n) % 4, "switch_confirmed": torch.zeros(n, dtype=torch.bool),
        "anchor_type": torch.arange(n) % 10}


def test_mass_zero_and_outside_probabilities():
    logits = torch.randn(100, 10, dtype=torch.float64)
    assert torch.equal(refine(logits, torch.zeros(3)), logits)
    output = refine(logits, torch.randn(100, 3, dtype=torch.float64))
    old, new = logits.softmax(-1), output.softmax(-1)
    outside = [i for i in range(10) if i not in TRIAD]
    torch.testing.assert_close(old[:, outside], new[:, outside], atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(old[:, list(TRIAD)].sum(-1), new[:, list(TRIAD)].sum(-1), atol=1e-14, rtol=1e-14)
    assert torch.equal(output[:, outside], logits[:, outside])


def test_outside_loss_gradient_zero():
    logits = torch.randn(12, 10, dtype=torch.float64)
    delta = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
    F.cross_entropy(refine(logits, delta), torch.ones(12, dtype=torch.long)).backward()
    assert delta.grad.abs().max() < 1e-14


def test_common_initialization_rng_zero_and_gradients():
    before = torch.get_rng_state().clone()
    a, b = TriadHead(23), TriadHead(23)
    assert torch.equal(before, torch.get_rng_state())
    assert tensor_hash(a.state_dict()) == tensor_hash(b.state_dict())
    assert sum(p.numel() for p in a.parameters()) == 2627
    c = fake_cache()
    optimizer = torch.optim.AdamW(a.parameters(), lr=.003, weight_decay=0.)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        delta = a(c["main_context"], c["state"])
        if step == 0:
            assert torch.equal(refine(c["base_event_logits"], delta), c["base_event_logits"])
        F.cross_entropy(refine(c["base_event_logits"], delta), c["event_true"]).backward()
        assert a.network[-1].weight.grad.abs().sum() > 0
        assert (a.network[0].weight.grad.abs().sum() == 0) == (step == 0)
        optimizer.step()


def test_fold_determinism_and_heldout_isolation():
    assert match_folds(np.arange(57)) == match_folds(np.arange(56, -1, -1))
    assert sorted(np.bincount(list(match_folds(np.arange(57)).values()))) == [11, 11, 11, 12, 12]
    caches = [fake_cache() for _ in range(3)]
    preds = grid_predictions(caches)
    rows = np.arange(20)
    old = choose_bias(caches, preds, rows)
    for c in caches:
        c["event_true"][20:] = (c["event_true"][20:] + 3) % 10
        c["base_event_logits"][20:] *= -30
    assert old == choose_bias(caches, grid_predictions(caches), rows)
    assert (0., 0., 0.) in GRID and len(GRID) == 169


def test_selection_guard_and_test_gate(tmp_path):
    with pytest.raises(RuntimeError, match="lock"):
        require_lock(tmp_path)
    with pytest.raises(RuntimeError, match="lock"):
        load_cache(20260715, "test", tmp_path)
    (tmp_path / "selection").mkdir()
    (tmp_path / "selection/event_triad_lock.json").write_text(json.dumps({"passed": False}))
    with pytest.raises(RuntimeError, match="prohibited"):
        require_lock(tmp_path)
    old = {"macro_f1": .6, "accuracy": .8, "ce": 1.}
    assert eligible({**old, "accuracy": .79}, old)
    assert not eligible({**old, "accuracy": .789}, old)
    assert selection_value(old, 0) < selection_value(old, 1)


def test_resume_exact_and_training_control(tmp_path):
    c = fake_cache()
    cfg = HeadConfig(23, "state", max_epochs=3, patience=10, batch_size=16)
    fit(c, c, cfg, tmp_path / "full", {})
    fit(c, c, cfg, tmp_path / "resume", {}, stop_after=1)
    fit(c, c, cfg, tmp_path / "resume", {}, resume=tmp_path / "resume/last.pt")
    a = torch.load(tmp_path / "full/last.pt", weights_only=False)
    b = torch.load(tmp_path / "resume/last.pt", weights_only=False)
    assert tensor_hash(a["model"]) == tensor_hash(b["model"])
    assert a["history"] == b["history"] and a["epoch"] == 3
    ctx = HeadConfig(23, "context", max_epochs=3, patience=10, batch_size=16)
    fit(c, c, ctx, tmp_path / "context", {})
    d = torch.load(tmp_path / "context/last.pt", weights_only=False)
    assert a["initial_hash"] == d["initial_hash"]
    assert [h["sample_order_sha256"] for h in a["history"]] == [h["sample_order_sha256"] for h in d["history"]]
    assert all(not value.requires_grad for value in c.values() if isinstance(value, torch.Tensor))


def test_bootstrap_pairing_shared_draws():
    import pandas as pd
    c = fake_cache()
    frame = pd.concat([prediction_frame(c, c["base_event_logits"], seed) for seed in (1, 2, 3)], ignore_index=True)
    result = compare(frame, frame.copy())
    assert result["macro_f1_gain"] == 0 and result["ci95"] == [0., 0.]
    assert result["resampling_unit"] == "match" and result["replicates"] == 10000
    again = compare(frame, frame.copy())
    assert result["draws_sha256"] == again["draws_sha256"]
    changed = frame.copy()
    changed.loc[0, "event_true"] = 9
    with pytest.raises(RuntimeError):
        compare(frame, changed)


def test_epoch_zero_fallback_early_stopping(tmp_path, monkeypatch):
    import football_hgt_targets_v4.event_triad as module
    calls = 0
    def declining_metrics(logits, target):
        nonlocal calls
        calls += 1
        return {"macro_f1": .6 if calls == 1 else .5, "accuracy": .8,
                "ce": 1., "samples": len(target), "confusion": np.eye(10, dtype=int).tolist()}
    monkeypatch.setattr(module, "metrics", declining_metrics)
    result = fit(fake_cache(), fake_cache(), HeadConfig(23, "context", patience=2), tmp_path, {})
    assert result["best_epoch"] == 0 and result["completed_epoch"] == 2 and result["completed"]
    saved = torch.load(tmp_path / "best_event.pt", weights_only=False)
    assert tensor_hash(saved["model"]) == tensor_hash(TriadHead(23).state_dict())


def test_state_inputs_have_no_target_path():
    c = fake_cache()
    original = state_features(c["anchor_type"], c["control_state"])
    c["event_true"].fill_(9)
    c["player_mask"].fill_(False)
    assert torch.equal(original, state_features(c["anchor_type"], c["control_state"]))
    assert original.shape[1] == 14 and torch.equal(original.sum(-1), torch.full((40,), 2.))
