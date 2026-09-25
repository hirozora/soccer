import json

import numpy as np
import pandas as pd
import pytest
import torch

from football_hgt_targets_v4.event_posterior_integration import (
    EventConfig, EventResidual, cm_scores, condition_for, confusion, eligible, expected_player_state,
    fit_cached, load_cache, paired_bootstrap, read_predictions, require_test_lock, selection_value,
)


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def toy():
    g = torch.Generator().manual_seed(5)
    return {"main_context": torch.randn(28, 64, generator=g), "condition": torch.randn(28, 64, generator=g),
            "base_event_logits": torch.randn(28, 10, generator=g), "event_true": torch.randint(10, (28,), generator=g)}


def test_zero_init_rng_and_two_stage_gradients():
    torch.manual_seed(7)
    rng = torch.get_rng_state().clone()
    a, b = EventResidual(9), EventResidual(9)
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in a.parameters()) == 8906
    for x, y in zip(a.parameters(), b.parameters()):
        assert torch.equal(x, y) and x.data_ptr() != y.data_ptr()
    data = toy()
    context = data["main_context"].requires_grad_()
    condition = data["condition"].requires_grad_()
    optimizer = torch.optim.AdamW(a.parameters(), lr=3e-4)
    a.eval()
    assert torch.equal(a(context, condition), torch.zeros(28, 10))
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(data["base_event_logits"] + a(context, condition), data["event_true"])
        loss.backward()
        assert a.network[-1].weight.grad.abs().sum() > 0
        assert bool(a.network[0].weight.grad.abs().sum() > 0) == (step == 1)
        assert context.grad is None and condition.grad is None
        optimizer.step()


def test_condition_targets_and_null():
    g = torch.Generator().manual_seed(3)
    states, scores, ptr = torch.randn(7, 64, generator=g), torch.randn(7, generator=g), torch.tensor([0, 3, 7])
    result = expected_player_state(scores, states, ptr)
    for row, (start, end) in enumerate(((0, 3), (3, 7))):
        p = scores[start:end].softmax(0)
        torch.testing.assert_close(p.sum(), torch.tensor(1.0))
        torch.testing.assert_close(result[row], (p[:, None] * states[start:end]).sum(0), rtol=0, atol=0)
    cache = {"condition": result, "player_mask": torch.tensor([True, False]), "player_true": torch.tensor([0, 1])}
    rows = torch.arange(2)
    before = condition_for(cache, rows, "base_post").clone()
    cache["player_mask"].logical_not_()
    cache["player_true"].fill_(999)
    assert torch.equal(before, condition_for(cache, rows, "base_post"))
    assert torch.count_nonzero(condition_for(cache, rows, "null")) == 0
    with pytest.raises(ValueError):
        expected_player_state(scores, states, torch.tensor([0, 0, 7]))


def test_guard_selection_epoch_zero(tmp_path):
    values = {"macro_f1": 0.5, "accuracy": 0.7, "ce": 1.0}
    assert eligible({**values, "accuracy": 0.69}, values)
    assert not eligible({**values, "accuracy": 0.689}, values)
    assert selection_value(values, 0) < selection_value(values, 1)
    data = toy()
    result = fit_cached(data, data, EventConfig(5, "null", lr=0, max_epochs=8, patience=2), tmp_path, {})
    assert result["best_epoch"] == 0 and result["completed_epoch"] == 2 and result["stopped_early"]


def test_exact_resume_and_order(tmp_path):
    data = toy()
    cfg = EventConfig(8, "base_post", max_epochs=4, patience=8, batch_size=7)
    fit_cached(data, data, cfg, tmp_path / "full", {})
    fit_cached(data, data, cfg, tmp_path / "resume", {}, stop_after_epoch=2)
    fit_cached(data, data, cfg, tmp_path / "resume", {}, resume_from=tmp_path / "resume/last.pt")
    a = torch.load(tmp_path / "full/last.pt", weights_only=False)
    b = torch.load(tmp_path / "resume/last.pt", weights_only=False)
    assert a["history"] == b["history"] and a["epoch"] == b["epoch"]
    for key in a["residual"]:
        assert torch.equal(a["residual"][key], b["residual"][key])
    assert torch.equal(a["rng"]["cpu"], b["rng"]["cpu"])
    assert torch.equal(a["rng"]["sampler"], b["rng"]["sampler"])
    fit_cached(data, data, EventConfig(8, "null", max_epochs=4, patience=8, batch_size=7), tmp_path / "null", {})
    c = torch.load(tmp_path / "null/last.pt", weights_only=False)
    assert c["initial_hash"] == a["initial_hash"]
    assert [x["sample_order_sha256"] for x in a["history"]] == [x["sample_order_sha256"] for x in c["history"]]


def test_cluster_confusions_shared_draws():
    frames = []
    for seed in (1, 2, 3):
        frames.append(pd.DataFrame({"seed": seed, "sample_id": ["a", "b", "c", "d"], "match_id": [10, 20, 20, 20],
            "current_event_index": [0, 0, 1, 2], "event_true": [0, 1, 1, 0], "player_mask": [False, True, False, True],
            "event_pred": [1, 0, 1, 0]}))
    ref = pd.concat(frames, ignore_index=True)
    cand = ref.copy()
    cand["event_pred"] = cand["event_true"]
    result = paired_bootstrap(ref, cand, replicates=100, selector_seed=4)
    sampled = np.random.default_rng(4).integers(2, size=(100, 2))
    original = frames[0]
    matrices = [confusion(group.event_true, group.event_pred) for _, group in original.groupby("match_id")]
    exact = []
    for draw in sampled:
        bad = sum(matrices[i] for i in draw)
        good = np.diag(bad.sum(1))
        exact.append(float(cm_scores(good)[0] - cm_scores(bad)[0]))
    np.testing.assert_allclose(result["ci95"], np.quantile(exact, [.025, .975]), atol=1e-12)
    assert result["effective"]
    assert not paired_bootstrap(ref, ref, 100, 4)["effective"]
    assert result["draws_sha256"] == paired_bootstrap(ref, ref, 100, 4)["draws_sha256"]
    cand.loc[0, "player_mask"] = True
    with pytest.raises(RuntimeError):
        paired_bootstrap(ref, cand)


def test_test_gate_before_io(tmp_path):
    with pytest.raises(RuntimeError):
        require_test_lock(tmp_path)
    with pytest.raises(RuntimeError):
        load_cache(20260715, "test", tmp_path)
    with pytest.raises(RuntimeError):
        read_predictions("test", "original", tmp_path)
    (tmp_path / "selection").mkdir()
    (tmp_path / "selection/event_posterior_lock.json").write_text('{"passed":false}')
    with pytest.raises(RuntimeError):
        require_test_lock(tmp_path)


@pytest.mark.parametrize("null_effective,original_effective", [(False, True), (True, False), (True, True)])
def test_admission_requires_both_controls(tmp_path, monkeypatch, null_effective, original_effective):
    import football_hgt_targets_v4.event_posterior_integration as module
    from football_hgt_targets_v4.position_head_refit import sha256
    for seed in module.CONFIRMATION_SEEDS:
        for mode in module.MODES:
            directory = module.run_dir(seed, mode, tmp_path)
            directory.mkdir(parents=True)
            checkpoint = directory / "best_event.pt"
            checkpoint.write_bytes(b"test checkpoint")
            (directory / "result.json").write_text(json.dumps({"completed": True, "initial_hash": "same", "best_epoch": 0}))
            (directory / "history.json").write_text('[{"sample_order_sha256":null}]')
            (directory / "online_verification.json").write_text(json.dumps({"passed": True, "head_sha256": sha256(checkpoint)}))
    monkeypatch.setattr(module, "comparisons", lambda *args: {
        "base_post_minus_null": {"effective": null_effective},
        "base_post_minus_original": {"effective": original_effective}})
    lock = module.select(tmp_path)
    assert lock["passed"] == (null_effective and original_effective)
    if lock["passed"]:
        module.require_test_lock(tmp_path)
        checkpoint.write_bytes(b"tampered")
    with pytest.raises(RuntimeError):
        module.require_test_lock(tmp_path)
    with pytest.raises(RuntimeError):
        load_cache(20260715, "test", tmp_path)
    with pytest.raises(RuntimeError):
        read_predictions("test", "original", tmp_path)
