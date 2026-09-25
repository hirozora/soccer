"""Protocol gates and deterministic sampling/history contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import pandas as pd
from torch.utils.data import DataLoader

from football_benchmark.data import CanonicalEventDataset
from football_hgt_targets_v4 import coverage_history_data as data
from football_hgt_targets_v4.coverage_history_probe import (HistoryResidual, ProbeConfig, condition_for,
    fit_history, predict, choose_history)
from football_hgt_targets_v4.coverage_history_training import choose_coverage, require_test, frame_path, loader_for
from football_hgt_targets_v4.position_head_refit import tensor_hash
from football_hgt_targets_v4.event_posterior_integration import paired_bootstrap


@pytest.fixture(autouse=True)
def native_cpu_kernels():
    with torch.backends.mkldnn.flags(enabled=False):
        yield


@pytest.mark.parametrize("size", [1, 127, 128, 1386, 1978])
def test_rotation(size):
    n = min(size, 128)
    first = [s * size // n for s in range(n)]
    state = torch.get_rng_state().clone()
    table = data.rotation_table(size, first, 101)
    assert torch.equal(table[0], torch.tensor(first))
    assert torch.equal(table, data.rotation_table(size, first, 101))
    assert torch.equal(state, torch.get_rng_state())
    assert len(torch.unique(table[:16])) == size
    for row in table:
        assert torch.all(row[1:] > row[:-1])
        assert row.min() >= 0 and row.max() < size


def fake_sample(self, index):
    return self.currents[0][index]


def test_persistent_workers_see_epoch_and_resume(monkeypatch):
    monkeypatch.setattr(CanonicalEventDataset, "__getitem__", fake_sample)
    ds = object.__new__(data.RotatingEventDataset)
    ds.total = 128
    ds.currents = []
    ds.rotate = True
    ds.tables = [data.rotation_table(1978, [s * 1978 // 128 for s in range(128)], 55).share_memory_()]
    ds.epoch_state = torch.ones(1, dtype=torch.long).share_memory_()
    ds.local_epoch = None
    loader = DataLoader(ds, batch_size=128, num_workers=2, persistent_workers=True)
    first = next(iter(loader))
    ds.set_epoch(2)
    second = next(iter(loader))
    assert torch.equal(first, ds.tables[0][0])
    assert torch.equal(second, ds.tables[0][1])
    ds.set_epoch(16)
    assert torch.equal(next(iter(loader)), ds.tables[0][15])
    loader._iterator._shutdown_workers()


def test_history_smoothing_and_cold_start():
    assert not data.summary13(np.zeros(10), 0, np.ones(10)).any()
    counts = np.zeros(10); counts[7] = 2
    x = data.summary13(counts, 1, np.ones(10))
    assert np.isclose(x[:10].sum(), 1)
    assert np.isclose(x[7], 4 / 22)
    assert x[-1] == 1 and np.isclose(x[10], np.log(3))


def test_history_cache_creates_directory_and_reloads(monkeypatch, tmp_path):
    from football_hgt_targets_v4 import coverage_history_probe as probe
    lock = tmp_path / "selection/coverage_lock.json"
    lock.parent.mkdir()
    lock.write_text('{}')
    monkeypatch.setattr(probe, "selected_coverage", lambda root: "rotate")
    monkeypatch.setattr(probe, "sources", lambda: {})
    monkeypatch.setattr(probe, "MATCHES", lock)
    monkeypatch.setattr(probe, "COUNTS", {"train": 1})
    model = SimpleNamespace(backbone=torch.nn.Linear(1, 1))
    model.parameters = model.backbone.parameters
    monkeypatch.setattr(probe, "load_deployed", lambda *a: model)
    monkeypatch.setattr(probe.ProtocolArtifacts, "load", lambda *a: None)
    monkeypatch.setattr(probe, "MatchHistoryIndex", lambda **kw: SimpleNamespace(offside_audit={}, snapshots={}))
    event = SimpleNamespace(ptr=torch.tensor([0, 1]), event_role_index=torch.tensor([1]),
        control_state_after_index=torch.tensor([1]), switch_confirmed=torch.tensor([False]))
    batch = {"targets": {"raw_event_10": torch.tensor([7]), "player_mask": torch.tensor([True]),
        "zone_20": torch.tensor([1])}, "graphs": {"f80": {"event": event}},
        "match_ids": torch.tensor([1]), "current_event_indices": torch.tensor([0]), "sample_ids": ["1:0"]}
    monkeypatch.setattr(probe, "loader_for", lambda *a: [batch])
    monkeypatch.setattr(probe, "_move_batch_to_device", lambda b, d: b)
    monkeypatch.setattr(probe, "forward_details", lambda *a: ({"event_logits": torch.zeros(1, 10)},
        torch.zeros(1, 64), torch.zeros(1, 39), torch.zeros(1, 39), torch.zeros(1, 3)))
    result = probe.cache_history(1, "train", "cpu", tmp_path)
    path = tmp_path / "history_cache/seed1/train.pt"
    assert path.exists() and path.with_suffix('.history_sources.json').exists()
    monkeypatch.setattr(probe, "loader_for", lambda *a: pytest.fail("Cache must be reused"))
    restored = probe.cache_history(1, "train", "cpu", tmp_path)
    assert torch.equal(result['main_context'], restored['main_context'])


def test_report_accepts_both_probability_schemas():
    from football_hgt_targets_v4.coverage_history_reporting import describe_event, probability_column
    frame = pd.DataFrame({"event_true": [5, 7], "event_pred": [5, 7], "event_probability_5": [.9, .1]})
    named = frame.rename(columns={"event_probability_5": "event_probability_5_Offside"})
    assert describe_event(frame) == describe_event(named)
    assert probability_column(named, 5) == 'event_probability_5_Offside'


def test_derangement():
    mapping = {1: 10, 2: 10, 3: 10, 4: 20, 5: 20, 6: 30}
    shuffled = data.derangement(mapping, 44)
    assert shuffled == data.derangement(mapping, 44)
    assert 6 not in shuffled
    assert set(shuffled) == set(shuffled.values())
    assert all(p != q and mapping[p] == mapping[q] for p, q in shuffled.items())


def synthetic_graph(label=7):
    return {"node_stores": {"event": {"event_type_index": torch.tensor([label, 5]),
        "team_local_index": torch.tensor([0, 1]), "player_local_index": torch.tensor([0, 1])},
        "team": {"raw_id": torch.tensor([10, 20])}, "player": {"raw_id": torch.tensor([1, 4])}}}


def test_causal_history_prefix_and_future(monkeypatch, tmp_path):
    times = ["2020-01-01 00:00:00", "2020-01-01 22:00:00", "2020-01-02 00:00:00", "2020-01-03 23:00:00"]
    records = [SimpleNamespace(match_id=i + 1, graph_path=Path(str(i + 1))) for i in range(4)]
    metadata = [{"wyId": i + 1, "dateutc": t, "status": "Played", "teamsData": {"10": {}, "20": {}}} for i, t in enumerate(times)]
    path = tmp_path / "matches.json"; path.write_text(json.dumps(metadata))
    monkeypatch.setattr(data, "MATCHES", path)
    monkeypatch.setattr(data, "load_roster_team_map", lambda: {i: {1: 10, 2: 10, 4: 20, 5: 20} for i in range(1, 5)})
    monkeypatch.setattr(data, "load_records", lambda split, **kwargs: records if split == "train" else [])
    loaded = []
    def read(path, **kwargs):
        loaded.append(int(str(path)))
        return synthetic_graph()
    monkeypatch.setattr(data.torch, "load", read)
    full = data.MatchHistoryIndex()
    assert full.snapshots[2]["source_matches"] == ()
    assert full.snapshots[3]["source_matches"] == (1,)
    assert 4 not in loaded
    monkeypatch.setattr(data, "load_records", lambda split, **kwargs: records[:3] if split == "train" else [])
    prefix = data.MatchHistoryIndex()
    assert np.array_equal(prefix.snapshots[3]["team"][10], full.snapshots[3]["team"][10])
    monkeypatch.setattr(data.torch, "load", lambda path, **kwargs: synthetic_graph(8) if str(path) == "4" else synthetic_graph())
    assert np.array_equal(data.MatchHistoryIndex().snapshots[3]["team"][10], prefix.snapshots[3]["team"][10])


def test_condition_only_uses_visible_arguments():
    index = object.__new__(data.MatchHistoryIndex)
    index.snapshots = {1: {"teams": [10, 20], "team": {10: np.ones(13), 20: np.ones(13)*2},
        "player": {1: np.ones(13)*3, 2: np.ones(13)*4}, "shuffle": {1: 2, 2: 1}, "team_games": {10: 5}}}
    correct, wrong, _ = data.condition39(index, [1], [10], [1, 2], [0, 2], torch.tensor([0., 0.]))
    assert torch.allclose(correct[:, 26:], torch.full((1, 13), 3.5, dtype=correct.dtype))
    assert torch.equal(correct, wrong)
    assert not condition_for({"condition": correct}, torch.tensor([0]), "null").any()
    assert not condition_for({"condition": correct}, torch.tensor([0]), "team")[:, 26:].any()


def tiny_cache():
    g = torch.Generator().manual_seed(42)
    return {"main_context": torch.randn(32, 64, generator=g), "condition": torch.randn(32, 39, generator=g),
        "shuffled_condition": torch.randn(32, 39, generator=g), "base_event_logits": torch.randn(32, 10, generator=g),
        "event_true": torch.arange(32) % 10}


def test_initialization_and_gradient_stages():
    before = torch.get_rng_state().clone()
    model = HistoryResidual(123)
    assert torch.equal(before, torch.get_rng_state())
    assert tensor_hash(model.state_dict()) == tensor_hash(HistoryResidual(123).state_dict())
    cache = tiny_cache()
    for mode in ("null", "team", "team_player", "shuffled_player"):
        assert torch.equal(predict(model, cache, mode), cache["base_event_logits"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for step in range(2):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = cache["base_event_logits"] + model(cache["main_context"], cache["condition"])
        torch.nn.functional.cross_entropy(logits, cache["event_true"]).backward()
        assert model.layers[-1].weight.grad.norm() > 0
        assert (model.layers[0].weight.grad.norm() == 0) if step == 0 else (model.layers[0].weight.grad.norm() > 0)
        optimizer.step()


def test_probe_exact_resume(tmp_path):
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    cache = tiny_cache()
    config = ProbeConfig(123, "team_player", epochs=3, patience=5, batch_size=8)
    fit_history(cache, cache, config, tmp_path / "continuous", {})
    fit_history(cache, cache, config, tmp_path / "resumed", {}, stop_after=1)
    fit_history(cache, cache, config, tmp_path / "resumed", {}, resume_from=tmp_path / "resumed/last.pt")
    a, b = [torch.load(tmp_path / name / "last.pt", weights_only=False) for name in ("continuous", "resumed")]
    assert tensor_hash(a["residual"]) == tensor_hash(b["residual"])
    assert a["history"] == b["history"]


def test_test_gate_before_loading(tmp_path):
    with pytest.raises(RuntimeError): require_test(tmp_path)
    with pytest.raises(RuntimeError): frame_path("original", 1, "test", tmp_path)
    with pytest.raises(RuntimeError): loader_for("test", 1, "cpu", tmp_path)
    with pytest.raises(RuntimeError): data.MatchHistoryIndex(include_test=True, root=tmp_path)


def test_history_selection_controls():
    good = {"effective": True, "ci95": [.001, .02], "improving_seeds": 3}
    pairs = ["team-null", "team-original", "team_player-null", "team_player-original", "team_player-team", "team_player-shuffled_player"]
    comparisons = {k: dict(good) for k in pairs}
    assert choose_history(comparisons, {"team": .6, "team_player": .62})[0] == "team_player"
    comparisons["team_player-shuffled_player"]["ci95"] = [-.001, .02]
    assert choose_history(comparisons, {"team": .6, "team_player": .62})[0] == "team"
    comparisons["team-null"]["effective"] = False
    assert choose_history(comparisons, {"team": .6, "team_player": .62})[0] == "original"


def test_coverage_no_effect_falls_back():
    comparison = {k: {"difference": 0., "per_seed": [0., 0., 0.], "ci_low": -.01, "ci_high": .01}
                  for k in ("event_macro_f1", "time_mae_seconds", "position_distance_mae_m")}
    assert choose_coverage({"fixed-original": comparison, "rotate-original": comparison},
                           {"original": .04, "fixed": .03, "rotate": .02}) == ("original", ["original"])


def test_match_bootstrap_shared_draws_and_alignment():
    rows = [{"seed": s, "sample_id": f"{m}:{i}", "match_id": m, "current_event_index": i,
             "event_true": i % 10, "event_pred": i % 10, "player_mask": bool(i % 2)}
            for s in (1, 2, 3) for m in (11, 12, 13) for i in range(20)]
    frame = pd.DataFrame(rows)
    a = paired_bootstrap(frame, frame.copy(), replicates=100)
    b = paired_bootstrap(frame, frame.copy(), replicates=100)
    assert a["ci95"] == [0., 0.] and a["macro_f1_gain"] == 0
    assert a["draws_sha256"] == b["draws_sha256"] and a["resampling_unit"] == "match"
    broken = frame.copy(); broken.loc[0, "match_id"] = 99
    with pytest.raises(RuntimeError): paired_bootstrap(frame, broken, replicates=100)
