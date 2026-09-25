"""Real-data coverage audit and GPU smoke before any scheduled training."""
import json
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from football_benchmark.data import load_records
from football_benchmark.protocol import ProtocolArtifacts
from football_benchmark.sampling import TargetSamplePlan
from .constants import SAMPLE_PLAN, POSSESSION_GRAPH_ROOT, FEASIBILITY_ARTIFACT
from .coverage_history_data import rotation_table, MatchHistoryIndex
from .coverage_history_training import (ROOT, SEEDS, loader_for, train_coverage, load_deployed, sources)
from .coverage_history_probe import (HistoryResidual, ProbeConfig, fit_history, forward_details, predict, condition_for)
from .fixed_budget_loss import fixed_budget_loss
from .model import build_partial_l2_model
from .position_head_refit import tensor_hash, sha256
from .spatiotemporal_training import write_json
from .training import set_seed, _move_batch_to_device


def audit(root=ROOT):
    plan = TargetSamplePlan.load(SAMPLE_PLAN)
    counts = torch.zeros((24, 10), dtype=torch.long)
    unique = torch.zeros(24, dtype=torch.long)
    tables, total, max_stratum = {}, 0, 0
    for record in load_records("train", graph_root=POSSESSION_GRAPH_ROOT):
        n = record.num_events - 1
        table = rotation_table(n, plan.currents_by_match("train")[record.match_id], record.match_id)
        graph = torch.load(record.graph_path, map_location="cpu", weights_only=False)
        labels = graph["node_stores"]["event"]["event_type_index"]
        total += n
        max_stratum = max(max_stratum, (n + 127) // 128)
        for e in range(24):
            counts[e] += torch.bincount(labels[table[e] + 1], minlength=10)
            unique[e] += torch.unique(table[:e + 1]).numel()
        if torch.unique(table[:16]).numel() != n:
            raise RuntimeError("Rotation failed full coverage")
        tables[record.match_id] = table
    assert total == 449025 and int(unique[15]) == total
    assert all(int(c.sum()) == 34048 for c in counts)
    path = root / "verification"
    path.mkdir(parents=True, exist_ok=True)
    torch.save(tables, path / "rotation_plan.pt")
    index = MatchHistoryIndex()
    history_meta = {str(m): {"cutoff": s["cutoff"], "source_matches": s["source_matches"],
                    "team_games": s["team_games"], "candidate_count": len(s["player"]),
                    "seen_candidates": sum(int(v[-1]) for v in s["player"].values()),
                    "shuffle_eligible_candidates": len(s["shuffle"])} for m, s in index.snapshots.items()}
    result = {"passed": True, "max_stratum": max_stratum, "train_targets": total,
              "selected_per_epoch": 34048, "steps_per_epoch": 133, "steps_total": 3192,
              "epochs": [{"epoch": e + 1, "unique_targets_so_far": int(unique[e]),
                          "event_counts": counts[e].tolist()} for e in range(24)],
              "rotation_plan_sha256": sha256(path / "rotation_plan.pt"),
              "offside_audit": index.offside_audit, "history": history_meta}
    write_json(path / "data_audit.json", result)
    return result


@torch.no_grad()
def verify_history_online(seed, mode, device, root=ROOT):
    """Same-batch cached/online equivalence and all four unaffected paths."""
    from .coverage_history_probe import probe_dir, selected_coverage, cache_history
    backbone = load_deployed(selected_coverage(root), seed, device, root)
    residual = HistoryResidual(seed).to(device)
    saved = torch.load(probe_dir(mode, seed, root) / "best_event.pt", map_location="cpu", weights_only=False)
    residual.load_state_dict(saved["residual"]); residual.eval()
    index = MatchHistoryIndex()
    raw = next(iter(loader_for("validation", seed, device, root, workers=0, limit=256)))
    batch = _move_batch_to_device(raw, torch.device(device))
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    base, context, condition, shuffled, _ = forward_details(backbone, batch, index, artifacts)
    cached = cache_history(seed, "validation", device, root)
    if cached["sample_ids"][:len(raw["sample_ids"])] != raw["sample_ids"]:
        raise RuntimeError("Cache/online sample mismatch")
    n = len(raw["sample_ids"])
    for key, value in (("main_context", context.cpu()), ("condition", condition), ("shuffled_condition", shuffled)):
        if (cached[key][:n] - value).abs().max() >= 1e-6:
            raise RuntimeError(f"Cache/online mismatch: {key}")
    conditions = {"condition": condition, "shuffled_condition": shuffled}
    enriched = dict(base)
    enriched["event_logits"] = base["event_logits"] + residual(context, condition_for(conditions, torch.arange(n), mode).to(device))
    if any(not torch.equal(base[k], enriched[k]) for k in ("time_seconds", "position_xy", "team_logits", "player_scores")):
        raise RuntimeError("History changed non-Event task")
    write_json(probe_dir(mode, seed, root) / "online_verification.json", {"passed": True, "samples": n,
               "other_outputs_unchanged": True, "frozen_backbone_hash": tensor_hash(backbone.backbone.state_dict())})


def smoke(root=ROOT, device="cuda:0"):
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("GPU training requires a successful CUDA smoke")
    source = sources()
    target = root / "smoke" / hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:12]
    result_path = root / "verification/smoke.json"
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old["sources"] == source and old["passed"]:
            return old
    seed = SEEDS[0]
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    set_seed(seed)
    model = build_partial_l2_model(artifacts).to(device).eval()
    initial = tensor_hash(model.state_dict())
    calls = {k: 0 for k in ("encoder", "shared", "main", "player")}
    handles = []
    for name, module in (("encoder", model.event_projection), ("shared", model.convolutions[0]),
                         ("main", model.convolutions[1]), ("player", model.player_convolution)):
        def hook(module, args, output, key=name): calls[key] += 1
        handles.append(module.register_forward_hook(hook))
    raw = next(iter(loader_for("validation", seed, device, target, workers=0, limit=32)))
    batch = _move_batch_to_device(raw, torch.device(device))
    with torch.no_grad():
        first = model(batch)
    for h in handles: h.remove()
    assert all(n == 1 for n in calls.values())
    set_seed(seed)
    other = build_partial_l2_model(artifacts).to(device).eval()
    with torch.no_grad():
        second = other(batch)
        error = max(float((first[k] - second[k]).abs().max()) for k in first)
        loss_error = abs(float(fixed_budget_loss(first, batch, artifacts, ("event", "time", "position", "team", "player"))[0]) -
                         float(fixed_budget_loss(second, batch, artifacts, ("event", "time", "position", "team", "player"))[0]))
    assert initial == tensor_hash(other.state_dict()) and error < 1e-6 and loss_error < 1e-6
    del model, other
    results = []
    for mode in ("fixed", "rotate"):
        results.append(train_coverage(mode, seed, device, target, workers=0, smoke=True))
    index = MatchHistoryIndex()
    deployed = load_deployed("original", seed, device, root)
    before = tensor_hash(deployed.backbone.state_dict())
    with torch.no_grad():
        output, context, condition, shuffled, groups = forward_details(deployed, batch, index, artifacts)
        old_targets = batch["targets"]
        batch["targets"] = {k: torch.zeros_like(v) for k, v in old_targets.items()}
        changed, context2, condition2, shuffled2, _ = forward_details(deployed, batch, index, artifacts)
        batch["targets"] = old_targets
    assert torch.equal(condition, condition2) and torch.equal(shuffled, shuffled2)
    assert all(torch.equal(output[k], changed[k]) for k in output)
    cache = {"main_context": context.cpu(), "condition": condition, "shuffled_condition": shuffled,
             "base_event_logits": output["event_logits"].cpu(), "event_true": old_targets["raw_event_10"].cpu()}
    hashes = []
    for mode in ("null", "team", "team_player", "shuffled_player"):
        trained = fit_history(cache, cache, ProbeConfig(seed, mode, device, epochs=1, batch_size=16), target / "heads" / mode, {})
        hashes.append(trained["initial_hash"])
    assert len(set(hashes)) == 1 and before == tensor_hash(deployed.backbone.state_dict())
    result = {"passed": True, "device": device, "sources": source, "initialization_max_error": error,
              "loss_max_error": loss_error, "hgt_calls": calls, "coverage_runs": results,
              "history_targets_do_not_affect_conditions": True, "frozen_backbone_unchanged": True}
    write_json(result_path, result)
    return result
