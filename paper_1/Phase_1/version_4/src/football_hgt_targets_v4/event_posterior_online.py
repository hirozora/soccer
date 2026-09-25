"""One-pass five-task composition; the Event condition precedes the Team prior."""

import json
import time
from pathlib import Path

import torch
from torch import nn

from football_benchmark.protocol import ProtocolArtifacts
from .constants import FEASIBILITY_ARTIFACT
from .event_posterior_integration import ROOT, EventResidual, expected_player_state, load_cache, predict, run_dir
from .five_task_training import FiveTaskTrainingConfig, _loader
from .oracle_dependency import load_roster_team_map
from .oracle_dependency_study import source_checkpoint, cache_path
from .position_head_refit import ROOT as POSITION_ROOT
from .position_head_refit import load_combined_model, sha256, tensor_hash, write_json
from .team_candidate_prior import _anchor_team_raw_ids, centered_log_prior
from .team_candidate_prior_study import lock_path as team_lock_path
from .training import _move_batch_to_device


class IntegratedEventModel(nn.Module):
    def __init__(self, backbone, artifacts, roster, residual=None, mode="base_post"):
        super().__init__()
        self.backbone = backbone
        self.residual = residual
        self.mode, self.artifacts, self.roster = mode, artifacts, roster
        self.backbone.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, batch):
        captured = []
        handle = self.backbone.player_actor_scorer.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0][:, 64:]))
        try:
            with torch.no_grad():
                outputs, contexts = self.backbone.forward_with_contexts(batch)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one Player scorer call")
        graph = batch["graphs"]["f80"]
        ptr = graph["player"].ptr
        raw_scores = outputs["player_scores"]
        if self.residual is not None:
            condition = (torch.zeros_like(contexts["f80"]) if self.mode == "null"
                         else expected_player_state(raw_scores, captured[0], ptr))
            outputs["event_logits"] = outputs["event_logits"] + self.residual(contexts["f80"], condition)
        raw_ids = graph["player"].raw_id.detach().cpu().tolist()
        anchors = _anchor_team_raw_ids(graph, self.artifacts)
        same = torch.zeros(len(raw_ids), dtype=torch.bool)
        valid = torch.zeros_like(same)
        for row, match in enumerate(batch["match_ids"].detach().cpu().tolist()):
            mapping = self.roster[int(match)]
            for index in range(int(ptr[row]), int(ptr[row + 1])):
                team = mapping.get(int(raw_ids[index]))
                if team is not None:
                    valid[index] = True
                    same[index] = team == int(anchors[row])
        prior = centered_log_prior({"team_logits": outputs["team_logits"].detach().cpu(),
            "candidate_ptr": ptr.detach().cpu(), "candidate_same_anchor": same, "candidate_team_valid": valid})
        outputs["player_scores"] = raw_scores + 1.5 * prior.to(raw_scores.device)
        return outputs


def load_integrated_model(seed, checkpoint=None, device="cpu"):
    position_lock_path = POSITION_ROOT / "selection/position_refit_lock.json"
    position_lock = json.loads(position_lock_path.read_text())
    actor_lock = json.loads(team_lock_path().read_text())
    if not position_lock["passed"] or actor_lock["final_lambda"] != 1.5:
        raise RuntimeError("Previously locked Position/Player components required")
    position = position_lock["checkpoints"][str(seed)]
    if sha256(Path(position["path"])) != position["sha256"]:
        raise RuntimeError("Locked Position checkpoint changed")
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        backbone = load_combined_model(seed, Path(position["path"]), device)
    residual, mode = None, "base_post"
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["config"]["seed"] != seed:
            raise RuntimeError("Event checkpoint seed mismatch")
        sources = saved["provenance"]["validation"]["sources"]
        if sources[str(source_checkpoint(seed))] != sha256(source_checkpoint(seed)):
            raise RuntimeError("Event checkpoint belongs to a different backbone")
        residual = EventResidual(seed).to(device)
        residual.load_state_dict(saved["residual"])
        mode = saved["config"]["mode"]
    model = IntegratedEventModel(backbone, ProtocolArtifacts.load(FEASIBILITY_ARTIFACT), load_roster_team_map(), residual, mode)
    return model.to(device).requires_grad_(False).eval()


@torch.no_grad()
def verify_online(seed, root=ROOT, checkpoint=None, device="cpu"):
    cache, provenance = load_cache(seed, "validation", root)
    original = load_integrated_model(seed, device=device)
    candidate = load_integrated_model(seed, checkpoint, device)
    if checkpoint is None:
        candidate.residual = EventResidual(seed).to(device).requires_grad_(False).eval()
    source = torch.load(cache_path(seed, "validation"), map_location="cpu", weights_only=False, mmap=True)
    before = tensor_hash(original.backbone.state_dict())
    if before != tensor_hash(candidate.backbone.state_dict()):
        raise RuntimeError("Non-residual parameters differ")
    config = FiveTaskTrainingConfig(mode="five_f80", output_dir=root, seed=seed, device=device,
        num_workers=0, max_validation_samples=32, batch_size=16)
    loader = _loader("validation", config, ProtocolArtifacts.load(FEASIBILITY_ARTIFACT), False)
    row_map = {s: i for i, s in enumerate(cache["sample_ids"])}
    errors = {key: 0.0 for key in ("time_seconds", "log_delta", "position_xy", "team_logits", "player_scores")}
    counts = {key: 0 for key in ("encoder", "shared_l1", "main_l2", "player_l2")}
    def hook(key):
        def record(*args):
            counts[key] += 1
        return record
    b = candidate.backbone
    handles = [b.event_projection.register_forward_hook(hook("encoder")),
        b.convolutions[0].register_forward_hook(hook("shared_l1")), b.convolutions[1].register_forward_hook(hook("main_l2")),
        b.player_convolution.register_forward_hook(hook("player_l2"))]
    cache_error, probability_error, zero_error, batches, ids = 0.0, 0.0, 0.0, 0, []
    latency = {"original_seconds": [], "candidate_seconds": []}
    for raw in loader:
        rows = torch.tensor([row_map[s] for s in raw["sample_ids"]])
        if (not torch.equal(raw["targets"]["raw_event_10"], cache["event_true"][rows])
                or not torch.equal(raw["targets"]["player_mask"], cache["player_mask"][rows])
                or not torch.equal(raw["current_event_indices"], cache["current_event_indices"][rows])):
            raise RuntimeError("Physical/cached samples differ")
        batch = _move_batch_to_device(raw, torch.device(device))
        graph = raw["graphs"]["f80"]
        for local_row, cached_row in enumerate(rows.tolist()):
            a, b = map(int, graph["player"].ptr[local_row:local_row + 2])
            c, d = map(int, source["candidate_ptr"][cached_row:cached_row + 2])
            if not torch.equal(graph["player"].raw_id[a:b].cpu(), source["candidate_raw"][c:d]):
                raise RuntimeError("Physical candidate IDs/order differ from Oracle cache")
        # Native ATen avoids this host's oneDNN GELU failures on irregular graph sizes.
        with torch.backends.mkldnn.flags(enabled=False):
            for name, model in (("original_seconds", original), ("candidate_seconds", candidate)):
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                started = time.perf_counter()
                output = model(batch)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                latency[name].append(time.perf_counter() - started)
                if name == "original_seconds":
                    old = output
                else:
                    new = output
        for key in errors:
            errors[key] = max(errors[key], float((old[key] - new[key]).abs().max()))
        if checkpoint is None:
            zero_error = max(zero_error, float((old["event_logits"] - new["event_logits"]).abs().max()))
        if checkpoint:
            subset = {key: value[rows] for key, value in cache.items() if isinstance(value, torch.Tensor)}
            cached = predict(candidate.residual, subset, candidate.mode)
        else:
            cached = cache["base_event_logits"][rows]
        cache_error = max(cache_error, float((new["event_logits"].cpu() - cached).abs().max()))
        probability_error = max(probability_error, float((new["event_logits"].cpu().softmax(-1) - cached.softmax(-1)).abs().max()))
        if not torch.equal(new["event_logits"].cpu().argmax(-1), cached.argmax(-1)):
            raise RuntimeError("Online/cached Event argmax differs")
        ids.extend(raw["sample_ids"])
        batches += 1
    for handle in handles:
        handle.remove()
    if (not ids or max([zero_error, probability_error, *errors.values()]) >= 1e-6 or any(v != batches for v in counts.values())
            or before != tensor_hash(candidate.backbone.state_dict()) or any(p.grad is not None for p in candidate.parameters())):
        raise RuntimeError(f"Online invariance failed: {cache_error}, {errors}, {counts}")
    result = {"passed": True, "seed": seed, "sample_ids": ids, "unaffected_output_max_errors": errors,
        "online_cache_event_logit_max_error": cache_error, "online_cache_event_probability_max_error": probability_error,
        "zero_initialization_same_backend_logit_error": zero_error if checkpoint is None else None,
        "layer_calls": counts, "backbone_hash": before,
        "head_sha256": sha256(checkpoint) if checkpoint else None, "provenance": provenance,
        "timing": {**latency, "measurement": "32 validation samples; CPU native ATen, includes posterior and fixed TC prior; descriptive only"}}
    path = checkpoint.parent / "online_verification.json" if checkpoint else root / "verification" / f"seed{seed}.json"
    write_json(path, result)
    return result


@torch.no_grad()
def benchmark_online(seed, root=ROOT, device="cpu", repetitions=7):
    """Warmed model-only timing on a fixed validation graph, never on test."""
    config = FiveTaskTrainingConfig(mode="five_f80", output_dir=root, seed=seed, device=device,
        num_workers=0, max_validation_samples=16, batch_size=16)
    started = time.perf_counter()
    raw = next(iter(_loader("validation", config, ProtocolArtifacts.load(FEASIBILITY_ARTIFACT), False)))
    collate_seconds = time.perf_counter() - started
    started = time.perf_counter()
    batch = _move_batch_to_device(raw, torch.device(device))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    transfer_seconds = time.perf_counter() - started
    models = {mode: load_integrated_model(seed, None if mode == "original" else run_dir(seed, mode, root) / "best_event.pt", device)
              for mode in ("original", "null", "base_post")}
    times = {mode: [] for mode in models}
    with torch.backends.mkldnn.flags(enabled=False):
        for model in models.values():
            for _ in range(2):
                model(batch)
        for iteration in range(repetitions):
            order = tuple(models) if iteration % 2 == 0 else tuple(reversed(models))
            for mode in order:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                started = time.perf_counter()
                models[mode](batch)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                times[mode].append(time.perf_counter() - started)
    baseline = sorted(times["original"])[repetitions // 2]
    result = []
    for mode, values in times.items():
        median = sorted(values)[repetitions // 2]
        result.append({"seed": seed, "mode": mode, "device": device, "samples": len(raw["sample_ids"]),
            "repetitions": repetitions, "warmups": 2, "forward_median_seconds": median,
            "relative_forward_cost": median / baseline, "samples_per_second": len(raw["sample_ids"]) / median,
            "parameters": sum(p.numel() for p in models[mode].parameters()), "hgt_layer_calls_per_forward": 3,
            "loader_setup_and_first_collate_seconds": collate_seconds, "transfer_seconds": transfer_seconds,
            "measurement": "fixed validation batch, model-only, includes fixed Position/TC composition; not full-dataset latency"})
    write_json(root / "efficiency" / f"seed{seed}.json", result)
    return result
