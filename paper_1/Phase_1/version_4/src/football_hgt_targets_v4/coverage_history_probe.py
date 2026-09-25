"""Frozen, full-population Event probes for past-match priors."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from football_benchmark.protocol import ProtocolArtifacts
from .constants import FEASIBILITY_ARTIFACT
from .coverage_history_data import MatchHistoryIndex, condition39, MATCHES
from .coverage_history_training import (ROOT, SEEDS, COUNTS, require_test, loader_for, load_deployed,
    frame_path, sources, evaluate_deployed)
from .event_posterior_integration import metrics, selection_value, eligible, paired_bootstrap
from .position_head_refit import capture_rng, restore_rng, save_checkpoint, sha256, tensor_hash
from .spatiotemporal_training import write_json
from .team_candidate_prior import _anchor_team_raw_ids
from .training import _move_batch_to_device, set_seed


MODES = ("null", "team", "team_player", "shuffled_player")


def probe_dir(mode, seed, root=ROOT):
    return root / "history" / mode / f"seed{seed}"


def selected_coverage(root=ROOT):
    path = root / "selection/coverage_lock.json"
    lock = json.loads(path.read_text())
    for p, digest in lock["checkpoints"].items():
        if sha256(Path(p)) != digest:
            raise RuntimeError("Stage 1 checkpoint changed")
    return lock["selected"]


class HistoryResidual(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed + 7919)
            self.layers = nn.Sequential(nn.Linear(103, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 10))
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    @torch.backends.mkldnn.flags(enabled=False)
    def forward(self, context, condition):
        return self.layers(torch.cat([context.detach(), condition.detach()], -1))


def condition_for(cache, rows, mode):
    if mode == "null":
        return torch.zeros((len(rows), 39), dtype=cache["condition"].dtype)
    if mode == "shuffled_player":
        return cache["shuffled_condition"][rows]
    condition = cache["condition"][rows].clone()
    if mode == "team":
        condition[:, 26:] = 0
    elif mode != "team_player":
        raise ValueError(mode)
    return condition


@torch.no_grad()
def forward_details(model, batch, index, artifacts):
    contexts = []
    hook = model.backbone.context_projection.register_forward_hook(lambda module, args, output: contexts.append(output))
    try:
        output = model(batch)
    finally:
        hook.remove()
    if len(contexts) != 1:
        raise RuntimeError("Expected one Main F80 context computation")
    graph = batch["graphs"]["f80"]
    condition, shuffled, groups = condition39(index, batch["match_ids"].cpu().tolist(),
        _anchor_team_raw_ids(graph, artifacts).cpu().tolist(), graph["player"].raw_id.cpu().tolist(),
        graph["player"].ptr.cpu(), output["player_scores"])
    return output, contexts[0].detach(), condition, shuffled, groups


@torch.no_grad()
def cache_history(seed, split, device, root=ROOT):
    if split == "test":
        require_test(root)
    mode = selected_coverage(root)
    path = root / "history_cache" / f"seed{seed}" / f"{split}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = {"coverage_lock": sha256(root / "selection/coverage_lock.json"),
                  "matches": sha256(MATCHES), "source": sources()}
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if cache["provenance"] != provenance:
            raise RuntimeError("Stale historical context cache")
        return cache
    index = MatchHistoryIndex(include_test=split == "test", root=root)
    model = load_deployed(mode, seed, device, root)
    before = tensor_hash(model.backbone.state_dict())
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    chunks, ids = {}, []
    for raw in loader_for(split, seed, device, root):
        batch = _move_batch_to_device(raw, torch.device(device))
        output, context, condition, shuffled, groups = forward_details(model, batch, index, artifacts)
        t, graph = batch["targets"], batch["graphs"]["f80"]
        anchors = graph["event"].ptr[1:] - 1
        values = {"main_context": context, "base_event_logits": output["event_logits"],
                  "condition": condition, "shuffled_condition": shuffled, "history_groups": groups,
                  "event_true": t["raw_event_10"], "player_mask": t["player_mask"], "zone_true": t["zone_20"],
                  "match_ids": batch["match_ids"], "current_event_indices": batch["current_event_indices"],
                  "event_role": graph["event"].event_role_index[anchors],
                  "control_state": graph["event"].control_state_after_index[anchors],
                  "switch_confirmed": graph["event"].switch_confirmed[anchors]}
        for k, value in values.items():
            chunks.setdefault(k, []).append(value.detach().cpu())
        ids.extend(raw["sample_ids"])
    if len(ids) != COUNTS[split] or len(set(ids)) != len(ids):
        raise RuntimeError("Historical cache population mismatch")
    if before != tensor_hash(model.backbone.state_dict()) or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen backbone changed")
    cache = {k: torch.cat(v) for k, v in chunks.items()}
    cache.update(sample_ids=ids, provenance=provenance, backbone_hash=before)
    save_checkpoint(path, cache)
    write_json(path.with_suffix(".history_sources.json"), {
        "offside_audit": index.offside_audit, "target_split": split,
        "matches": {m: {"cutoff": s["cutoff"], "source_matches": s["source_matches"]}
                    for m, s in index.snapshots.items() if m in set(cache["match_ids"].tolist())}})
    return cache


@torch.no_grad()
def predict(model, cache, mode):
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    for rows in torch.arange(len(cache["main_context"])).split(1024):
        chunks.append(cache["base_event_logits"][rows] + model(cache["main_context"][rows].to(device),
                      condition_for(cache, rows, mode).to(device)).cpu())
    return torch.cat(chunks)


@dataclass(frozen=True)
class ProbeConfig:
    seed: int
    mode: str
    device: str = "cpu"
    epochs: int = 30
    patience: int = 5
    batch_size: int = 1024


@torch.backends.mkldnn.flags(enabled=False)
def fit_history(train, validation, config, output, provenance, resume_from=None, stop_after=None):
    if config.mode not in MODES:
        raise ValueError(config.mode)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = HistoryResidual(config.seed).to(config.device)
    initial = tensor_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    original = metrics(validation["base_event_logits"], validation["event_true"])
    epoch, stale, elapsed = 0, 0, 0.
    history = [{"epoch": 0, "validation": original, "eligible": True}]
    best = {"epoch": 0, "selection_value": selection_value(original, 0),
            "residual": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    if resume_from:
        saved = torch.load(resume_from, map_location="cpu", weights_only=False)
        if saved["config"] != asdict(config) or saved["provenance"] != provenance:
            raise RuntimeError("History resume provenance mismatch")
        model.load_state_dict(saved["residual"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, epoch, stale, elapsed = (saved[k] for k in ("history", "best", "epoch", "stale", "elapsed"))
        restore_rng(saved["rng"], generator)
    elif not torch.equal(predict(model, validation, config.mode), validation["base_event_logits"]):
        raise RuntimeError("History zero initialization changed predictions")
    start = time.monotonic()

    def persist():
        common = {"config": asdict(config), "provenance": provenance, "initial_hash": initial}
        save_checkpoint(output / "best_event.pt", {**common, **best})
        save_checkpoint(output / "last.pt", {**common, "residual": model.state_dict(), "optimizer": optimizer.state_dict(),
            "history": history, "best": best, "epoch": epoch, "stale": stale,
            "rng": capture_rng(generator), "elapsed": elapsed + time.monotonic() - start})
        write_json(output / "history.json", history)

    persist()
    budget = config.epochs if stop_after is None else min(stop_after, config.epochs)
    while epoch < budget and stale < config.patience:
        epoch += 1
        model.train()
        order = torch.randperm(len(train["event_true"]), generator=generator)
        total = 0.
        for rows in order.split(config.batch_size):
            logits = train["base_event_logits"][rows].to(config.device) + model(
                train["main_context"][rows].to(config.device), condition_for(train, rows, config.mode).to(config.device))
            loss = F.cross_entropy(logits, train["event_true"][rows].to(config.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        values = metrics(predict(model, validation, config.mode), validation["event_true"])
        if not np.isfinite(values["ce"]):
            raise RuntimeError("Nonfinite historical probe loss")
        allowed = eligible(values, original)
        history.append({"epoch": epoch, "validation": values, "eligible": allowed,
                        "train_loss": total / len(order), "sample_order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest()})
        score = selection_value(values, epoch)
        if allowed and score < tuple(best["selection_value"]):
            best = {"epoch": epoch, "selection_value": score,
                    "residual": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        persist()
        print(json.dumps({"mode": config.mode, "seed": config.seed, "epoch": epoch,
                          "macro_f1": values["macro_f1"], "best_epoch": best["epoch"]}), flush=True)
    result = {"config": asdict(config), "provenance": provenance, "best_epoch": best["epoch"],
              "initial_hash": initial, "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "completed": epoch == config.epochs or stale >= config.patience, "epochs": epoch,
              "validation": history[best["epoch"]]["validation"], "original": original}
    write_json(output / "result.json", result)
    return result


def event_frame(cache, logits, seed):
    frame = pd.DataFrame({"seed": seed, "sample_id": cache["sample_ids"], "match_id": cache["match_ids"].numpy(),
        "current_event_index": cache["current_event_indices"].numpy(), "event_true": cache["event_true"].numpy(),
        "event_pred": logits.argmax(-1).numpy(), "player_mask": cache["player_mask"].numpy()})
    for i in range(10):
        frame[f"event_probability_{i}"] = logits.softmax(-1)[:, i].numpy()
    frame["team_history_games"] = cache["history_groups"][:, 0].numpy()
    frame["player_history_seen_mass"] = cache["history_groups"][:, 1].numpy()
    frame["shuffle_candidate_coverage"] = cache["history_groups"][:, 2].numpy()
    return frame


def train_history(mode, seed, root=ROOT, device="cpu", resume_from=None):
    if (root / "selection/history_lock.json").exists():
        raise RuntimeError("History model selection already locked")
    path = root / "history_cache" / f"seed{seed}"
    train, validation = [torch.load(path / f"{s}.pt", map_location="cpu", weights_only=False, mmap=True)
                         for s in ("train", "validation")]
    if len(train["event_true"]) != COUNTS["train"] or len(validation["event_true"]) != COUNTS["validation"]:
        raise RuntimeError("History probes require full train/validation")
    provenance = {s: sha256(path / f"{s}.pt") for s in ("train", "validation")}
    provenance["code"] = sha256(Path(__file__))
    out = probe_dir(mode, seed, root)
    if (out / "result.json").exists():
        old = json.loads((out / "result.json").read_text())
        if old["provenance"] != provenance:
            raise RuntimeError("History run inputs changed")
        if old["completed"] and (out / "validation.parquet").exists():
            return old
    resume = resume_from or (out / "last.pt" if (out / "last.pt").exists() else None)
    result = fit_history(train, validation, ProbeConfig(seed, mode, device), out, provenance, resume)
    model = HistoryResidual(seed).to(device)
    model.load_state_dict(torch.load(out / "best_event.pt", map_location="cpu", weights_only=False)["residual"])
    event_frame(validation, predict(model, validation, mode), seed).to_parquet(out / "validation.parquet", index=False)
    return result


def choose_history(comparisons, scores):
    eligible_modes = []
    for mode in ("team", "team_player"):
        if not all(comparisons[f"{mode}-{ref}"]["effective"] for ref in ("null", "original")):
            continue
        if mode == "team_player":
            shuffled = comparisons["team_player-shuffled_player"]
            if not comparisons["team_player-team"]["effective"] or shuffled["ci95"][0] <= 0 or shuffled["improving_seeds"] < 2:
                continue
        eligible_modes.append(mode)
    if not eligible_modes:
        return "original", []
    maximum = max(scores[m] for m in eligible_modes)
    return next(m for m in ("team", "team_player") if m in eligible_modes and scores[m] >= maximum - .001), eligible_modes


def select_history(root=ROOT):
    path = root / "selection/history_lock.json"
    if path.exists():
        return json.loads(path.read_text())
    frames = {}
    scores = {}
    for mode in MODES:
        pieces, values = [], []
        for seed in SEEDS:
            out = probe_dir(mode, seed, root)
            result = json.loads((out / "result.json").read_text())
            if not result["completed"]:
                raise RuntimeError("History training incomplete")
            pieces.append(pd.read_parquet(out / "validation.parquet"))
            values.append(result["validation"]["macro_f1"])
        frames[mode] = pd.concat(pieces, ignore_index=True).sort_values(["seed", "sample_id"]).reset_index(drop=True)
        scores[mode] = float(np.mean(values))
    originals = []
    for seed in SEEDS:
        cache = torch.load(root / "history_cache" / f"seed{seed}/validation.pt", map_location="cpu", weights_only=False, mmap=True)
        originals.append(event_frame(cache, cache["base_event_logits"], seed))
    frames["original"] = pd.concat(originals, ignore_index=True).sort_values(["seed", "sample_id"]).reset_index(drop=True)
    pairs = [("null", "original"), ("team", "null"), ("team", "original"),
             ("team_player", "null"), ("team_player", "original"), ("team_player", "team"), ("team_player", "shuffled_player")]
    comparisons = {f"{a}-{b}": paired_bootstrap(frames[b], frames[a]) for a, b in pairs}
    selected, eligible_modes = choose_history(comparisons, scores)
    paths = [probe_dir(m, s, root) / "best_event.pt" for m in MODES for s in SEEDS]
    lock = {"selected": selected, "eligible": eligible_modes, "comparisons": comparisons, "scores": scores,
            "coverage_lock_sha256": sha256(root / "selection/coverage_lock.json"),
            "checkpoints": {str(p): sha256(p) for p in paths}, "test_accessed": False}
    write_json(path, lock)
    return lock


def evaluate_history_test(seed, device, root=ROOT):
    _, lock = require_test(root)
    if lock["selected"] == "original":
        return
    cache = cache_history(seed, "test", device, root)
    modes = ("null", "team") if lock["selected"] == "team" else MODES
    out = root / "history_test" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    event_frame(cache, cache["base_event_logits"], seed).to_parquet(out / "original.parquet", index=False)
    for mode in modes:
        model = HistoryResidual(seed)
        model.load_state_dict(torch.load(probe_dir(mode, seed, root) / "best_event.pt", map_location="cpu", weights_only=False)["residual"])
        event_frame(cache, predict(model, cache, mode), seed).to_parquet(out / f"{mode}.parquet", index=False)
