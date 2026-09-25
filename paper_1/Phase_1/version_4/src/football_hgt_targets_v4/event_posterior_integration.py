"""Frozen, zero-initialized Player-posterior residual integration for Event."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .constants import CONFIRMATION_SEEDS, EXPERIMENT_ROOT, SAMPLE_PLAN
from .oracle_dependency_study import cache_path, source_checkpoint
from .player_posterior_study import condition_cache_path
from .position_head_refit import capture_rng, restore_rng, save_checkpoint, sha256, tensor_hash, write_json

ROOT = EXPERIMENT_ROOT / "event_posterior_integration_v1"
MODES = ("null", "base_post")
COUNTS = {"train": 34048, "validation": 7296, "test": 96854}


def run_dir(seed, mode, root=ROOT):
    if mode not in MODES:
        raise ValueError(mode)
    return root / "training" / mode / f"seed{seed}"


def require_test_lock(root=ROOT):
    path = root / "selection/event_posterior_lock.json"
    if not path.exists():
        raise RuntimeError("Event validation lock required before any test read")
    lock = json.loads(path.read_text())
    if not lock["passed"]:
        raise RuntimeError("Validation failed; test access is prohibited")
    for item in lock["checkpoints"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("Locked Event checkpoint changed")
    return lock


class EventResidual(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 10))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        assert sum(p.numel() for p in self.parameters()) == 8906

    def forward(self, context, condition):
        values = torch.cat((context.detach(), condition.detach()), dim=-1)
        if values.device.type == "cpu":
            # Same exact GELU, without this host's failing oneDNN allocation.
            with torch.backends.mkldnn.flags(enabled=False):
                return self.network(values)
        return self.network(values)


def expected_player_state(scores, states, ptr):
    """No targets, masks, or Team information are accepted by this interface."""
    if (ptr.ndim != 1 or int(ptr[0]) != 0 or int(ptr[-1]) != len(scores)
            or states.shape != (len(scores), 64) or bool(((ptr[1:] - ptr[:-1]) <= 0).any())):
        raise ValueError("Invalid candidate segmentation")
    outputs = []
    for start, stop in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
        probabilities = scores[start:stop].float().softmax(dim=0)
        outputs.append((probabilities[:, None] * states[start:stop].float()).sum(dim=0))
    return torch.stack(outputs).detach()


def load_cache(seed, split, root=ROOT):
    if split not in COUNTS or seed not in CONFIRMATION_SEEDS:
        raise ValueError("Unsupported split or seed")
    if split == "test":
        require_test_lock(root)
    path, stage_path, checkpoint = cache_path(seed, split), condition_cache_path(seed, split), source_checkpoint(seed)
    raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    stage = torch.load(stage_path, map_location="cpu", weights_only=False, mmap=True)
    source_hash = sha256(checkpoint)
    if (raw["source_checkpoint_sha256"] != source_hash or Path(raw["source_checkpoint"]).resolve() != checkpoint.resolve()
            or raw["seed"] != seed or raw["split"] != split or stage["seed"] != seed or stage["split"] != split
            or Path(stage["source_model_checkpoint"]).resolve() != checkpoint.resolve()
            or Path(stage["source_oracle_cache"]).resolve() not in (path.resolve(), stage_path.resolve())):
        raise RuntimeError("Cache provenance mismatch")
    ids = [f"{int(m)}:{int(i)}" for m, i in zip(raw["match_ids"], raw["current_event_indices"])]
    if ids != raw["sample_ids"] or ids != stage["sample_ids"] or len(ids) != COUNTS[split] or len(set(ids)) != len(ids):
        raise RuntimeError("Sample IDs/counts differ")
    if split != "test":
        plan = json.loads(SAMPLE_PLAN.read_text())["selections"][split]
        planned = {f"{m}:{a}" for m, anchors in plan.items() for a in anchors}
        if set(ids) != planned:
            raise RuntimeError("Fixed sample plan mismatch")
    for key in ("main_context", "event_true", "player_mask", "match_ids", "current_event_indices", "zone_true"):
        if not torch.equal(raw[key], stage[key]):
            raise RuntimeError(f"Stage B/Oracle cache mismatch: {key}")
    condition = expected_player_state(raw["candidate_scores"], raw["candidate_states"], raw["candidate_ptr"])
    condition_error = float((condition - stage["base_post_condition"]).abs().max())
    if condition_error >= 1e-6:
        raise RuntimeError(f"Base posterior cache mismatch: {condition_error}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    if state["event_head.weight"].shape != (10, 64) or state["event_head.bias"].shape != (10,):
        raise RuntimeError("Expected original Linear(64,10) Event Head")
    initial = F.linear(raw["main_context"], state["event_head.weight"], state["event_head.bias"])
    error = float((initial - raw["base_event_logits"]).abs().max())
    probability_error = float((initial.softmax(-1) - raw["base_event_logits"].softmax(-1)).abs().max())
    # GPU-produced float32 logits can differ by a few ulps from CPU GEMM.
    # Epoch-zero residual equality remains bitwise; source identity is hashed.
    if (not torch.allclose(initial, raw["base_event_logits"], rtol=1e-6, atol=1e-6)
            or probability_error >= 1e-6 or not torch.equal(initial.argmax(-1), raw["base_event_logits"].argmax(-1))):
        raise RuntimeError(f"Original Event/cache mismatch: logits={error}, probability={probability_error}")
    keys = ("main_context", "base_event_logits", "event_true", "player_mask", "match_ids", "current_event_indices", "zone_true")
    cache = {key: raw[key].detach().clone() for key in keys}
    cache.update(sample_ids=ids, condition=condition)
    for key in ("event_role", "control_state", "switch_confirmed"):
        cache[key] = stage[key].detach().clone()
    if (cache["main_context"].shape != (len(ids), 64) or not torch.isfinite(condition).all()
            or not torch.isfinite(cache["main_context"]).all() or not torch.isfinite(initial).all()
            or bool(((cache["event_true"] < 0) | (cache["event_true"] >= 10)).any())):
        raise RuntimeError("Invalid Event/context values")
    sources = {str(p): sha256(p) for p in (path, stage_path, checkpoint, SAMPLE_PLAN)}
    return cache, {"sources": sources, "samples": len(ids), "unknown_player_samples": int((~raw["player_mask"]).sum()),
                   "stage_b_declared_source_oracle_cache": stage["source_oracle_cache"],
                   "legacy_self_reference_metadata": Path(stage["source_oracle_cache"]).resolve() == stage_path.resolve(),
                   "verified_oracle_cache": str(path),
                   "base_condition_max_error": condition_error, "epoch0_max_error": 0.0,
                   "cross_backend_original_logit_error": error, "cross_backend_original_probability_error": probability_error,
                   "candidate_ptr_sha256": tensor_hash({"ptr": raw["candidate_ptr"], "raw": raw["candidate_raw"]})}


def condition_for(cache, rows, mode):
    if mode == "null":
        return torch.zeros((len(rows), 64), dtype=torch.float32)
    if mode == "base_post":
        return cache["condition"][rows].detach()
    raise ValueError(mode)


@torch.no_grad()
def predict(model, cache, mode, batch_size=1024):
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    for rows in torch.arange(len(cache["main_context"])).split(batch_size):
        residual = model(cache["main_context"][rows].to(device), condition_for(cache, rows, mode).to(device))
        chunks.append(cache["base_event_logits"][rows] + residual.cpu())
    return torch.cat(chunks)


def confusion(target, prediction):
    return np.bincount(np.asarray(target, dtype=np.int64) * 10 + np.asarray(prediction, dtype=np.int64), minlength=100).reshape(10, 10)


def cm_scores(cm):
    diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(axis=-1), cm.sum(axis=-2)
    f1 = np.divide(2.0 * diagonal, support + predicted, out=np.zeros_like(diagonal, dtype=float), where=(support + predicted) > 0)
    return f1.mean(axis=-1), diagonal.sum(axis=-1) / cm.sum(axis=(-1, -2))


def metrics(logits, target):
    cm = confusion(target.numpy(), logits.argmax(-1).numpy())
    f1, accuracy = cm_scores(cm)
    return {"samples": len(target), "macro_f1": float(f1), "accuracy": float(accuracy),
            "ce": float(F.cross_entropy(logits, target)), "confusion": cm.tolist()}


def selection_value(values, epoch):
    return -values["macro_f1"], -values["accuracy"], values["ce"], epoch


def eligible(values, original):
    return values["accuracy"] >= original["accuracy"] - 0.01 - 1e-12


@dataclass(frozen=True)
class EventConfig:
    seed: int
    mode: str
    device: str = "cpu"
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 1e-4


@torch.backends.mkldnn.flags(enabled=False)
def fit_cached(train, validation, config, output, provenance, resume_from=None, stop_after_epoch=None):
    if config.mode not in MODES or min(config.max_epochs, config.patience, config.batch_size) < 1:
        raise ValueError("Invalid experiment configuration")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = EventResidual(config.seed).to(config.device)
    initial_hash = tensor_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    original = metrics(validation["base_event_logits"], validation["event_true"])
    if resume_from:
        saved = torch.load(resume_from, map_location="cpu", weights_only=False)
        if saved["config"] != asdict(config) or saved["provenance"] != provenance:
            raise RuntimeError("Resume configuration or input hashes changed")
        model.load_state_dict(saved["residual"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, stale, epoch = saved["history"], saved["best"], saved["stale"], saved["epoch"]
        elapsed = saved["elapsed_seconds"]
        restore_rng(saved["rng"], generator)
    else:
        zero = predict(model, validation, config.mode)
        if float((zero - validation["base_event_logits"]).abs().max()) >= 1e-6:
            raise RuntimeError("Zero initialization changed original logits")
        history = [{"epoch": 0, "validation": original, "eligible": True, "train_loss": None, "sample_order_sha256": None}]
        best = {"epoch": 0, "selection_value": selection_value(original, 0),
                "residual": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        stale, epoch, elapsed = 0, 0, 0.0
    started = time.monotonic()

    def persist():
        common = {"config": asdict(config), "provenance": provenance, "initial_hash": initial_hash}
        save_checkpoint(output / "best_event.pt", {**common, **best})
        save_checkpoint(output / "last.pt", {**common, "residual": model.state_dict(), "optimizer": optimizer.state_dict(),
            "history": history, "best": best, "stale": stale, "epoch": epoch,
            "rng": capture_rng(generator), "elapsed_seconds": elapsed + time.monotonic() - started})
        write_json(output / "history.json", history)

    persist()
    budget = config.max_epochs if stop_after_epoch is None else min(config.max_epochs, stop_after_epoch)
    while epoch < budget and stale < config.patience:
        epoch += 1
        model.train()
        order = torch.randperm(len(train["main_context"]), generator=generator)
        loss_sum = 0.0
        for rows in order.split(config.batch_size):
            logits = train["base_event_logits"][rows].to(config.device) + model(
                train["main_context"][rows].to(config.device), condition_for(train, rows, config.mode).to(config.device))
            loss = F.cross_entropy(logits, train["event_true"][rows].to(config.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(rows)
        values = metrics(predict(model, validation, config.mode), validation["event_true"])
        allowed = eligible(values, original)
        history.append({"epoch": epoch, "validation": values, "eligible": allowed,
            "train_loss": loss_sum / len(order), "sample_order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest()})
        score = selection_value(values, epoch)
        if allowed and score < tuple(best["selection_value"]):
            best = {"epoch": epoch, "selection_value": score,
                    "residual": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        persist()
    result = {"config": asdict(config), "provenance": provenance, "initial_hash": initial_hash,
        "best_epoch": best["epoch"], "completed_epoch": epoch, "stopped_early": stale >= config.patience,
        "completed": epoch == config.max_epochs or stale >= config.patience,
        "validation": history[best["epoch"]]["validation"], "original": original,
        "trainable_parameters": 8906, "elapsed_seconds": elapsed + time.monotonic() - started}
    write_json(output / "result.json", result)
    return result


def prediction_frame(cache, logits, seed):
    result = pd.DataFrame({"seed": seed, "sample_id": cache["sample_ids"], "match_id": cache["match_ids"].numpy(),
        "current_event_index": cache["current_event_indices"].numpy(), "event_true": cache["event_true"].numpy(),
        "event_pred": logits.argmax(-1).numpy(), "player_mask": cache["player_mask"].numpy(),
        "zone_true": cache["zone_true"].numpy(), "event_role": cache["event_role"].numpy(),
        "control_state": cache["control_state"].numpy(), "switch_confirmed": cache["switch_confirmed"].numpy()})
    for i in range(10):
        result[f"logit_{i}"] = logits[:, i].numpy()
    return result


def train_seed(seed, root=ROOT, device="cpu", mode=None, resume_from=None, smoke=False):
    if (root / "selection/event_posterior_lock.json").exists():
        raise RuntimeError("Locked experiment cannot be retrained")
    train, train_sources = load_cache(seed, "train", root)
    validation, val_sources = load_cache(seed, "validation", root)
    provenance = {"train": train_sources, "validation": val_sources, "source_sha256": sha256(Path(__file__))}
    for selected_mode in ((mode,) if mode else MODES):
        output = root / "smoke" / provenance["source_sha256"][:12] / selected_mode / f"seed{seed}" if smoke else run_dir(seed, selected_mode, root)
        resume = resume_from or (output / "last.pt" if (output / "last.pt").exists() else None)
        result = fit_cached(train, validation, EventConfig(seed, selected_mode, device), output, provenance, resume,
                            stop_after_epoch=1 if smoke else None)
        saved = torch.load(output / "best_event.pt", map_location="cpu", weights_only=False)
        model = EventResidual(seed).to(device)
        model.load_state_dict(saved["residual"])
        for name, logits in (("original", validation["base_event_logits"]),
                             (selected_mode, predict(model, validation, selected_mode))):
            prediction_frame(validation, logits, seed).to_parquet(output / f"validation_{name}.parquet", index=False)
        print(f"{selected_mode} seed={seed} best={result['best_epoch']} epochs={result['completed_epoch']} F1={result['validation']['macro_f1']:.6f}", flush=True)


def paired_bootstrap(reference, candidate, replicates=10000, selector_seed=20260913):
    keys = ["seed", "sample_id", "match_id", "current_event_index", "event_true", "player_mask"]
    if not reference[keys].equals(candidate[keys]) or reference.duplicated(["seed", "sample_id"]).any():
        raise RuntimeError("Paired samples/targets mismatch")
    seeds, matches = sorted(reference.seed.unique()), np.sort(reference.match_id.unique())
    first = reference.loc[reference.seed == seeds[0], keys[1:]].reset_index(drop=True)
    for seed in seeds[1:]:
        if not first.equals(reference.loc[reference.seed == seed, keys[1:]].reset_index(drop=True)):
            raise RuntimeError("Cross-seed populations differ")
    sampled = np.random.default_rng(selector_seed).integers(len(matches), size=(replicates, len(matches)))
    draws = np.array([np.bincount(row, minlength=len(matches)) for row in sampled])
    per_seed, boot_f1, boot_acc = [], [], []
    for seed in seeds:
        ref = reference[reference.seed == seed]
        cand = candidate[candidate.seed == seed]
        pair = []
        for frame in (ref, cand):
            matrices = np.stack([confusion(group.event_true, group.event_pred) for _, group in frame.groupby("match_id", sort=True)])
            pair.append((cm_scores(matrices.sum(0)), cm_scores((draws @ matrices.reshape(len(matches), 100)).reshape(-1, 10, 10))))
        f1 = float(pair[1][0][0] - pair[0][0][0])
        acc = float(pair[1][0][1] - pair[0][0][1])
        per_seed.append({"seed": int(seed), "macro_f1_gain": f1, "accuracy_gain": acc})
        boot_f1.append(pair[1][1][0] - pair[0][1][0])
        boot_acc.append(pair[1][1][1] - pair[0][1][1])
    gain = float(np.mean([p["macro_f1_gain"] for p in per_seed]))
    acc = float(np.mean([p["accuracy_gain"] for p in per_seed]))
    ci = np.quantile(np.mean(boot_f1, axis=0), [0.025, 0.975]).tolist()
    improving = sum(p["macro_f1_gain"] > 0 for p in per_seed)
    return {"macro_f1_gain": gain, "ci95": ci, "accuracy_gain": acc,
        "accuracy_ci95": np.quantile(np.mean(boot_acc, axis=0), [0.025, 0.975]).tolist(),
        "improving_seeds": improving, "seed_differences": per_seed, "replicates": replicates,
        "resampling_unit": "match", "draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
        "effective": bool(gain >= 0.005 and improving >= 2 and ci[0] > 0 and acc >= -0.01)}


def read_predictions(split, mode, root=ROOT):
    if split == "test":
        require_test_lock(root)
    frames = []
    for seed in CONFIRMATION_SEEDS:
        output = run_dir(seed, "null" if mode == "original" else mode, root) if split == "validation" else root / "test" / f"seed{seed}"
        frames.append(pd.read_parquet(output / f"{split}_{mode}.parquet"))
    return pd.concat(frames, ignore_index=True)


def comparisons(split, root=ROOT):
    frames = {mode: read_predictions(split, mode, root) for mode in ("original", *MODES)}
    return {f"{new}_minus_{old}": paired_bootstrap(frames[old], frames[new])
            for new, old in (("null", "original"), ("base_post", "null"), ("base_post", "original"))}


def select(root=ROOT):
    path = root / "selection/event_posterior_lock.json"
    if path.exists():
        return json.loads(path.read_text())
    checkpoints = {}
    for seed in CONFIRMATION_SEEDS:
        histories, hashes = [], []
        for mode in MODES:
            directory = run_dir(seed, mode, root)
            result = json.loads((directory / "result.json").read_text())
            verified = json.loads((directory / "online_verification.json").read_text())
            checkpoint = directory / "best_event.pt"
            if not result["completed"] or not verified["passed"] or verified["head_sha256"] != sha256(checkpoint):
                raise RuntimeError("Completed training and matching online verification are required")
            histories.append(json.loads((directory / "history.json").read_text()))
            hashes.append(result["initial_hash"])
            checkpoints[f"{seed}/{mode}"] = {"path": str(checkpoint), "sha256": sha256(checkpoint), "epoch": result["best_epoch"]}
        if len(set(hashes)) != 1:
            raise RuntimeError("Null/BasePost initialization differs")
        for a, b in zip(*histories):
            if a["sample_order_sha256"] != b["sample_order_sha256"]:
                raise RuntimeError("Null/BasePost sample order differs")
    values = comparisons("validation", root)
    passed = all(values[key]["effective"] for key in ("base_post_minus_null", "base_post_minus_original"))
    lock = {"passed": passed, "selected_method": "EI-BasePost" if passed else "EI-Original",
        "primary_population": "all Raw-10 events including unknown Player", "comparisons": values,
        "checkpoints": checkpoints, "historical_test_results_previously_visible": True,
        "test_accessed": False, "test_policy": "confirmation only; never reselect"}
    write_json(path, lock)
    return lock


def evaluate_test(seed, root=ROOT, device="cpu"):
    require_test_lock(root)
    cache, provenance = load_cache(seed, "test", root)
    output = root / "test" / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame(cache, cache["base_event_logits"], seed).to_parquet(output / "test_original.parquet", index=False)
    for mode in MODES:
        saved = torch.load(run_dir(seed, mode, root) / "best_event.pt", map_location="cpu", weights_only=False)
        model = EventResidual(seed).to(device)
        model.load_state_dict(saved["residual"])
        prediction_frame(cache, predict(model, cache, mode), seed).to_parquet(output / f"test_{mode}.parquet", index=False)
    write_json(output / "provenance.json", provenance)
