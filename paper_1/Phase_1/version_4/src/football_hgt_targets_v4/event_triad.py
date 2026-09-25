"""Frozen Event triad refinement, with cross-fitted calibration first."""

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .constants import CONFIRMATION_SEEDS, EXPERIMENT_ROOT, POSSESSION_GRAPH_ROOT, SAMPLE_PLAN
from .event_posterior_integration import metrics, selection_value, eligible, paired_bootstrap, prediction_frame, cm_scores
from .oracle_dependency_study import cache_path, source_checkpoint
from .position_head_refit import capture_rng, restore_rng, save_checkpoint, sha256, tensor_hash, write_json

ROOT = EXPERIMENT_ROOT / "event_triad_refinement_v1"
TRIAD = (0, 6, 7)
MODES = ("context", "state")
GRID = tuple((float(d), float(o), 0.) for d, o in itertools.product(np.arange(-1.5, 1.501, .25), repeat=2))
COUNTS = {"train": 34048, "validation": 7296, "test": 96854}


def refine(logits, correction):
    """Preserve the triad's partition function and all outside logits."""
    group = logits[..., list(TRIAD)]
    changed = group + correction
    shift = group.logsumexp(-1, keepdim=True) - changed.logsumexp(-1, keepdim=True)
    result = logits.clone()
    result[..., list(TRIAD)] = changed + shift
    return result


def state_features(anchor_type, control_state):
    return torch.cat((F.one_hot(anchor_type.long(), 10), F.one_hot(control_state.long(), 4)), -1).float()


class TriadHead(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(nn.Linear(78, 32), nn.GELU(), nn.Dropout(.1), nn.Linear(32, 3))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        assert sum(p.numel() for p in self.parameters()) == 2627

    def forward(self, context, state):
        return self.network(torch.cat((context.detach(), state.detach()), -1))


def require_lock(root):
    path = root / "selection/event_triad_lock.json"
    if not path.exists():
        raise RuntimeError("Triad validation lock required before test access")
    lock = json.loads(path.read_text())
    if not lock["passed"]:
        raise RuntimeError("Validation failed: test access prohibited")
    for item in lock["artifacts"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("Locked artifact changed")
    return lock


def load_cache(seed, split, root=ROOT):
    if split not in COUNTS or seed not in CONFIRMATION_SEEDS:
        raise ValueError("Unknown population")
    if split == "test":
        require_lock(root)
    path, checkpoint = cache_path(seed, split), source_checkpoint(seed)
    raw = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    checkpoint_hash = sha256(checkpoint)
    if (raw["source_checkpoint_sha256"] != checkpoint_hash or raw["seed"] != seed or raw["split"] != split
            or Path(raw["source_checkpoint"]).resolve() != checkpoint.resolve()):
        raise RuntimeError("Cache provenance mismatch")
    ids = [f"{int(m)}:{int(a)}" for m, a in zip(raw["match_ids"], raw["current_event_indices"])]
    if ids != raw["sample_ids"] or len(ids) != COUNTS[split] or len(set(ids)) != len(ids):
        raise RuntimeError("Sample alignment mismatch")
    if split != "test":
        plan = json.loads(SAMPLE_PLAN.read_text())["selections"][split]
        if set(ids) != {f"{m}:{a}" for m, anchors in plan.items() for a in anchors}:
            raise RuntimeError("Sample plan mismatch")
    keys = ("main_context", "base_event_logits", "event_true", "player_mask", "match_ids", "current_event_indices", "zone_true")
    cache = {k: raw[k].detach().clone() for k in keys}
    cache["sample_ids"] = ids
    vocab_path = POSSESSION_GRAPH_ROOT.parent / "v1/metadata/vocabularies.json"
    vocab = json.loads(vocab_path.read_text())["event_type_ids"]
    sources = {str(p): sha256(p) for p in (checkpoint, path, SAMPLE_PLAN, vocab_path)}
    for field in ("anchor_type", "control_state", "event_role", "switch_confirmed"):
        cache[field] = torch.zeros(len(ids), dtype=torch.long)
    for match in torch.unique(raw["match_ids"]).tolist():
        graph_path = POSSESSION_GRAPH_ROOT / "graphs/England" / f"{match}.pt"
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        sources[str(graph_path)] = sha256(graph_path)
        event = graph["node_stores"]["event"]
        rows = torch.where(raw["match_ids"] == match)[0]
        anchor = raw["current_event_indices"][rows].long()
        types = torch.tensor([int(vocab[i]) - 1 for i in event["event_type_index"].tolist()])
        if not torch.equal(types[anchor + 1], cache["event_true"][rows]):
            raise RuntimeError("Adjacent target/source mismatch")
        cache["anchor_type"][rows] = types[anchor]
        for field, name in (("control_state", "control_state_after_index"), ("event_role", "event_role_index"),
                            ("switch_confirmed", "switch_confirmed")):
            cache[field][rows] = event[name][anchor].long()
    cache["state"] = state_features(cache["anchor_type"], cache["control_state"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    original = F.linear(cache["main_context"], saved["event_head.weight"], saved["event_head.bias"])
    error = float((original.softmax(-1) - cache["base_event_logits"].softmax(-1)).abs().max())
    if error >= 1e-6 or not torch.equal(original.argmax(-1), cache["base_event_logits"].argmax(-1)):
        raise RuntimeError("Checkpoint/cache probability mismatch")
    return cache, {"sources": sources, "cache_probability_error": error,
                   "code_sha256": sha256(Path(__file__)), "samples": len(ids)}


def triad_f1(frame):
    scores = []
    for _, part in frame.groupby("seed", sort=True):
        cm = np.bincount(part.event_true.to_numpy() * 10 + part.event_pred.to_numpy(), minlength=100).reshape(10, 10)
        denom = cm.sum(0) + cm.sum(1)
        f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(10), where=denom > 0)
        scores.append(float(f1[list(TRIAD)].mean()))
    return float(np.mean(scores))


def compare(reference, candidate):
    result = paired_bootstrap(reference, candidate)
    result["triad_macro_f1_gain"] = triad_f1(candidate) - triad_f1(reference)
    result["effective"] = bool(result["effective"] and result["triad_macro_f1_gain"] > 0)
    return result


def match_folds(matches):
    ordered = np.sort(np.unique(matches))
    np.random.default_rng(20260913).shuffle(ordered)
    return {int(match): i % 5 for i, match in enumerate(ordered)}


def grid_predictions(caches):
    return np.stack([np.stack([refine(c["base_event_logits"], torch.tensor(b)).argmax(-1).numpy()
                              for b in GRID]) for c in caches])


def choose_bias(caches, predictions, rows):
    """Only tuning rows participate, including the Accuracy guard."""
    targets = caches[0]["event_true"].numpy()[rows]
    original = np.mean([metrics(c["base_event_logits"][rows], c["event_true"][rows])["accuracy"] for c in caches])
    choices = []
    for j, bias in enumerate(GRID):
        scores = []
        for s in range(len(caches)):
            cm = np.bincount(targets * 10 + predictions[s, j, rows], minlength=100).reshape(10, 10)
            scores.append(cm_scores(cm))
        f1, accuracy = np.mean(scores, axis=0)
        if accuracy >= original - .01 - 1e-12:
            choices.append(((-f1, -accuracy, float(np.dot(bias, bias)), bias), j))
    return min(choices)[1]


def calibrate(root=ROOT):
    output = root / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    caches, provenance = [], {}
    for seed in CONFIRMATION_SEEDS:
        cache, sources = load_cache(seed, "validation", root)
        caches.append(cache)
        provenance[str(seed)] = sources
    for c in caches[1:]:
        if c["sample_ids"] != caches[0]["sample_ids"] or not torch.equal(c["event_true"], caches[0]["event_true"]):
            raise RuntimeError("Cross-seed alignment mismatch")
    folds = match_folds(caches[0]["match_ids"].numpy())
    fold = np.array([folds[int(m)] for m in caches[0]["match_ids"]])
    predictions = grid_predictions(caches)
    choices = {str(k): choose_bias(caches, predictions, np.flatnonzero(fold != k)) for k in range(5)}
    originals, corrected = [], []
    for seed, c in zip(CONFIRMATION_SEEDS, caches):
        logits = c["base_event_logits"].clone()
        for k in range(5):
            rows = torch.from_numpy(np.flatnonzero(fold == k))
            logits[rows] = refine(logits[rows], torch.tensor(GRID[choices[str(k)]]))
        base = prediction_frame(c, c["base_event_logits"], seed)
        pred = prediction_frame(c, logits, seed)
        for f in (base, pred):
            f["anchor_type"] = c["anchor_type"].numpy()
            f["fold"] = fold
        originals.append(base)
        corrected.append(pred)
    original, oof = pd.concat(originals, ignore_index=True), pd.concat(corrected, ignore_index=True)
    original.to_parquet(output / "validation_original.parquet", index=False)
    oof.to_parquet(output / "validation_oof.parquet", index=False)
    outcome = compare(original, oof)
    final_index = choose_bias(caches, predictions, np.arange(len(fold)))
    result = {"passed": outcome["effective"], "comparison": outcome,
              "folds": folds, "fold_biases": {k: GRID[v] for k, v in choices.items()},
              "final_bias": GRID[final_index], "grid": GRID, "provenance": provenance,
              "final_bias_used_only_after_oof_decision": True}
    write_json(output / "decision.json", result)
    print("calibration: " + json.dumps(outcome), flush=True)
    return result


@torch.no_grad()
def predict(model, cache, mode, batch_size=1024):
    model.eval()
    device = next(model.parameters()).device
    result = []
    for rows in torch.arange(len(cache["event_true"])).split(batch_size):
        state = cache["state"][rows] if mode == "state" else torch.zeros((len(rows), 14))
        delta = model(cache["main_context"][rows].to(device), state.to(device))
        result.append(refine(cache["base_event_logits"][rows].to(device), delta).cpu())
    return torch.cat(result)


@dataclass(frozen=True)
class HeadConfig:
    seed: int
    mode: str
    device: str = "cpu"
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 1e-4


@torch.backends.mkldnn.flags(enabled=False)
def fit(train, validation, config, output, provenance, resume=None, stop_after=None):
    if config.mode not in MODES:
        raise ValueError(config.mode)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = TriadHead(config.seed).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    initial_hash = tensor_hash(model.state_dict())
    original = metrics(validation["base_event_logits"], validation["event_true"])
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["config"] != asdict(config) or saved["provenance"] != provenance:
            raise RuntimeError("Resume provenance/configuration differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, stale, best, history = saved["epoch"], saved["stale"], saved["best"], saved["history"]
        elapsed = saved["elapsed_seconds"]
        restore_rng(saved["rng"], generator)
    else:
        if float((predict(model, validation, config.mode) - validation["base_event_logits"]).abs().max()) >= 1e-6:
            raise RuntimeError("Epoch zero not equivalent")
        epoch, stale, elapsed = 0, 0, 0.
        history = [{"epoch": 0, "validation": original, "eligible": True, "sample_order_sha256": None}]
        best = {"epoch": 0, "selection_value": selection_value(original, 0),
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    started = time.monotonic()

    def persist():
        common = {"config": asdict(config), "provenance": provenance, "initial_hash": initial_hash}
        save_checkpoint(output / "best_event.pt", {**common, **best})
        save_checkpoint(output / "last.pt", {**common, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "stale": stale, "best": best, "history": history, "rng": capture_rng(generator),
            "elapsed_seconds": elapsed + time.monotonic() - started})
        write_json(output / "history.json", history)

    persist()
    budget = config.max_epochs if stop_after is None else min(config.max_epochs, stop_after)
    while epoch < budget and stale < config.patience:
        epoch += 1
        model.train()
        order = torch.randperm(len(train["event_true"]), generator=generator)
        total = 0.
        for rows in order.split(config.batch_size):
            state = train["state"][rows] if config.mode == "state" else torch.zeros((len(rows), 14))
            delta = model(train["main_context"][rows].to(config.device), state.to(config.device))
            logits = refine(train["base_event_logits"][rows].to(config.device), delta)
            loss = F.cross_entropy(logits, train["event_true"][rows].to(config.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        values = metrics(predict(model, validation, config.mode), validation["event_true"])
        allowed = eligible(values, original)
        score = selection_value(values, epoch)
        history.append({"epoch": epoch, "validation": values, "eligible": allowed, "train_loss": total / len(order),
                        "sample_order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest()})
        if allowed and score < tuple(best["selection_value"]):
            best = {"epoch": epoch, "selection_value": score,
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        persist()
    model.load_state_dict(best["model"])
    for name, logits in (("original", validation["base_event_logits"]), (config.mode, predict(model, validation, config.mode))):
        frame = prediction_frame(validation, logits, config.seed)
        frame["anchor_type"] = validation["anchor_type"].numpy()
        frame.to_parquet(output / f"validation_{name}.parquet", index=False)
    result = {"completed": epoch == config.max_epochs or stale >= config.patience, "completed_epoch": epoch,
              "best_epoch": best["epoch"], "initial_hash": initial_hash, "config": asdict(config),
              "validation": history[best["epoch"]]["validation"], "elapsed_seconds": elapsed + time.monotonic() - started}
    write_json(output / "result.json", result)
    return result


def train_seed(seed, root=ROOT, device="cpu", mode=None, resume=None, smoke=False):
    if (root / "selection/event_triad_lock.json").exists():
        raise RuntimeError("Locked experiment cannot be retrained")
    if not smoke:
        decision = json.loads((root / "calibration/decision.json").read_text())
        if decision["passed"]:
            raise RuntimeError("Calibration passed: Head stage prohibited")
    train, ptrain = load_cache(seed, "train", root)
    validation, pval = load_cache(seed, "validation", root)
    for selected in ((mode,) if mode else MODES):
        directory = root / ("smoke" if smoke else "training") / selected / f"seed{seed}"
        last = resume or (directory / "last.pt" if (directory / "last.pt").exists() else None)
        result = fit(train, validation, HeadConfig(seed, selected, device), directory,
                     {"train": ptrain, "validation": pval}, last, 1 if smoke else None)
        print(f"{selected} seed={seed} best={result['best_epoch']} epoch={result['completed_epoch']}", flush=True)


def read_validation(root, mode):
    frames = []
    for seed in CONFIRMATION_SEEDS:
        path = root / "training" / ("context" if mode == "original" else mode) / f"seed{seed}" / f"validation_{mode}.parquet"
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def select(root=ROOT):
    path = root / "selection/event_triad_lock.json"
    if path.exists():
        return json.loads(path.read_text())
    calibration = json.loads((root / "calibration/decision.json").read_text())
    artifacts = {"calibration": root / "calibration/decision.json"}
    comparisons = {"calibration_oof_minus_original": calibration["comparison"]}
    if calibration["passed"]:
        selected, test_modes = "calibration", ["original", "calibration"]
    else:
        frames = {m: read_validation(root, m) for m in ("original", *MODES)}
        for new, old in (("context", "original"), ("state", "context"), ("state", "original")):
            comparisons[f"{new}_minus_{old}"] = compare(frames[old], frames[new])
        for seed in CONFIRMATION_SEEDS:
            histories, hashes = [], []
            for mode in MODES:
                directory = root / "training" / mode / f"seed{seed}"
                result = json.loads((directory / "result.json").read_text())
                if not result["completed"]:
                    raise RuntimeError("Training incomplete")
                histories.append(json.loads((directory / "history.json").read_text()))
                hashes.append(result["initial_hash"])
                artifacts[f"{seed}/{mode}"] = directory / "best_event.pt"
            if len(set(hashes)) != 1 or any(a["sample_order_sha256"] != b["sample_order_sha256"] for a, b in zip(*histories)):
                raise RuntimeError("Head initialization/order mismatch")
        candidates = ["original"]
        if comparisons["context_minus_original"]["effective"]:
            candidates.append("context")
        if comparisons["state_minus_original"]["effective"] and comparisons["state_minus_context"]["effective"]:
            candidates.append("state")
        def score(mode):
            return np.mean([float(cm_scores(np.bincount(p.event_true.to_numpy() * 10 + p.event_pred.to_numpy(), minlength=100).reshape(10, 10))[0])
                            for _, p in frames[mode].groupby("seed")])
        best = max(map(score, candidates))
        selected = next(m for m in candidates if best - score(m) < 1e-4)
        test_modes = ["original", selected] if selected != "state" else ["original", "context", "state"]
    for seed in CONFIRMATION_SEEDS:
        verification = root / "verification" / f"{selected}_seed{seed}.json"
        from .event_triad_online import verify_online
        verify_online(seed, selected, root)
        checked = json.loads(verification.read_text())
        if not checked["passed"]:
            raise RuntimeError("Online verification required")
        artifacts[f"online/{seed}"] = verification
    lock = {"passed": selected != "original", "selected_method": selected, "test_modes": list(dict.fromkeys(test_modes)),
            "comparisons": comparisons, "artifacts": {k: {"path": str(p), "sha256": sha256(p)} for k, p in artifacts.items()},
            "historical_test_previously_seen": True, "test_policy": "confirmation only, no reselection"}
    write_json(path, lock)
    return lock


def correction_model(seed, mode, root=ROOT, device="cpu"):
    if mode in ("original", "calibration"):
        bias = [0., 0., 0.] if mode == "original" else json.loads((root / "calibration/decision.json").read_text())["final_bias"]
        return torch.tensor(bias, device=device)
    saved = torch.load(root / "training" / mode / f"seed{seed}/best_event.pt", map_location="cpu", weights_only=False)
    model = TriadHead(seed).to(device)
    model.load_state_dict(saved["model"])
    return model.eval().requires_grad_(False)


def evaluate_test(seed, root=ROOT, device="cpu"):
    lock = require_lock(root)
    cache, provenance = load_cache(seed, "test", root)
    output = root / "test" / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    for mode in lock["test_modes"]:
        model = correction_model(seed, mode, root, device)
        logits = refine(cache["base_event_logits"], model.cpu()) if isinstance(model, torch.Tensor) else predict(model, cache, mode)
        frame = prediction_frame(cache, logits, seed)
        frame["anchor_type"] = cache["anchor_type"].numpy()
        frame.to_parquet(output / f"test_{mode}.parquet", index=False)
    write_json(output / "provenance.json", provenance)
