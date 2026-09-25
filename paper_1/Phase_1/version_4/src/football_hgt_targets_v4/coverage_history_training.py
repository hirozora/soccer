"""Isolated coverage study; existing checkpoints and protocols stay read-only."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from football_benchmark.data import MatchBlockShuffleSampler, load_records
from football_benchmark.protocol import ProtocolArtifacts
from .constants import EXPERIMENT_ROOT, FEASIBILITY_ARTIFACT, SAMPLE_PLAN, POSSESSION_GRAPH_ROOT
from .coverage_history_data import RotatingEventDataset
from .event_posterior_online import IntegratedEventModel, load_integrated_model
from .five_task_training import _loader
from .fixed_budget_study import ALL_TASKS, training_dir as five_dir
from .fixed_budget_training import (FixedBudgetConfig, _loader_config, _common_hash, _rng_state,
    _restore_rng, _train_epoch, evaluate_fixed_budget, guarded_core_eligible)
from .model import build_partial_l2_model, build_five_task_model
from .oracle_dependency import load_roster_team_map
from .partial_sharing_study import training_dir as base_dir
from .position_head_refit import (save_checkpoint, sha256, tensor_hash, RefitConfig, fit_cached,
    head_from_state, predict, ROOT as POSITION_ROOT)
from .spatiotemporal_training import write_json
from .spatiotemporal_reporting import compare, core_loss, effective, guards
from .training import set_seed, _move_batch_to_device


ROOT = EXPERIMENT_ROOT / "coverage_history_v1"
SEEDS = (20260715, 20260716, 20260717)
COUNTS = {"train": 449025, "validation": 96891, "test": 96854}


def run_dir(mode, seed, root=ROOT):
    return root / "coverage" / mode / f"seed{seed}"


def frame_path(mode, seed, split="validation", root=ROOT):
    if split == "test":
        require_test(root)
    return root / "predictions" / split / mode / f"seed{seed}.parquet"


def sources():
    files = [FEASIBILITY_ARTIFACT, SAMPLE_PLAN]
    files += list(Path(__file__).parent.glob("coverage_history*.py"))
    files += [Path(__file__).resolve().parents[2] / "scripts/run_coverage_history.py"]
    files += [Path(__file__).parent / f"{m}.py" for m in
              ("model", "fixed_budget_training", "five_task_training", "fixed_budget_loss", "position_head_refit")]
    files += [base_dir(s) / "best_guarded_core.pt" for s in SEEDS]
    from .coverage_history_data import MATCHES, PHASE
    files += [MATCHES, PHASE / "version_1/data_splits/temporal_match_split_v1.csv"]
    files += [p for p in (POSSESSION_GRAPH_ROOT / "metadata").glob("*") if p.is_file()]
    return {str(p): sha256(p) for p in files}


def require_test(root=ROOT):
    paths = [root / "selection/coverage_lock.json", root / "selection/history_lock.json"]
    if not all(p.exists() for p in paths):
        raise RuntimeError("Both validation locks are required before any test access")
    locks = [json.loads(p.read_text()) for p in paths]
    if locks[1]["coverage_lock_sha256"] != sha256(paths[0]):
        raise RuntimeError("Coverage selection changed after history lock")
    for lock in locks:
        for path, digest in lock["checkpoints"].items():
            if sha256(Path(path)) != digest:
                raise RuntimeError("Locked checkpoint changed")
    return locks


def loader_for(split, seed, device, root=ROOT, full=True, workers=2, limit=None):
    if split == "test":
        require_test(root)
    cfg = FixedBudgetConfig("partial_l2", root, seed, device, training_budget=24,
        sample_plan_path=None if full else SAMPLE_PLAN, num_workers=workers,
        max_train_samples=limit if split == "train" else None,
        max_validation_samples=limit if split == "validation" else None,
        max_test_samples=limit if split == "test" else None)
    return _loader(split, _loader_config(cfg), ProtocolArtifacts.load(FEASIBILITY_ARTIFACT), False)


def rotation_loader(mode, seed, device, root, workers=2, smoke=False):
    cfg = FixedBudgetConfig("partial_l2", run_dir(mode, seed, root), seed, device,
        training_budget=24, num_workers=workers)
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    template = _loader("train", _loader_config(cfg), artifacts, True)
    dataset = RotatingEventDataset(load_records("train", graph_root=POSSESSION_GRAPH_ROOT), artifacts,
                                  rotate=mode == "rotate", max_samples=256 if smoke else None)
    options = {"persistent_workers": True, "prefetch_factor": 2} if workers else {}
    loader = DataLoader(dataset, batch_size=256, sampler=MatchBlockShuffleSampler(dataset, seed),
        num_workers=workers, collate_fn=template.collate_fn, generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(), drop_last=False, **options)
    return loader, cfg


def load_backbone(mode, seed, device="cpu", root=ROOT):
    path = base_dir(seed) / "best_guarded_core.pt" if mode == "original" else run_dir(mode, seed, root) / "best_guarded_core.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    with torch.random.fork_rng(devices=[]):
        model = build_partial_l2_model(ProtocolArtifacts.load(FEASIBILITY_ARTIFACT))
    model.load_state_dict(saved["model"])
    return model.to(device).requires_grad_(False).eval(), path


def load_deployed(mode, seed, device="cpu", root=ROOT):
    if mode == "original":
        return load_integrated_model(seed, device=device)
    model, _ = load_backbone(mode, seed, device, root)
    state = torch.load(run_dir(mode, seed, root) / "position_refit/best_position.pt", map_location="cpu", weights_only=False)
    model.position_head.load_state_dict(state["head"])
    return IntegratedEventModel(model, ProtocolArtifacts.load(FEASIBILITY_ARTIFACT), load_roster_team_map()).to(device).eval()


@torch.no_grad()
def reference_evaluation(seed, device, root=ROOT, smoke=False):
    out = root / "references" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    paths = {"original": base_dir(seed) / "best_guarded_core.pt", "five": five_dir("five_f80", seed) / "best_joint.pt"}
    result = {}
    for name, path in paths.items():
        cache = out / f"{name}.json"
        if cache.exists():
            old = json.loads(cache.read_text())
            if old["checkpoint_sha256"] != sha256(path) or old["smoke"] != smoke:
                raise RuntimeError("Stale reference evaluation")
            result[name] = old["metrics"]
            continue
        with torch.random.fork_rng(devices=[]):
            model = build_partial_l2_model(artifacts) if name == "original" else build_five_task_model(artifacts, "five_f80")
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
        model.to(device).eval()
        loader = loader_for("validation", seed, device, root, workers=0 if smoke else 2, limit=64 if smoke else None)
        metrics, frame = evaluate_fixed_budget(model, loader, artifacts, ALL_TASKS, torch.device(device))
        if len(frame) != (64 if smoke else COUNTS["validation"]):
            raise RuntimeError("Reference validation population mismatch")
        frame.to_parquet(out / f"{name}_raw.parquet", index=False)
        write_json(cache, {"metrics": metrics, "checkpoint_sha256": sha256(path), "smoke": smoke})
        result[name] = metrics
    return result


def actor_reference(metrics):
    return {"team_accuracy": metrics["team"]["accuracy"], "player_top1": metrics["player"]["top1_accuracy"]}


def train_coverage(mode, seed, device, root=ROOT, workers=2, smoke=False, resume_from=None):
    if mode not in ("fixed", "rotate"):
        raise ValueError(mode)
    if (root / "selection/coverage_lock.json").exists():
        raise RuntimeError("Coverage selection is locked")
    out = run_dir(mode, seed, root)
    out.mkdir(parents=True, exist_ok=True)
    source = sources()
    if (out / "result.json").exists():
        result = json.loads((out / "result.json").read_text())
        if result["sources"] != source:
            raise RuntimeError("Completed run has different sources")
        return result
    reference = reference_evaluation(seed, device, root, smoke)
    set_seed(seed)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    model = build_partial_l2_model(artifacts).to(dev)
    initial = _common_hash(model)
    expected = json.loads((base_dir(seed) / "result.json").read_text())["initial_common_sha256"]
    if initial != expected:
        raise RuntimeError("Initial backbone/head parameters differ from baseline")
    optimizer = torch.optim.AdamW(model.parameters(), lr=9e-4, weight_decay=1e-4)
    train_loader, cfg = rotation_loader(mode, seed, device, root, workers, smoke)
    validation = loader_for("validation", seed, device, root, workers=workers, limit=64 if smoke else None)
    guard_refs = [(actor_reference(reference["five"]), .01), (actor_reference(reference["original"]), .005)]
    best = {"core": float("inf"), "joint": float("inf"), "guarded_core": float("inf")}
    history, elapsed, steps = [], 0., 0
    last = Path(resume_from) if resume_from else out / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["sources"] != source or saved["mode"] != mode or saved["seed"] != seed or saved["smoke"] != smoke:
            raise RuntimeError("Resume source/configuration mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, elapsed, steps = saved["history"], saved["best"], saved["elapsed"], saved["steps"]
        _restore_rng(saved["rng"], train_loader)
    started = time.monotonic()
    budget = 1 if smoke else 24
    for epoch in range(len(history) + 1, budget + 1):
        train_loader.dataset.set_epoch(epoch)
        train_metrics = _train_epoch(model, train_loader, optimizer, artifacts, cfg, dev)
        steps += len(train_loader)
        row = {"epoch": epoch, "train": train_metrics, "steps": steps,
               "target_plan_sha256": train_loader.dataset.plan_hash(epoch), "validation": None}
        improved = []
        if smoke or epoch % 4 == 0:
            with torch.no_grad():
                values, frame = evaluate_fixed_budget(model, validation, artifacts, ALL_TASKS, dev)
            allowed = all(guarded_core_eligible(values, r, margin=m) for r, m in guard_refs)
            row.update(validation=values, guarded_core_eligible=allowed)
            if not np.isfinite(values["core_etp_loss"] + values["joint_active_loss"]):
                raise RuntimeError("Nonfinite validation loss")
            frame.to_parquet(out / f"validation_epoch{epoch:02d}.parquet", index=False)
            for name, key in (("core", "core_etp_loss"), ("joint", "joint_active_loss"), ("guarded_core", "core_etp_loss")):
                if (name != "guarded_core" or allowed) and values[key] < best[name]:
                    best[name] = values[key]
                    improved.append(name)
        history.append(row)
        payload = {"architecture": "partial_l2", "mode": mode, "seed": seed, "smoke": smoke,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
                   "epoch": epoch, "steps": steps, "best": best, "rng": _rng_state(train_loader),
                   "sources": source, "elapsed": elapsed + time.monotonic() - started,
                   "initial_common_sha256": initial}
        for name in improved:
            save_checkpoint(out / f"best_{name}.pt", payload)
        save_checkpoint(out / "last.pt", payload)
        write_json(out / "history.json", history)
        print(json.dumps({"mode": mode, "seed": seed, "epoch": epoch, "steps": steps,
                          "train": train_metrics, "validation_core": (row["validation"] or {}).get("core_etp_loss")}), flush=True)
    if steps != (1 if smoke else 3192):
        raise RuntimeError("Wrong optimizer budget")
    result = {"mode": mode, "seed": seed, "epochs": budget, "steps": steps,
              "eligible": (out / "best_guarded_core.pt").exists(), "sources": source,
              "initial_common_sha256": initial, "seconds": elapsed + time.monotonic() - started,
              "peak_memory_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else 0}
    write_json(out / "result.json", result)
    return result


@torch.no_grad()
def position_cache(mode, seed, split, device, root=ROOT):
    if split == "test":
        require_test(root)
    model, checkpoint = load_backbone(mode, seed, device, root)
    out = run_dir(mode, seed, root) / f"{split}_position_context.pt"
    digest = sha256(checkpoint)
    if out.exists():
        cache = torch.load(out, map_location="cpu", weights_only=False)
        if cache["checkpoint_sha256"] != digest:
            raise RuntimeError("Stale Position cache")
        return cache
    loader = loader_for(split, seed, device, root, full=split != "train")
    chunks, sample_ids = {}, []
    for raw in loader:
        batch = _move_batch_to_device(raw, torch.device(device))
        predictions, contexts = model.forward_with_contexts(batch)
        context, t = contexts["f80"], batch["targets"]
        if (model.position_head(context).sigmoid() - predictions["position_xy"]).abs().max() >= 1e-6:
            raise RuntimeError("Position/context mismatch")
        values = {"main_context": context, "base_position_xy": predictions["position_xy"],
                  "position_true": t["position_xy"], "position_mask": t["position_mask"],
                  "player_mask": t["player_mask"], "event_true": t["raw_event_10"], "zone_true": t["zone_20"],
                  "match_ids": batch["match_ids"], "current_event_indices": batch["current_event_indices"]}
        for k, v in values.items():
            chunks.setdefault(k, []).append(v.detach().cpu())
        sample_ids.extend(raw["sample_ids"])
    cache = {k: torch.cat(v) for k, v in chunks.items()}
    expected = 34048 if split == "train" else COUNTS[split]
    if len(sample_ids) != expected or len(set(sample_ids)) != expected:
        raise RuntimeError("Position cache population mismatch")
    cache.update(sample_ids=sample_ids, checkpoint_sha256=digest)
    save_checkpoint(out, cache)
    return cache


@torch.no_grad()
def evaluate_deployed(mode, seed, split, device, root=ROOT):
    if split == "test":
        require_test(root)
    path = frame_path(mode, seed, split, root)
    if path.exists():
        return pd.read_parquet(path)
    model = load_deployed(mode, seed, device, root)
    metrics, frame = evaluate_fixed_budget(model, loader_for(split, seed, device, root),
        ProtocolArtifacts.load(FEASIBILITY_ARTIFACT), ALL_TASKS, torch.device(device))
    if len(frame) != COUNTS[split]:
        raise RuntimeError("Deployed evaluation population mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), metrics)
    return frame


def refit_coverage(mode, seed, device, root=ROOT):
    if mode == "original":
        evaluate_deployed(mode, seed, "validation", device, root)
        return
    out = run_dir(mode, seed, root)
    if not json.loads((out / "result.json").read_text())["eligible"]:
        return
    train = position_cache(mode, seed, "train", device, root)
    validation = position_cache(mode, seed, "validation", device, root)
    model, path = load_backbone(mode, seed, "cpu", root)
    head = head_from_state(model.state_dict())
    if (predict(head, validation["main_context"]) - validation["base_position_xy"]).abs().max() >= 1e-6:
        raise RuntimeError("Position epoch zero mismatch")
    target = out / "position_refit"
    fit_cached(head, train, validation, RefitConfig(seed), target, {"checkpoint": str(path), "sha256": sha256(path)},
               resume_from=target / "last.pt" if (target / "last.pt").exists() else None)
    evaluate_deployed(mode, seed, "validation", device, root)


def choose_coverage(comparisons, losses):
    eligible = ["original"]
    for mode in ("fixed", "rotate"):
        comparison = comparisons.get(f"{mode}-original")
        if comparison and effective(comparison) and guards(comparison):
            eligible.append(mode)
    minimum = min(losses[m] for m in eligible)
    return next(m for m in ("original", "fixed", "rotate") if m in eligible and losses[m] < minimum + 1e-4), eligible


def coverage_checkpoints(mode, root=ROOT):
    paths = []
    for seed in SEEDS:
        if mode == "original":
            paths += [base_dir(seed) / "best_guarded_core.pt", POSITION_ROOT / "training" / f"seed{seed}" / "best_position.pt"]
        else:
            paths += [run_dir(mode, seed, root) / "best_guarded_core.pt", run_dir(mode, seed, root) / "position_refit/best_position.pt"]
    return {str(p): sha256(p) for p in paths}


def select_coverage(root=ROOT):
    path = root / "selection/coverage_lock.json"
    if path.exists():
        return json.loads(path.read_text())
    complete = ["original"]
    for mode in ("fixed", "rotate"):
        results = [json.loads((run_dir(mode, s, root) / "result.json").read_text()) for s in SEEDS]
        if any(r["epochs"] != 24 or r["steps"] != 3192 for r in results):
            raise RuntimeError("Incomplete fixed training budget")
        if all(r["eligible"] for r in results):
            complete.append(mode)
    frames = {m: [pd.read_parquet(frame_path(m, s, root=root)) for s in SEEDS] for m in complete}
    comparisons = {f"{a}-{b}": compare(frames[b], frames[a]) for a, b in
                   (("fixed", "original"), ("rotate", "original"), ("rotate", "fixed")) if a in frames and b in frames}
    losses = {m: float(np.mean([core_loss(f) for f in fs])) for m, fs in frames.items()}
    selected, eligible = choose_coverage(comparisons, losses)
    checkpoint_hashes = {p: h for mode in complete for p, h in coverage_checkpoints(mode, root).items()}
    lock = {"selected": selected, "eligible": eligible, "complete": complete, "comparisons": comparisons,
            "core_losses": losses, "checkpoints": checkpoint_hashes,
            "test_accessed": False, "validation_samples": COUNTS["validation"],
            "coverage_supported_tasks": effective(comparisons["rotate-fixed"]) if "rotate-fixed" in comparisons else []}
    write_json(path, lock)
    return lock
