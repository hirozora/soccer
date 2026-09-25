"""Same-architecture Position-only refinement of frozen Partial-L2 contexts."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .constants import CONFIRMATION_SEEDS, EXPERIMENT_ROOT, SAMPLE_PLAN
from .oracle_dependency_study import cache_path, source_checkpoint


ROOT = EXPERIMENT_ROOT / "position_head_refit_v1"
EXPECTED_COUNTS = {"train": 34048, "validation": 7296, "test": 96854}
FIELDS = (
    "main_context", "position_true", "position_mask", "player_mask",
    "base_position_xy", "event_true", "zone_true", "match_ids",
    "current_event_indices",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run_dir(seed: int, root: Path = ROOT) -> Path:
    return root / "training" / f"seed{seed}"


def require_test_lock(root: Path = ROOT) -> dict[str, Any]:
    path = root / "selection/position_refit_lock.json"
    if not path.exists():
        raise RuntimeError("Position-refit validation lock required before any test read")
    lock = json.loads(path.read_text())
    if not lock.get("passed") or lock.get("selected_method") != "PR-Refit":
        raise RuntimeError("Validation did not authorize a new test evaluation")
    for item in lock["checkpoints"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("A locked Head checkpoint changed")
    return lock


def head_from_state(state: dict[str, torch.Tensor]) -> nn.Linear:
    values = {key.removeprefix("position_head."): value for key, value in state.items()
              if key.startswith("position_head.")}
    if set(values) != {"weight", "bias"} or values["weight"].shape != (2, 64) or values["bias"].shape != (2,):
        raise RuntimeError("Expected the existing Linear(64,2) Position Head")
    with torch.random.fork_rng(devices=[]):
        head = nn.Linear(64, 2)
    head.load_state_dict(values)
    return head


@torch.no_grad()
def predict(head: nn.Linear, context: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    head.eval()
    device = next(head.parameters()).device
    return torch.cat([head(x.to(device)).sigmoid().cpu()
                      for x in context.detach().split(batch_size)])


def position_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Index before evaluating the loss so invalid coordinates cannot introduce NaNs.
    mask = mask.bool()
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[mask], target[mask], beta=1.0, reduction="mean")


def metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    mask = mask.bool()
    if not bool(mask.any()):
        raise ValueError("An evaluation population must contain valid positions")
    delta = (prediction[mask].double() - target[mask].double()).numpy()
    delta *= np.array([105.0, 68.0])
    distances = np.linalg.norm(delta, axis=1)
    return {
        "samples": int(mask.sum()), "mean_distance_m": float(distances.mean()),
        "median_distance_m": float(np.median(distances)),
        "distance_rmse_m": float(np.sqrt(np.mean(distances ** 2))),
        "x_mae_m": float(np.abs(delta[:, 0]).mean()),
        "y_mae_m": float(np.abs(delta[:, 1]).mean()),
        "loss": float(position_loss(prediction, target, mask)),
    }


def selection_value(values: dict[str, Any], epoch: int) -> tuple[float, float, int]:
    return values["mean_distance_m"], values["loss"], epoch


def load_cache(seed: int, split: str, root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    if seed not in CONFIRMATION_SEEDS or split not in EXPECTED_COUNTS:
        raise ValueError("Unsupported seed/split")
    if split == "test":
        require_test_lock(root)
    path, checkpoint_path = cache_path(seed, split), source_checkpoint(seed)
    raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_hash = sha256(checkpoint_path)
    if (raw["seed"] != seed or raw["split"] != split
            or Path(raw["source_checkpoint"]).resolve() != checkpoint_path.resolve()
            or raw["source_checkpoint_sha256"] != checkpoint_hash):
        raise RuntimeError("Cache/checkpoint provenance mismatch")
    count = len(raw["sample_ids"])
    if count != EXPECTED_COUNTS[split] or len(set(raw["sample_ids"])) != count:
        raise RuntimeError("Incorrect cache sample count or duplicate sample IDs")
    expected_ids = [f"{m}:{a}" for m, a in zip(raw["match_ids"].tolist(), raw["current_event_indices"].tolist())]
    if expected_ids != raw["sample_ids"]:
        raise RuntimeError("Sample IDs do not identify their match and causal anchor")
    if split != "test":
        plan = json.loads(SAMPLE_PLAN.read_text())["selections"][split]
        planned = {f"{m}:{a}" for m, anchors in plan.items() for a in anchors}
        if set(expected_ids) != planned:
            raise RuntimeError("Cache differs from the fixed sample plan")
    cache = {key: raw[key].detach().cpu() for key in FIELDS}
    cache["sample_ids"] = list(raw["sample_ids"])
    for key in FIELDS:
        if cache[key].shape[0] != count:
            raise RuntimeError(f"Misaligned cache field: {key}")
    if cache["main_context"].shape != (count, 64) or cache["position_true"].shape != (count, 2):
        raise RuntimeError("Invalid context/target dimensions")
    for key in ("position_mask", "player_mask"):
        if cache[key].dtype != torch.bool:
            raise RuntimeError(f"{key} must be boolean")
    valid_targets = cache["position_true"][cache["position_mask"]]
    if (not torch.isfinite(cache["main_context"]).all() or not torch.isfinite(valid_targets).all()
            or bool(((valid_targets < 0) | (valid_targets > 1)).any())):
        raise RuntimeError("Invalid frozen contexts or normalized target coordinates")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    initial = predict(head_from_state(state), cache["main_context"])
    error = float((initial - cache["base_position_xy"]).abs().max())
    if error >= 1e-6:
        raise RuntimeError(f"Epoch 0 cache mismatch: {error}")
    provenance = {
        "cache_path": str(path), "cache_sha256": sha256(path),
        "source_checkpoint": str(checkpoint_path), "source_checkpoint_sha256": checkpoint_hash,
        "sample_plan_sha256": sha256(SAMPLE_PLAN), "samples": count,
        "epoch0_max_difference": error,
        "valid_positions": int(cache["position_mask"].sum()),
        "valid_unknown_player_positions": int((cache["position_mask"] & ~cache["player_mask"]).sum()),
    }
    return cache, provenance


def capture_rng(generator: torch.Generator) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(), "sampler": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(rng: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["cpu"].cpu())
    generator.set_state(rng["sampler"].cpu())
    if rng["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in rng["cuda"]])


@dataclass(frozen=True)
class RefitConfig:
    seed: int
    device: str = "cpu"
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0


def fit_cached(
    head: nn.Linear, train: dict[str, Any], validation: dict[str, Any],
    config: RefitConfig, output: Path, provenance: dict[str, Any],
    resume_from: Path | None = None, stop_after_epoch: int | None = None,
) -> dict[str, Any]:
    """The optional epoch boundary is for smoke/resumption verification, not selection."""
    if config.max_epochs < 1 or config.patience < 1 or config.batch_size < 1:
        raise ValueError("Invalid training budget")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    head = head.to(config.device)
    if sum(p.numel() for p in head.parameters()) != 130:
        raise RuntimeError("Only the original 130 Position parameters may be trained")
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if resume_from is None:
        values = metrics(predict(head, validation["main_context"]), validation["position_true"], validation["position_mask"])
        history = [{"epoch": 0, "validation": values, "train_loss": None, "sample_order_sha256": None}]
        best = {"epoch": 0, "selection_value": selection_value(values, 0),
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
        epoch, stale, elapsed = 0, 0, 0.0
        initial_hash = tensor_hash(head.state_dict())
    else:
        restored = torch.load(resume_from, map_location="cpu", weights_only=False)
        if restored["config"] != asdict(config) or restored["provenance"] != provenance:
            raise RuntimeError("Resume configuration or inputs changed")
        head.load_state_dict(restored["head"])
        optimizer.load_state_dict(restored["optimizer"])
        epoch, stale = restored["epoch"], restored["stale"]
        history, best = restored["history"], restored["best"]
        elapsed, initial_hash = restored["elapsed_seconds"], restored["initial_head_sha256"]
        restore_rng(restored["rng"], generator)
    started = time.monotonic()

    def persist() -> None:
        total_elapsed = elapsed + time.monotonic() - started
        common = {"config": asdict(config), "provenance": provenance, "initial_head_sha256": initial_hash}
        save_checkpoint(output / "best_position.pt", {
            **common, "head": best["head"], "epoch": best["epoch"], "selection_value": best["selection_value"],
        })
        save_checkpoint(output / "last.pt", {
            **common, "head": head.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "stale": stale, "history": history, "best": best,
            "rng": capture_rng(generator), "elapsed_seconds": total_elapsed,
        })
        write_json(output / "history.json", history)

    persist()
    budget = config.max_epochs if stop_after_epoch is None else min(config.max_epochs, stop_after_epoch)
    for epoch in range(epoch + 1, budget + 1):
        if stale >= config.patience:
            # Do not advance the saved epoch when resuming an already stopped run.
            epoch = history[-1]["epoch"]
            break
        head.train()
        order = torch.randperm(len(train["main_context"]), generator=generator)
        order_hash = hashlib.sha256(order.numpy().tobytes()).hexdigest()
        total_loss, count = 0.0, 0
        for indices in order.split(config.batch_size):
            mask = train["position_mask"][indices].bool().to(config.device)
            if not bool(mask.any()):
                continue
            context = train["main_context"][indices].detach().to(config.device)
            target = train["position_true"][indices].detach().to(config.device)
            prediction = head(context).sigmoid()
            loss = position_loss(prediction, target, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), config.gradient_clip)
            optimizer.step()
            valid_count = int(mask.sum())
            total_loss += float(loss.detach()) * valid_count
            count += valid_count
        values = metrics(predict(head, validation["main_context"]), validation["position_true"], validation["position_mask"])
        history.append({"epoch": epoch, "validation": values, "train_loss": total_loss / max(count, 1),
                        "sample_order_sha256": order_hash})
        score = selection_value(values, epoch)
        if score < tuple(best["selection_value"]):
            best = {"epoch": epoch, "selection_value": score,
                    "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        persist()
        if stale >= config.patience:
            break
    result = {"config": asdict(config), "provenance": provenance, "best_epoch": best["epoch"],
              "completed_epoch": history[-1]["epoch"], "initial_head_sha256": initial_hash,
              "validation": history[best["epoch"]]["validation"], "trainable_parameters": 130,
              "stopped_early": stale >= config.patience, "test_accessed": False}
    write_json(output / "result.json", result)
    return result


def prediction_frame(cache: dict[str, Any], prediction: torch.Tensor, seed: int) -> pd.DataFrame:
    frame = pd.DataFrame({"seed": seed, "sample_id": cache["sample_ids"],
                          "match_id": cache["match_ids"].numpy(),
                          "current_event_index": cache["current_event_indices"].numpy(),
                          "position_mask": cache["position_mask"].numpy(),
                          "player_mask": cache["player_mask"].numpy(),
                          "event_true": cache["event_true"].numpy(), "zone_true": cache["zone_true"].numpy()})
    for dim, name in enumerate(("x", "y")):
        frame[f"position_true_{name}"] = cache["position_true"][:, dim].numpy()
        frame[f"position_pred_{name}"] = prediction[:, dim].numpy()
    return frame


def train_seed(seed: int, root: Path = ROOT, device: str = "cpu", resume_from: Path | None = None,
               smoke: bool = False) -> dict[str, Any]:
    if (root / "selection/position_refit_lock.json").exists():
        raise RuntimeError("A locked experiment cannot be retrained")
    train, train_source = load_cache(seed, "train", root)
    validation, val_source = load_cache(seed, "validation", root)
    state = torch.load(source_checkpoint(seed), map_location="cpu", weights_only=False)["model"]
    head = head_from_state(state)
    provenance = {"train": train_source, "validation": val_source, "source_sha256": sha256(Path(__file__))}
    output = root / "smoke" / f"seed{seed}" if smoke else run_dir(seed, root)
    result = fit_cached(head, train, validation, RefitConfig(seed=seed, device=device), output,
                        provenance, resume_from, stop_after_epoch=1 if smoke else None)
    selected = torch.load(output / "best_position.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(selected["head"])
    for method, xy in (("original", validation["base_position_xy"]),
                       ("refit", predict(head, validation["main_context"]))):
        prediction_frame(validation, xy, seed).to_parquet(output / f"validation_{method}.parquet", index=False)
    assert sha256(source_checkpoint(seed)) == train_source["source_checkpoint_sha256"]
    return result


def paired_bootstrap(reference: pd.DataFrame, candidate: pd.DataFrame, evaluation_mask: np.ndarray,
                     replicates: int = 10000, selector_seed: int = 20260912) -> dict[str, Any]:
    """Explicit population; match clusters shared across all fixed model seeds."""
    keys = ["seed", "sample_id", "match_id", "current_event_index", "position_mask", "player_mask",
            "position_true_x", "position_true_y"]
    if not reference[keys].equals(candidate[keys]):
        raise RuntimeError("Paired positions, masks, IDs or order differ")
    mask = np.asarray(evaluation_mask)
    if mask.dtype != bool or mask.shape != (len(reference),):
        raise ValueError("An explicit aligned boolean evaluation mask is required")
    if np.any(mask & ~reference.position_mask.to_numpy(dtype=bool)):
        raise ValueError("Evaluation includes invalid positions")
    matches = np.sort(reference.match_id.unique())
    seeds = np.sort(reference.seed.unique())
    population_keys = [key for key in keys if key != "seed"]
    first_population = reference.loc[reference.seed == seeds[0], population_keys].reset_index(drop=True)
    for seed in seeds[1:]:
        if not first_population.equals(reference.loc[reference.seed == seed, population_keys].reset_index(drop=True)):
            raise RuntimeError("Model seeds do not share identical samples and targets")
    sampled = np.random.default_rng(selector_seed).integers(len(matches), size=(replicates, len(matches)))
    draws = np.zeros((replicates, len(matches)), dtype=np.int64)
    for row, indices in enumerate(sampled):
        draws[row] = np.bincount(indices, minlength=len(matches))

    def errors(frame: pd.DataFrame) -> np.ndarray:
        delta = frame[["position_pred_x", "position_pred_y"]].to_numpy(dtype=float) - frame[["position_true_x", "position_true_y"]].to_numpy(dtype=float)
        return np.linalg.norm(delta * [105.0, 68.0], axis=1)

    differences = errors(reference) - errors(candidate)
    draws_by_seed, seed_deltas = [], []
    for seed in seeds:
        active = mask & (reference.seed.to_numpy() == seed)
        if not active.any():
            raise ValueError("A seed has no evaluation samples")
        group = pd.DataFrame({"match": reference.match_id.to_numpy()[active], "delta": differences[active]})
        stats = group.groupby("match").delta.agg(["sum", "count"]).reindex(matches, fill_value=0)
        denominator = draws @ stats["count"].to_numpy(dtype=float)
        if np.any(denominator == 0):
            raise ValueError("A bootstrap draw contains no valid samples")
        draws_by_seed.append((draws @ stats["sum"].to_numpy()) / denominator)
        seed_deltas.append({"seed": int(seed), "improvement_m": float(differences[active].mean()), "samples": int(active.sum())})
    boot = np.mean(draws_by_seed, axis=0)
    mean = float(np.mean([x["improvement_m"] for x in seed_deltas]))
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    improving = sum(x["improvement_m"] > 0 for x in seed_deltas)
    return {"improvement_m": mean, "ci95": ci, "improving_seeds": improving,
            "seed_differences": seed_deltas, "resampling_unit": "match", "replicates": replicates,
            "seed_aggregation": "arithmetic mean of per-seed sample-weighted differences",
            "draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
            "practical_threshold_m": 0.25, "effective": bool(mean >= 0.25 and improving >= 2 and ci[0] > 0)}


def load_predictions(split: str, method: str, root: Path = ROOT) -> pd.DataFrame:
    if split == "test":
        require_test_lock(root)
    if split not in ("validation", "test") or method not in ("original", "refit"):
        raise ValueError("Unknown prediction split/method")
    return pd.concat([pd.read_parquet((run_dir(seed, root) if split == "validation" else root / "test" / f"seed{seed}")
                                     / f"{split}_{method}.parquet")
                      for seed in CONFIRMATION_SEEDS], ignore_index=True)


def select(root: Path = ROOT) -> Path:
    path = root / "selection/position_refit_lock.json"
    if path.exists():
        return path
    checkpoints = {}
    for seed in CONFIRMATION_SEEDS:
        output = run_dir(seed, root)
        result = json.loads((output / "result.json").read_text())
        if not (result["completed_epoch"] == 30 or result["stopped_early"]):
            raise RuntimeError("Incomplete training run")
        verification = json.loads((output / "online_verification.json").read_text())
        best_path = output / "best_position.pt"
        if (not verification["passed"] or verification["head_sha256"] != sha256(best_path)
                or verification["source_checkpoint_sha256"] != sha256(source_checkpoint(seed))):
            raise RuntimeError("Current checkpoint lacks passing five-task integration verification")
        checkpoints[str(seed)] = {"path": str(best_path), "sha256": sha256(best_path), "epoch": result["best_epoch"]}
    original, refit = load_predictions("validation", "original", root), load_predictions("validation", "refit", root)
    comparison = paired_bootstrap(original, refit, original.position_mask.to_numpy(dtype=bool))
    write_json(path, {"selection_split": "validation", "primary_population": "position_mask",
                      "comparison": comparison, "passed": comparison["effective"],
                      "selected_method": "PR-Refit" if comparison["effective"] else "PR-Original",
                      "checkpoints": checkpoints, "test_accessed": False,
                      "historical_test_results_previously_visible": True})
    return path


def evaluate_test(seed: int, root: Path = ROOT, device: str = "cpu") -> None:
    lock = require_test_lock(root)
    cache, provenance = load_cache(seed, "test", root)
    head_state = torch.load(lock["checkpoints"][str(seed)]["path"], map_location="cpu", weights_only=False)
    source = torch.load(source_checkpoint(seed), map_location="cpu", weights_only=False)["model"]
    head = head_from_state(source).to(device)
    head.load_state_dict(head_state["head"])
    output = root / "test" / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    for method, xy in (("original", cache["base_position_xy"]), ("refit", predict(head, cache["main_context"]))):
        prediction_frame(cache, xy, seed).to_parquet(output / f"test_{method}.parquet", index=False)
    write_json(output / "provenance.json", provenance)


def load_combined_model(seed: int, head_checkpoint: Path | None = None, device: str = "cpu") -> nn.Module:
    from football_benchmark.protocol import ProtocolArtifacts
    from .constants import FEASIBILITY_ARTIFACT
    from .model import build_partial_l2_model
    source = source_checkpoint(seed)
    state = torch.load(source, map_location="cpu", weights_only=False)
    with torch.random.fork_rng(devices=[]):
        model = build_partial_l2_model(ProtocolArtifacts.load(FEASIBILITY_ARTIFACT))
    model.load_state_dict(state["model"])
    if head_checkpoint is not None:
        checkpoint = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["provenance"]["validation"]["source_checkpoint_sha256"] != sha256(source):
            raise RuntimeError("Refitted Head does not belong to this source model")
        model.position_head.load_state_dict(checkpoint["head"])
    model.requires_grad_(False)
    return model.to(device).eval()


def report(root: Path = ROOT) -> Path:
    lock = json.loads((root / "selection/position_refit_lock.json").read_text())
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    rows, grouped, test_comparison = [], [], None
    splits = ["validation", "test"] if lock["passed"] else ["validation"]
    for split in splits:
        original, refit = load_predictions(split, "original", root), load_predictions(split, "refit", root)
        if split == "test":
            test_comparison = paired_bootstrap(original, refit, original.position_mask.to_numpy(dtype=bool), selector_seed=20260913)
        for method, frame in (("PR-Original", original), ("PR-Refit", refit)):
            for seed, part in frame.groupby("seed"):
                pred = torch.tensor(part[["position_pred_x", "position_pred_y"]].to_numpy())
                target = torch.tensor(part[["position_true_x", "position_true_y"]].to_numpy())
                mask = torch.tensor(part.position_mask.to_numpy(dtype=bool))
                rows.append({"split": split, "method": method, "seed": seed, **metrics(pred, target, mask)})
                for field in ("event_true", "zone_true", "player_mask"):
                    for value in sorted(part[field].unique()):
                        subset = mask & torch.tensor(part[field].to_numpy() == value)
                        if bool(subset.any()):
                            grouped.append({"split": split, "method": method, "seed": seed,
                                            "group": field, "value": int(value), **metrics(pred, target, subset)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics_by_seed.csv", index=False)
    pd.DataFrame(grouped).to_csv(output / "grouped_metrics_by_seed.csv", index=False)
    metric_names = ["mean_distance_m", "median_distance_m", "distance_rmse_m", "x_mae_m", "y_mae_m", "loss"]
    summary = frame.groupby(["split", "method"])[metric_names].agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    provenance_checks = {}
    for seed in CONFIRMATION_SEEDS:
        result = json.loads((run_dir(seed, root) / "result.json").read_text())
        for split, info in result["provenance"].items():
            if split not in ("train", "validation"):
                continue
            valid = (sha256(Path(info["cache_path"])) == info["cache_sha256"]
                     and sha256(Path(info["source_checkpoint"])) == info["source_checkpoint_sha256"])
            if not valid:
                raise RuntimeError("An original input artifact changed")
            provenance_checks[f"{seed}/{split}"] = valid
    final = {"validation_lock": lock, "test_confirmation": test_comparison,
             "test_skipped": not lock["passed"], "test_did_not_change_selection": True,
             "source_artifacts_unchanged": provenance_checks,
             "summary": [{"split": split, "method": method, **{name: {stat: float(row[(name, stat)]) for stat in ("mean", "std")} for name in metric_names}}
                         for (split, method), row in summary.iterrows()],
             "interpretation": "Additional Position-only optimization of fixed representations; not isolated proof of backbone conflict.",
             "historical_test_results_previously_visible": True,
             "environment": {"python": platform.python_version(), "torch": torch.__version__}}
    write_json(output / "report.json", final)
    lines = ["# Position Head Refit", "", "All position-valid samples; three fixed seeds.", "",
             "| Split | Method | Mean distance (m) |", "|---|---|---:|"]
    for item in final["summary"]:
        value = item["mean_distance_m"]
        lines.append(f"| {item['split']} | {item['method']} | {value['mean']:.4f} +/- {value['std']:.4f} |")
    lines += ["", f"Validation selected: {lock['selected_method']}.",
              f"Validation improvement and CI (m): {lock['comparison']['improvement_m']:.6f}, {lock['comparison']['ci95']}.",
              "Test skipped because validation failed." if not lock["passed"] else f"Test practical confirmation: {test_comparison['effective']}.",
              "", "The original Position Head is Linear(64,2) plus sigmoid. No Player condition or hidden layer is added.",
              "Prior test results were visible before this experiment; this is not a pristine holdout."]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return output / "report.json"
