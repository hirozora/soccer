"""Deterministic single-task and joint training for Version 4."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from football_benchmark.data import (
    CanonicalEventDataset,
    MatchBlockShuffleSampler,
    collate_semantic_hgt,
    load_records,
)
from football_benchmark.protocol import ProtocolArtifacts
from football_benchmark.sampling import TargetSamplePlan

from .constants import POSSESSION_GRAPH_ROOT, SEMANTIC_GRAPH_ROOT, WINDOW_SIZE
from .diagnostics import HEAD_PREFIXES, compute_gradient_diagnostics
from .losses import compute_loss
from .metrics import PredictionAccumulator
from .model import (
    MULTIVIEW_MODES,
    TargetStudyHGT,
    TaskViewFusionHGT,
    build_multiview_model,
    build_target_model,
)
from .possession_data import (
    collate_multiview_possession_hgt,
    collate_possession_hgt,
)
from .subgraph_views import DEFAULT_SELECTOR_SEED


@dataclass(frozen=True)
class TargetTrainingConfig:
    task: str
    method: str
    artifact_path: Path
    output_dir: Path
    learning_rate: float
    seed: int
    device: str
    max_epochs: int
    patience: int
    sample_plan_path: Path | None = None
    joint_methods: dict[str, str] | None = None
    joint_loss_weights: dict[str, float] | None = None
    effective_batch_size: int = 256
    micro_batch_size: int = 256
    num_workers: int = 2
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    evaluate_test: bool = False
    full_test: bool = False
    max_test_samples: int | None = None
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    graph_variant: str = "semantic_v2"
    possession_topology: str = "none"
    possession_feature_level: str = "topology"
    snapshot_scope: str = "selected_events"
    context_view: str = "f80"
    selector_seed: int = DEFAULT_SELECTOR_SEED
    fusion_mode: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader_worker_init(_: int) -> None:
    """Keep each loader process single-threaded to avoid CPU oversubscription."""

    torch.set_num_threads(1)


def _move_batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Transfer graph tensors asynchronously when pinned memory is available."""

    result = dict(batch)
    if "graph" in result:
        result["graph"] = result["graph"].to(device, non_blocking=True)
    if "graphs" in result:
        result["graphs"] = {
            view: graph.to(device, non_blocking=True)
            for view, graph in result["graphs"].items()
        }
    if "sequence" in result:
        result["sequence"] = {
            key: value.to(device, non_blocking=True)
            for key, value in result["sequence"].items()
        }
        result["valid_mask"] = result["valid_mask"].to(device, non_blocking=True)
    result["targets"] = {
        key: value.to(device, non_blocking=True)
        for key, value in result["targets"].items()
    }
    result["match_ids"] = result["match_ids"].to(device, non_blocking=True)
    result["current_event_indices"] = result["current_event_indices"].to(
        device, non_blocking=True
    )
    if "anchor_state" in result:
        result["anchor_state"] = result["anchor_state"].to(
            device, non_blocking=True
        )
    if "rf_metadata" in result:
        def move(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(device, non_blocking=True)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            return value

        result["rf_metadata"] = move(result["rf_metadata"])
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _source_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def backbone_state_hash(model: TargetStudyHGT) -> str:
    """Hash only the shared encoder, HGT layers, and context projection."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(HEAD_PREFIXES):
            continue
        if name.startswith("fusion_logits."):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loader(
    split: str,
    config: TargetTrainingConfig,
    artifacts: ProtocolArtifacts,
    shuffle: bool,
) -> DataLoader:
    if config.num_workers > 0:
        # PyG batches contain many tensors. File-system sharing avoids exhausting
        # ancillary file-descriptor transfers in long-lived worker queues.
        torch.multiprocessing.set_sharing_strategy("file_system")
    graph_root = (
        POSSESSION_GRAPH_ROOT
        if config.graph_variant == "semantic_v3_possession"
        else SEMANTIC_GRAPH_ROOT
    )
    records = load_records(split, graph_root=graph_root)
    selected = None
    if config.sample_plan_path is not None and not (split == "test" and config.full_test):
        selected = TargetSamplePlan.load(config.sample_plan_path).currents_by_match(split)
    limit = {
        "train": config.max_train_samples,
        "validation": config.max_validation_samples,
        "test": config.max_test_samples,
    }[split]
    dataset = CanonicalEventDataset(
        records,
        artifacts,
        WINDOW_SIZE,
        max_samples=limit,
        selected_currents=selected,
    )
    sampler = MatchBlockShuffleSampler(dataset, config.seed) if shuffle else None
    if config.fusion_mode is not None:
        if config.graph_variant != "semantic_v3_possession":
            raise ValueError("Multi-view fusion requires Semantic V3 Possession")
        if config.possession_topology != "membership":
            raise ValueError("Multi-view fusion requires membership topology")
        # Required views are a pure function of mode; avoid constructing a model
        # in worker setup merely to discover them.
        required = {
            "f80": ("f80",),
            "fixed_a": ("f80", "p1", "p2"),
            "fixed_b": ("p1", "p2"),
            "sf_a": ("f80", "p1", "p2"),
            "sf_b": ("p1", "p2"),
            "recency_sf_b": ("lp1", "lp2"),
        }[config.fusion_mode]
        collate = partial(
            collate_multiview_possession_hgt,
            artifacts=artifacts,
            window_size=WINDOW_SIZE,
            topology=config.possession_topology,
            feature_level=config.possession_feature_level,
            snapshot_scope=config.snapshot_scope,
            context_views=required,
            selector_seed=config.selector_seed,
        )
    else:
        collate = (
            partial(
                collate_possession_hgt,
                artifacts=artifacts,
                window_size=WINDOW_SIZE,
                topology=config.possession_topology,
                feature_level=config.possession_feature_level,
                snapshot_scope=config.snapshot_scope,
                context_view=config.context_view,
                selector_seed=config.selector_seed,
            )
            if config.graph_variant == "semantic_v3_possession"
            else partial(
                collate_semantic_hgt, artifacts=artifacts, window_size=WINDOW_SIZE
            )
        )
    loader_options: dict[str, Any] = {}
    if config.num_workers > 0:
        loader_options.update(
            prefetch_factor=2,
            worker_init_fn=_loader_worker_init,
        )
    return DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        generator=torch.Generator().manual_seed(config.seed),
        drop_last=False,
        **loader_options,
    )


def _selection_value(task: str, metrics: dict[str, Any]) -> tuple[float, bool]:
    if task == "event":
        return float(metrics["event"]["macro_f1"]), True
    if task == "time":
        return float(metrics["time"]["mae_seconds"]), False
    if task == "position":
        return float(metrics["position"]["distance_mae_m"]), False
    return float(metrics["loss"]), False


def _is_better(value: float, best: float | None, maximize: bool) -> bool:
    return best is None or (value > best + 1e-12 if maximize else value < best - 1e-12)


def train_epoch(
    model: TargetStudyHGT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TargetTrainingConfig,
    artifacts: ProtocolArtifacts,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    accumulation = max(1, int(np.ceil(config.effective_batch_size / config.micro_batch_size)))
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {"loss": 0.0}
    count = 0
    pending = 0
    for batch_index, raw_batch in enumerate(loader):
        batch = _move_batch_to_device(raw_batch, device)
        predictions = model(batch)
        loss, components = compute_loss(
            predictions,
            batch,
            config.task,
            config.method,
            artifacts,
            config.joint_methods,
            config.joint_loss_weights,
        )
        (loss / accumulation).backward()
        pending += 1
        size = len(batch["sample_ids"])
        sums["loss"] += float(loss.detach()) * size
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach()) * size
        count += size
        if pending == accumulation or batch_index + 1 == len(loader):
            if pending < accumulation:
                correction = accumulation / pending
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
    return {name: value / max(count, 1) for name, value in sums.items()}


@torch.no_grad()
def evaluate(
    model: TargetStudyHGT,
    loader: DataLoader,
    config: TargetTrainingConfig,
    artifacts: ProtocolArtifacts,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    accumulator = PredictionAccumulator(config.task)
    for raw_batch in loader:
        batch = _move_batch_to_device(raw_batch, device)
        predictions = model(batch)
        loss, _ = compute_loss(
            predictions,
            batch,
            config.task,
            config.method,
            artifacts,
            config.joint_methods,
            config.joint_loss_weights,
        )
        accumulator.update(predictions, batch, loss)
    return accumulator.compute(), accumulator.frame()


def _write_metric_tables(metrics: dict[str, Any], output_dir: Path) -> None:
    if "event" in metrics:
        pd.DataFrame(metrics["event"]["confusion_matrix"], index=range(10), columns=range(10)).to_csv(
            output_dir / "event_confusion_matrix.csv"
        )
        pd.DataFrame.from_dict(metrics["event"]["per_class"], orient="index").to_csv(
            output_dir / "event_per_class.csv", index_label="event"
        )
    if "time" in metrics:
        pd.DataFrame.from_dict(metrics["time"]["by_interval"], orient="index").to_csv(
            output_dir / "time_by_interval.csv", index_label="interval"
        )
        pd.DataFrame.from_dict(metrics["time"]["by_event"], orient="index").to_csv(
            output_dir / "time_by_event.csv", index_label="event"
        )
    if "position" in metrics:
        pd.DataFrame.from_dict(metrics["position"]["by_zone"], orient="index").to_csv(
            output_dir / "position_by_zone.csv", index_label="zone"
        )
        pd.DataFrame.from_dict(metrics["position"]["by_event"], orient="index").to_csv(
            output_dir / "position_by_event.csv", index_label="event"
        )


def run_training(config: TargetTrainingConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = ProtocolArtifacts.load(config.artifact_path)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    if config.fusion_mode is not None:
        if config.fusion_mode not in MULTIVIEW_MODES:
            raise ValueError(f"Unknown fusion mode {config.fusion_mode!r}")
        model = build_multiview_model(artifacts, config.fusion_mode).to(device)
    else:
        model = build_target_model(
            artifacts,
            config.task,
            config.method,
            config.joint_methods,
            graph_variant=config.graph_variant,
            possession_topology=config.possession_topology,
            possession_feature_level=config.possession_feature_level,
        ).to(device)
    initial_backbone_sha256 = backbone_state_hash(model)
    fusion_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("fusion_logits.")
    ]
    fusion_ids = {id(parameter) for parameter in fusion_parameters}
    base_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in fusion_ids
    ]
    parameter_groups: list[dict[str, Any]] = [
        {"params": base_parameters, "weight_decay": config.weight_decay}
    ]
    if fusion_parameters:
        parameter_groups.append({"params": fusion_parameters, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(parameter_groups, lr=config.learning_rate)
    train_loader = _loader("train", config, artifacts, True)
    validation_loader = _loader("validation", config, artifacts, False)
    diagnostic_batch = None
    initial_gradient_diagnostics = None
    if config.task == "joint":
        diagnostic_batch = _move_batch_to_device(next(iter(validation_loader)), device)
        initial_gradient_diagnostics = compute_gradient_diagnostics(
            model,
            diagnostic_batch,
            artifacts,
            config.method,
            config.joint_methods or {},
            config.joint_loss_weights
            or {"event": 1.0, "time": 1.0, "position": 1.0},
        )
    checkpoint = output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    best: float | None = None
    stale = 0
    started = time.monotonic()
    for epoch in range(1, config.max_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, config, artifacts, device)
        validation_metrics, _ = evaluate(model, validation_loader, config, artifacts, device)
        value, maximize = _selection_value(config.task, validation_metrics)
        epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "selection_value": value,
        }
        if isinstance(model, TaskViewFusionHGT):
            model.eval()
            with torch.no_grad():
                _, diagnostic_contexts = model.forward_with_contexts(diagnostic_batch)
            epoch_record["fusion"] = {
                "logits": {
                    task: logits.detach().cpu().tolist()
                    for task, logits in model.fusion_logits.items()
                },
                "weights": {
                    task: weights.detach().cpu().tolist()
                    for task, weights in model.fusion_weights().items()
                },
                "context": model.context_diagnostics(diagnostic_contexts),
            }
        history.append(epoch_record)
        if _is_better(value, best, maximize):
            best = value
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "selection_value": value,
                    "config": asdict(config),
                    "graph_variant": config.graph_variant,
                    "window_size": WINDOW_SIZE,
                    "source_sha256": _source_hash(),
                    "initial_backbone_sha256": initial_backbone_sha256,
                    "fusion_mode": config.fusion_mode,
                },
                checkpoint,
            )
        else:
            stale += 1
        if stale >= config.patience:
            break
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    best_gradient_diagnostics = None
    if diagnostic_batch is not None:
        best_gradient_diagnostics = compute_gradient_diagnostics(
            model,
            diagnostic_batch,
            artifacts,
            config.method,
            config.joint_methods or {},
            config.joint_loss_weights
            or {"event": 1.0, "time": 1.0, "position": 1.0},
        )
    validation_metrics, validation_predictions = evaluate(
        model, validation_loader, config, artifacts, device
    )
    validation_predictions.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    _write_metric_tables(validation_metrics, output_dir)
    test_metrics = None
    if config.evaluate_test:
        test_loader = _loader("test", config, artifacts, False)
        test_metrics, test_predictions = evaluate(model, test_loader, config, artifacts, device)
        test_predictions.to_parquet(output_dir / "test_predictions.parquet", index=False)
    result = {
        "config": asdict(config),
        "best_epoch": int(state["epoch"]),
        "selection_value": float(state["selection_value"]),
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "source_sha256": _source_hash(),
        "initial_backbone_sha256": initial_backbone_sha256,
        "gradient_diagnostics": {
            "initial": initial_gradient_diagnostics,
            "best": best_gradient_diagnostics,
        },
        "fusion": (
            {
                "logits": {
                    task: logits.detach().cpu().tolist()
                    for task, logits in model.fusion_logits.items()
                },
                "weights": {
                    task: weights.detach().cpu().tolist()
                    for task, weights in model.fusion_weights().items()
                },
            }
            if isinstance(model, TaskViewFusionHGT)
            else None
        ),
        "test_accessed": config.evaluate_test,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    return result


def evaluate_checkpoint(config: TargetTrainingConfig, checkpoint: Path) -> dict[str, Any]:
    started = time.monotonic()
    set_seed(config.seed)
    artifacts = ProtocolArtifacts.load(config.artifact_path)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("graph_variant") != config.graph_variant or state.get("window_size") != WINDOW_SIZE:
        raise ValueError("Checkpoint graph variant or window size does not match Version 4")
    checkpoint_config = state.get("config", {})
    if checkpoint_config.get("task") != config.task or checkpoint_config.get("method") != config.method:
        raise ValueError("Checkpoint task/method does not match evaluation configuration")
    for field in (
        "graph_variant",
        "possession_topology",
        "possession_feature_level",
        "snapshot_scope",
        "context_view",
        "selector_seed",
        "fusion_mode",
    ):
        expected = checkpoint_config.get(field, getattr(TargetTrainingConfig, field, None))
        if expected != getattr(config, field):
            raise ValueError(f"Checkpoint {field} does not match evaluation configuration")
    model = (
        build_multiview_model(artifacts, config.fusion_mode)
        if config.fusion_mode is not None
        else build_target_model(
            artifacts,
            config.task,
            config.method,
            config.joint_methods,
            graph_variant=config.graph_variant,
            possession_topology=config.possession_topology,
            possession_feature_level=config.possession_feature_level,
        )
    ).to(device)
    model.load_state_dict(state["model"])
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = "test" if config.evaluate_test else "validation"
    metrics, predictions = evaluate(
        model, _loader(split, config, artifacts, False), config, artifacts, device
    )
    predictions.to_parquet(output_dir / f"{split}_predictions.parquet", index=False)
    _write_metric_tables(metrics, output_dir)
    result = {
        "config": asdict(config),
        "evaluation_only": True,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _file_hash(checkpoint),
        "best_epoch": int(state["epoch"]),
        "validation": metrics if split == "validation" else None,
        "test": metrics if split == "test" else None,
        "elapsed_seconds": time.monotonic() - started,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "source_sha256": _source_hash(),
        "fusion": (
            {
                "logits": {
                    task: logits.detach().cpu().tolist()
                    for task, logits in model.fusion_logits.items()
                },
                "weights": {
                    task: weights.detach().cpu().tolist()
                    for task, weights in model.fusion_weights().items()
                },
            }
            if isinstance(model, TaskViewFusionHGT)
            else None
        ),
        "test_accessed": split == "test",
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    return result
