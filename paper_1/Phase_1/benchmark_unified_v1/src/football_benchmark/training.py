"""Deterministic training and evaluation for one benchmark configuration."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .data import (
    CanonicalEventDataset,
    MatchBlockShuffleSampler,
    collate_hgt,
    collate_semantic_hgt,
    collate_sequence,
    load_records,
    move_batch_to_device,
)
from .constants import GRAPH_ROOT, SEMANTIC_GRAPH_ROOT
from .losses import compute_benchmark_loss
from .metrics import MetricAccumulator
from .models import ModelSpec, build_model, parameter_count
from .protocol import ProtocolArtifacts
from .sampling import TargetSamplePlan


@dataclass(frozen=True)
class TrainingConfig:
    contract: str
    family: str
    window_size: int
    learning_rate: float
    seed: int
    output_dir: Path
    device: str = "cuda:0"
    max_epochs: int = 30
    patience: int = 5
    effective_batch_size: int = 256
    micro_batch_size: int | None = None
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    num_workers: int = 0
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None
    evaluate_test: bool = True
    sample_plan_path: Path | None = None
    unified_event_loss_mode: str = "legacy_balanced"
    graph_variant: str = "legacy"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)


def _micro_batch_size(config: TrainingConfig) -> int:
    if config.micro_batch_size is not None:
        return config.micro_batch_size
    return 32 if config.family == "hgt" else config.effective_batch_size


def _loader(
    split: str,
    config: TrainingConfig,
    artifacts: ProtocolArtifacts,
    shuffle: bool,
) -> DataLoader:
    graph_root = SEMANTIC_GRAPH_ROOT if config.graph_variant == "semantic_v2" else GRAPH_ROOT
    records = load_records(split, graph_root=graph_root)
    selected_currents = None
    if config.sample_plan_path is not None:
        plan = TargetSamplePlan.load(config.sample_plan_path)
        selected_currents = plan.currents_by_match(split)
    limit = {
        "train": config.max_train_samples,
        "validation": config.max_validation_samples,
        "test": config.max_test_samples,
    }[split]
    dataset = CanonicalEventDataset(
        records,
        artifacts,
        config.window_size,
        max_samples=limit,
        selected_currents=selected_currents,
    )
    if config.family == "hgt":
        collate = (
            partial(
                collate_semantic_hgt,
                artifacts=artifacts,
                window_size=config.window_size,
            )
            if config.graph_variant == "semantic_v2"
            else partial(collate_hgt, artifacts=artifacts)
        )
    else:
        collate = partial(
            collate_sequence,
            artifacts=artifacts,
            padded_width=config.window_size,
        )
    generator = torch.Generator().manual_seed(config.seed)
    sampler = MatchBlockShuffleSampler(dataset, config.seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=_micro_batch_size(config),
        shuffle=False,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate,
        generator=generator,
        persistent_workers=config.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    artifacts: ProtocolArtifacts,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    micro_batch = _micro_batch_size(config)
    accumulation = max(1, int(np.ceil(config.effective_batch_size / micro_batch)))
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {"loss": 0.0}
    count = 0
    pending = 0
    for batch_index, raw_batch in enumerate(loader):
        batch = move_batch_to_device(raw_batch, device)
        predictions = model(batch)
        loss, components = compute_benchmark_loss(
            predictions,
            batch,
            config.family,
            config.contract,
            artifacts,
            unified_event_loss_mode=config.unified_event_loss_mode,
        )
        (loss / accumulation).backward()
        pending += 1
        batch_size = len(batch["sample_ids"])
        sums["loss"] += float(loss.detach()) * batch_size
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach()) * batch_size
        count += batch_size
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
    model: torch.nn.Module,
    loader: DataLoader,
    config: TrainingConfig,
    artifacts: ProtocolArtifacts,
    device: torch.device,
    disabled_relation_families: tuple[str, ...] = (),
) -> tuple[dict[str, Any], Any]:
    model.eval()
    accumulator = MetricAccumulator(config.family, config.contract, artifacts)
    for raw_batch in loader:
        if disabled_relation_families:
            raw_batch["disabled_relation_families"] = disabled_relation_families
        batch = move_batch_to_device(raw_batch, device)
        predictions = model(batch)
        loss, _ = compute_benchmark_loss(
            predictions,
            batch,
            config.family,
            config.contract,
            artifacts,
            unified_event_loss_mode=config.unified_event_loss_mode,
        )
        accumulator.update(predictions, batch, loss)
    return accumulator.compute(), accumulator.frame()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    config: TrainingConfig,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state


def run_training(
    config: TrainingConfig, artifacts: ProtocolArtifacts
) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    _reset_cuda_peak(device)
    spec = ModelSpec(
        family=config.family,
        contract=config.contract,
        window_size=config.window_size,
        graph_variant=config.graph_variant,
    )
    model = build_model(spec, artifacts).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_loader = _loader("train", config, artifacts, shuffle=True)
    validation_loader = _loader("validation", config, artifacts, shuffle=False)
    checkpoint_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    stale_epochs = 0
    started = time.monotonic()
    for epoch in range(1, config.max_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, config, artifacts, device
        )
        validation_metrics, _ = evaluate(
            model, validation_loader, config, artifacts, device
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        validation_loss = float(validation_metrics["loss"])
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch, validation_loss, config
            )
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    best_state = load_checkpoint(checkpoint_path, model)
    validation_metrics, validation_predictions = evaluate(
        model, validation_loader, config, artifacts, device
    )
    validation_predictions.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    test_metrics = None
    if config.evaluate_test:
        test_loader = _loader("test", config, artifacts, shuffle=False)
        test_metrics, test_predictions = evaluate(
            model, test_loader, config, artifacts, device
        )
        test_predictions.to_parquet(output_dir / "test_predictions.parquet", index=False)
    result = {
        "config": asdict(config),
        "parameters": parameter_count(model),
        "best_epoch": int(best_state["epoch"]),
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "protocol_metadata": artifacts.metadata,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=True, indent=2, default=_json_default)
    return result


def evaluate_checkpoint(
    config: TrainingConfig,
    artifacts: ProtocolArtifacts,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Evaluate a selected validation checkpoint without retraining it."""

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    _reset_cuda_peak(device)
    state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    source_config = state.get("config", {})
    expected = {
        "contract": config.contract,
        "family": config.family,
        "window_size": config.window_size,
        "seed": config.seed,
        "unified_event_loss_mode": config.unified_event_loss_mode,
        "graph_variant": config.graph_variant,
    }
    for name, value in expected.items():
        fallback = {
            "unified_event_loss_mode": "legacy_balanced",
            "graph_variant": "legacy",
        }.get(name)
        source_value = source_config.get(name, fallback)
        if source_value != value:
            raise ValueError(
                f"Checkpoint {name}={source_value!r}, expected {value!r}"
            )
    spec = ModelSpec(
        family=config.family,
        contract=config.contract,
        window_size=config.window_size,
        graph_variant=config.graph_variant,
    )
    model = build_model(spec, artifacts).to(device)
    model.load_state_dict(state["model"])

    started = time.monotonic()
    validation_loader = _loader("validation", config, artifacts, shuffle=False)
    validation_metrics, validation_predictions = evaluate(
        model, validation_loader, config, artifacts, device
    )
    validation_predictions.to_parquet(
        output_dir / "validation_predictions.parquet", index=False
    )
    test_metrics = None
    if config.evaluate_test:
        test_loader = _loader("test", config, artifacts, shuffle=False)
        test_metrics, test_predictions = evaluate(
            model, test_loader, config, artifacts, device
        )
        test_predictions.to_parquet(
            output_dir / "test_predictions.parquet", index=False
        )
    result = {
        "config": asdict(config),
        "parameters": parameter_count(model),
        "best_epoch": int(state["epoch"]),
        "elapsed_seconds": time.monotonic() - started,
        "history": [],
        "validation": validation_metrics,
        "test": test_metrics,
        "protocol_metadata": artifacts.metadata,
        "source_checkpoint": str(Path(checkpoint_path).resolve()),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=True, indent=2, default=_json_default)
    return result
