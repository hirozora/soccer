"""Fixed-budget private-branch training for causal Player histories."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from football_benchmark.data import CanonicalEventDataset, MatchBlockShuffleSampler, load_records
from football_benchmark.protocol import ProtocolArtifacts
from football_benchmark.sampling import TargetSamplePlan

from .actor_training import _prediction_frame, compute_actor_metrics, player_cross_entropy
from .constants import FEASIBILITY_ARTIFACT, POSSESSION_GRAPH_ROOT, SAMPLE_PLAN, WINDOW_SIZE
from .model import build_player_history_model
from .player_history_data import collate_player_history_hgt
from .player_history_study import CONDITIONS, SELECTOR_SEED, decision_path, source_checkpoint
from .training import _move_batch_to_device, set_seed


@dataclass(frozen=True)
class PlayerHistoryTrainingConfig:
    condition: str
    output_dir: Path
    seed: int
    device: str
    max_epochs: int = 24
    batch_size: int = 256
    num_workers: int = 2
    learning_rate: float = 9e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    artifact_path: Path = FEASIBILITY_ARTIFACT
    sample_plan_path: Path | None = SAMPLE_PLAN
    selector_seed: int = SELECTOR_SEED
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None
    evaluate_test: bool = False
    full_test: bool = False

    def validate(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {self.condition}")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if self.full_test and not self.evaluate_test:
            raise ValueError("full_test requires evaluate_test")


def _worker_init(_: int) -> None:
    torch.set_num_threads(1)


def _loader(
    split: str,
    config: PlayerHistoryTrainingConfig,
    artifacts: ProtocolArtifacts,
    shuffle: bool,
) -> DataLoader:
    selected = None
    if config.sample_plan_path is not None and not (split == "test" and config.full_test):
        selected = TargetSamplePlan.load(config.sample_plan_path).currents_by_match(split)
    limit = {
        "train": config.max_train_samples,
        "validation": config.max_validation_samples,
        "test": config.max_test_samples,
    }[split]
    dataset = CanonicalEventDataset(
        load_records(split, graph_root=POSSESSION_GRAPH_ROOT),
        artifacts,
        WINDOW_SIZE,
        max_samples=limit,
        selected_currents=selected,
    )
    sampler = MatchBlockShuffleSampler(dataset, config.seed) if shuffle else None
    options: dict[str, Any] = {}
    if config.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
        options.update(
            prefetch_factor=2,
            persistent_workers=True,
            worker_init_fn=_worker_init,
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(
            collate_player_history_hgt,
            artifacts=artifacts,
            window_size=WINDOW_SIZE,
            condition=config.condition,
            selector_seed=config.selector_seed,
        ),
        generator=torch.Generator().manual_seed(config.seed),
        drop_last=False,
        **options,
    )


def _tensor_hash(model: torch.nn.Module, *, trainable_only: bool) -> str:
    digest = hashlib.sha256()
    parameters = dict(model.named_parameters())
    for name, value in sorted(model.state_dict().items()):
        if trainable_only and (name not in parameters or not parameters[name].requires_grad):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_source_model(
    artifacts: ProtocolArtifacts, seed: int, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    source = torch.load(source_checkpoint(seed), map_location="cpu", weights_only=False)
    model = build_player_history_model(artifacts)
    incompatible = model.load_state_dict(source["model"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected source keys: {incompatible.unexpected_keys}")
    if not incompatible.missing_keys or any(
        not name.startswith(("player_history_encoder.", "player_history_adapter."))
        for name in incompatible.missing_keys
    ):
        raise RuntimeError(f"Unexpected missing source keys: {incompatible.missing_keys}")
    model.configure_trainable_private_branch()
    return model.to(device), source


def _actor_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {**batch, "graph": batch["graphs"]["f80"]}


def _move_player_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = _move_batch_to_device(batch, device)
    result["player_history"] = {
        name: value.to(device, non_blocking=True)
        for name, value in batch["player_history"].items()
    }
    return result


def _loss(predictions: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
    graph = batch["graphs"]["f80"]
    return player_cross_entropy(
        predictions["player_scores"],
        graph["player"].ptr,
        batch["targets"]["player_local"],
        batch["targets"]["player_mask"].bool(),
    )


def _enrich_frame(frame: pd.DataFrame, batch: dict[str, Any]) -> pd.DataFrame:
    history = batch["player_history"]
    frame["target_team_mapping_valid"] = history["target_team_mapping_valid"].detach().cpu().numpy()
    frame["target_shuffle_eligible"] = history["target_shuffle_eligible"].detach().cpu().numpy()
    frame["target_history_count"] = history["target_history_count"].detach().cpu().numpy()
    frame["target_recency_seconds"] = history["target_recency_seconds"].detach().cpu().numpy()
    frame["target_current_possession_participated"] = history[
        "target_current_possession_participated"
    ].detach().cpu().numpy()
    return frame


@torch.no_grad()
def evaluate_player_history(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    frames, loss_sum, count = [], 0.0, 0
    for raw_batch in loader:
        batch = _move_player_batch(raw_batch, device)
        predictions = model(batch)
        loss = _loss(predictions, batch)
        frame = _prediction_frame(predictions, _actor_batch(batch), "player")
        frames.append(_enrich_frame(frame, batch))
        size = len(batch["sample_ids"])
        loss_sum += float(loss) * size
        count += size
    frame = pd.concat(frames, ignore_index=True)
    return compute_actor_metrics(frame, "player", loss_sum / max(count, 1)), frame


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: PlayerHistoryTrainingConfig,
    device: torch.device,
) -> float:
    model.train()
    total, count = 0.0, 0
    for raw_batch in loader:
        batch = _move_player_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model(batch), batch)
        loss.backward()
        clip_grad_norm_((p for p in model.parameters() if p.requires_grad), config.gradient_clip)
        optimizer.step()
        size = len(batch["sample_ids"])
        total += float(loss.detach()) * size
        count += size
    return total / max(count, 1)


def _selection(metrics: dict[str, Any]) -> tuple[float, float, float]:
    player = metrics["player"]
    return (
        float(player["top1_accuracy"]),
        float(player["mrr"]),
        -float(metrics["loss"]),
    )


def run_player_history_training(config: PlayerHistoryTrainingConfig) -> dict[str, Any]:
    config.validate()
    set_seed(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = ProtocolArtifacts.load(config.artifact_path)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model, source = _load_source_model(artifacts, config.seed, device)
    trainable_hash = _tensor_hash(model, trainable_only=True)
    frozen_hash = _tensor_hash(model, trainable_only=False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _loader("train", config, artifacts, True)
    validation_loader = _loader("validation", config, artifacts, False)

    diagnostic = _move_player_batch(next(iter(validation_loader)), device)
    model.eval()
    with torch.no_grad():
        predictions, baseline_scores = model.forward_with_base_scores(diagnostic)
        history_scores = predictions["player_scores"]
        maximum_initial_difference = float((history_scores - baseline_scores).abs().max())
    if maximum_initial_difference >= 1e-6:
        raise RuntimeError(f"Player-history initialization mismatch: {maximum_initial_difference}")

    history: list[dict[str, Any]] = []
    best_value = (-float("inf"), -float("inf"), -float("inf"))
    checkpoint = output / "best_player.pt"
    started = time.monotonic()
    for epoch in range(1, config.max_epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, config, device)
        validation, _ = evaluate_player_history(model, validation_loader, device)
        value = _selection(validation)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation,
            "selection_value": value,
        })
        if value > best_value:
            best_value = value
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": asdict(config),
                "selection_value": value,
                "initial_trainable_sha256": trainable_hash,
                "initial_frozen_sha256": frozen_hash,
            }, checkpoint)

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    validation, predictions = evaluate_player_history(model, validation_loader, device)
    predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    result = {
        "config": asdict(config),
        "best_epoch": int(state["epoch"]),
        "validation": validation,
        "test": None,
        "history": history,
        "initial_trainable_sha256": trainable_hash,
        "initial_frozen_sha256": frozen_hash,
        "maximum_initial_score_difference": maximum_initial_difference,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "test_accessed": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def evaluate_player_history_checkpoint(
    config: PlayerHistoryTrainingConfig,
    checkpoint: Path,
) -> dict[str, Any]:
    config.validate()
    if not decision_path().exists():
        raise RuntimeError("Validation Player-history decision must be locked before test")
    set_seed(config.seed)
    artifacts = ProtocolArtifacts.load(config.artifact_path)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, _ = _load_source_model(artifacts, config.seed, device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test_loader = _loader("test", config, artifacts, False)
    metrics, predictions = evaluate_player_history(model, test_loader, device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "test_predictions.parquet", index=False)
    result = {
        "config": asdict(config),
        "best_epoch": int(state["epoch"]),
        "validation": None,
        "test": metrics,
        "test_accessed": True,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result
