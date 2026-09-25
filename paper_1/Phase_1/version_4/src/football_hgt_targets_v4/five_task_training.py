"""Training and evaluation for the controlled five-task view experiment."""

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

from .actor_training import _prediction_frame, compute_actor_metrics
from .constants import FEASIBILITY_ARTIFACT, POSSESSION_GRAPH_ROOT, SAMPLE_PLAN, WINDOW_SIZE
from .five_task_data import collate_five_task_hgt
from .receptive_field import collate_receptive_field_hgt
from .five_task_diagnostics import FIVE_TASK_HEAD_PREFIXES, five_task_gradient_diagnostics
from .five_task_loss import five_task_loss
from .five_task_study import FIVE_TASK_MODES
from .metrics import PredictionAccumulator
from .model import FiveTaskViewHGT, build_five_task_model
from .subgraph_views import DEFAULT_SELECTOR_SEED
from .training import _move_batch_to_device, _write_metric_tables, set_seed


@dataclass(frozen=True)
class FiveTaskTrainingConfig:
    mode: str
    output_dir: Path
    seed: int
    device: str
    learning_rate: float = 9e-4
    max_epochs: int = 8
    patience: int = 2
    batch_size: int = 256
    num_workers: int = 2
    artifact_path: Path = FEASIBILITY_ARTIFACT
    sample_plan_path: Path | None = SAMPLE_PLAN
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None
    evaluate_test: bool = False
    full_test: bool = False
    selector_seed: int = DEFAULT_SELECTOR_SEED
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    receptive_field_configuration: str | None = None

    def validate(self) -> None:
        if self.mode not in FIVE_TASK_MODES:
            raise ValueError(f"Unknown five-task mode {self.mode!r}")
        if self.full_test and not self.evaluate_test:
            raise ValueError("full_test requires evaluate_test")


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _state_hash(model: FiveTaskViewHGT, *, backbone_only: bool) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("fusion_logits."):
            continue
        if backbone_only and name.startswith(FIVE_TASK_HEAD_PREFIXES):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def five_task_common_initialization_hash(model: FiveTaskViewHGT) -> str:
    """Hash all parameters shared by F80, Hard, and Soft."""

    return _state_hash(model, backbone_only=False)


def five_task_backbone_hash(model: FiveTaskViewHGT) -> str:
    return _state_hash(model, backbone_only=True)


def _worker_init(_: int) -> None:
    torch.set_num_threads(1)


def _required_views(mode: str) -> tuple[str, ...]:
    return ("f80",) if mode == "five_f80" else ("f80", "p1", "p2")


def _loader(
    split: str,
    config: FiveTaskTrainingConfig,
    artifacts: ProtocolArtifacts,
    shuffle: bool,
    *,
    batch_size: int | None = None,
) -> DataLoader:
    if config.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
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
        options.update(prefetch_factor=2, persistent_workers=True, worker_init_fn=_worker_init)
    return DataLoader(
        dataset,
        batch_size=batch_size or config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=(
            partial(
                collate_receptive_field_hgt,
                artifacts=artifacts,
                window_size=WINDOW_SIZE,
                configuration=config.receptive_field_configuration,
                selector_seed=config.selector_seed,
            )
            if config.receptive_field_configuration is not None
            else partial(
                collate_five_task_hgt,
                artifacts=artifacts,
                window_size=WINDOW_SIZE,
                context_views=_required_views(config.mode),
                selector_seed=config.selector_seed,
            )
        ),
        generator=torch.Generator().manual_seed(config.seed),
        drop_last=False,
        **options,
    )


def _merge_prediction_frames(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    loss: torch.Tensor,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    accumulator = PredictionAccumulator("joint")
    accumulator.update(predictions, batch, loss)
    frame = accumulator.frame()
    actor_batch = {**batch, "graph": batch["graphs"]["f80"]}
    team = _prediction_frame(predictions, actor_batch, "team")
    player = _prediction_frame(predictions, actor_batch, "player")
    for candidate in (team, player):
        if candidate.sample_id.tolist() != frame.sample_id.tolist():
            raise RuntimeError("Five-task prediction rows are not aligned")
    team_columns = [name for name in team if name.startswith("team_")]
    player_columns = [
        name for name in player
        if name.startswith("player_") or name == "candidate_count"
    ]
    return pd.concat(
        (frame.reset_index(drop=True), team[team_columns], player[player_columns]), axis=1
    ), accumulator.compute()


@torch.no_grad()
def evaluate_five_task(
    model: FiveTaskViewHGT,
    loader: DataLoader,
    artifacts: ProtocolArtifacts,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    frames: list[pd.DataFrame] = []
    total_loss = 0.0
    component_sums = {name: 0.0 for name in ("event", "time", "position", "team", "player")}
    count = 0
    for raw_batch in loader:
        batch = _move_batch_to_device(raw_batch, device)
        predictions = model(batch)
        loss, components = five_task_loss(predictions, batch, artifacts)
        frame, _ = _merge_prediction_frames(predictions, batch, loss)
        frames.append(frame)
        size = len(batch["sample_ids"])
        total_loss += float(loss) * size
        for name, value in components.items():
            component_sums[name] += float(value) * size
        count += size
    frame = pd.concat(frames, ignore_index=True)
    from .metrics import compute_metrics

    metrics = compute_metrics(frame, total_loss / max(count, 1))
    team = compute_actor_metrics(frame, "team", component_sums["team"] / max(count, 1))
    player = compute_actor_metrics(frame, "player", component_sums["player"] / max(count, 1))
    metrics["team"] = team["team"]
    metrics["player"] = player["player"]
    metrics["task_losses"] = {
        name: value / max(count, 1) for name, value in component_sums.items()
    }
    return metrics, frame


def _train_epoch(
    model: FiveTaskViewHGT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    artifacts: ProtocolArtifacts,
    config: FiveTaskTrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    sums = {name: 0.0 for name in ("loss", "event", "time", "position", "team", "player")}
    count = 0
    for raw_batch in loader:
        batch = _move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, components = five_task_loss(model(batch), batch, artifacts)
        loss.backward()
        clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        size = len(batch["sample_ids"])
        sums["loss"] += float(loss.detach()) * size
        for name, value in components.items():
            sums[name] += float(value.detach()) * size
        count += size
    return {name: value / max(count, 1) for name, value in sums.items()}


def _fusion_record(model: FiveTaskViewHGT, batch: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        _, contexts = model.forward_with_contexts(batch)
    return {
        "logits": {name: value.detach().cpu().tolist() for name, value in model.fusion_logits.items()},
        "weights": {name: value.detach().cpu().tolist() for name, value in model.fusion_weights().items()},
        "context": model.context_diagnostics(contexts),
    }


def _build_optimizer(model: FiveTaskViewHGT, config: FiveTaskTrainingConfig) -> torch.optim.Optimizer:
    fusion = list(model.fusion_logits.parameters())
    fusion_ids = {id(parameter) for parameter in fusion}
    base = [parameter for parameter in model.parameters() if id(parameter) not in fusion_ids]
    groups: list[dict[str, Any]] = [{"params": base, "weight_decay": config.weight_decay}]
    if fusion:
        groups.append({"params": fusion, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def run_five_task_training(config: FiveTaskTrainingConfig) -> dict[str, Any]:
    config.validate()
    set_seed(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = ProtocolArtifacts.load(config.artifact_path)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_five_task_model(artifacts, config.mode).to(device)
    common_hash = five_task_common_initialization_hash(model)
    backbone_hash = five_task_backbone_hash(model)
    optimizer = _build_optimizer(model, config)
    train_loader = _loader("train", config, artifacts, True)
    validation_loader = _loader("validation", config, artifacts, False)
    diagnostic_loader = _loader("validation", config, artifacts, False, batch_size=8)
    diagnostic_batch = _move_batch_to_device(next(iter(diagnostic_loader)), device)
    initial_diagnostics = five_task_gradient_diagnostics(model, diagnostic_batch, artifacts)
    checkpoint = output / "best.pt"
    history: list[dict[str, Any]] = []
    best = float("inf")
    stale = 0
    started = time.monotonic()
    for epoch in range(1, config.max_epochs + 1):
        train = _train_epoch(model, train_loader, optimizer, artifacts, config, device)
        validation, _ = evaluate_five_task(model, validation_loader, artifacts, device)
        selection = float(validation["loss"])
        record: dict[str, Any] = {
            "epoch": epoch,
            "train": train,
            "validation": validation,
            "selection_value": selection,
        }
        if config.mode == "five_soft":
            record["fusion"] = _fusion_record(model, diagnostic_batch)
        history.append(record)
        if selection < best - 1e-12:
            best, stale = selection, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "selection_value": selection,
                    "config": asdict(config),
                    "mode": config.mode,
                    "window_size": WINDOW_SIZE,
                    "source_sha256": _source_hash(),
                    "initial_common_sha256": common_hash,
                    "initial_backbone_sha256": backbone_hash,
                },
                checkpoint,
            )
        else:
            stale += 1
        if stale >= config.patience:
            break
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    best_diagnostics = five_task_gradient_diagnostics(model, diagnostic_batch, artifacts)
    validation, predictions = evaluate_five_task(model, validation_loader, artifacts, device)
    predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    _write_metric_tables(validation, output)
    result = {
        "config": asdict(config),
        "best_epoch": int(state["epoch"]),
        "selection_value": float(state["selection_value"]),
        "history": history,
        "validation": validation,
        "test": None,
        "elapsed_seconds": time.monotonic() - started,
        "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "initial_common_sha256": common_hash,
        "initial_backbone_sha256": backbone_hash,
        "source_sha256": _source_hash(),
        "gradient_diagnostics": {"initial": initial_diagnostics, "best": best_diagnostics},
        "fusion": _fusion_record(model, diagnostic_batch) if config.mode == "five_soft" else None,
        "test_accessed": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def evaluate_five_task_checkpoint(
    config: FiveTaskTrainingConfig, checkpoint: Path
) -> dict[str, Any]:
    config.validate()
    set_seed(config.seed)
    artifacts = ProtocolArtifacts.load(config.artifact_path)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("mode") != config.mode or int(state["config"]["seed"]) != config.seed:
        raise ValueError("Five-task checkpoint mode/seed mismatch")
    model = build_five_task_model(artifacts, config.mode).to(device)
    model.load_state_dict(state["model"])
    split = "test" if config.evaluate_test else "validation"
    started = time.monotonic()
    metrics, predictions = evaluate_five_task(
        model, _loader(split, config, artifacts, False), artifacts, device
    )
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / f"{split}_predictions.parquet", index=False)
    _write_metric_tables(metrics, output)
    result = {
        "config": asdict(config),
        "evaluation_only": True,
        "source_checkpoint": str(checkpoint),
        "best_epoch": int(state["epoch"]),
        "validation": metrics if split == "validation" else None,
        "test": metrics if split == "test" else None,
        "elapsed_seconds": time.monotonic() - started,
        "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "initial_common_sha256": state["initial_common_sha256"],
        "initial_backbone_sha256": state["initial_backbone_sha256"],
        "source_sha256": _source_hash(),
        "fusion": _fusion_record(model, _move_batch_to_device(next(iter(_loader("validation", config, artifacts, False, batch_size=8))), device)) if config.mode == "five_soft" else None,
        "test_accessed": split == "test",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
