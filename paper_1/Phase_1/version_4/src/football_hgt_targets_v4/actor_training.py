"""Independent next-Team and match-local next-Player scale experiments."""

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from football_benchmark.data import CanonicalEventDataset, MatchBlockShuffleSampler, load_records
from football_benchmark.protocol import ProtocolArtifacts
from football_benchmark.sampling import TargetSamplePlan

from .actor_data import collate_actor_hgt
from .actor_study import ACTOR_TASKS, ACTOR_VIEWS
from .constants import FEASIBILITY_ARTIFACT, POSSESSION_GRAPH_ROOT, SAMPLE_PLAN, WINDOW_SIZE
from .model import ActorScaleHGT
from .subgraph_views import DEFAULT_SELECTOR_SEED
from .training import _move_batch_to_device, set_seed


@dataclass(frozen=True)
class ActorTrainingConfig:
    task: str
    view: str
    output_dir: Path
    seed: int
    device: str
    learning_rate: float = 9e-4
    max_epochs: int = 8
    patience: int = 2
    batch_size: int = 256
    num_workers: int = 3
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

    def validate(self) -> None:
        if self.task not in ACTOR_TASKS:
            raise ValueError(f"Unknown actor task {self.task!r}")
        if self.view not in ACTOR_VIEWS:
            raise ValueError(f"Unknown actor view {self.view!r}")


def _loader_worker_init(_: int) -> None:
    torch.set_num_threads(1)


def _source_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def actor_backbone_hash(model: ActorScaleHGT) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(("team_actor_head.", "player_actor_scorer.")):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _loader(
    split: str,
    config: ActorTrainingConfig,
    artifacts: ProtocolArtifacts,
    shuffle: bool,
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
        options.update(
            prefetch_factor=2,
            persistent_workers=True,
            worker_init_fn=_loader_worker_init,
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(
            collate_actor_hgt,
            artifacts=artifacts,
            window_size=WINDOW_SIZE,
            context_view=config.view,
            selector_seed=config.selector_seed,
        ),
        generator=torch.Generator().manual_seed(config.seed),
        drop_last=False,
        **options,
    )


def player_cross_entropy(
    scores: torch.Tensor,
    ptr: torch.Tensor,
    local_targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for row in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        start, stop = int(ptr[row]), int(ptr[row + 1])
        target = start + int(local_targets[row])
        losses.append(torch.logsumexp(scores[start:stop], dim=0) - scores[target])
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def actor_loss(
    predictions: dict[str, torch.Tensor], batch: dict[str, Any], task: str
) -> torch.Tensor:
    if task == "team":
        return F.cross_entropy(predictions["team_logits"], batch["targets"]["team_actor"])
    return player_cross_entropy(
        predictions["player_scores"],
        batch["graph"]["player"].ptr,
        batch["targets"]["player_local"],
        batch["targets"]["player_mask"].bool(),
    )


def _prediction_frame(
    predictions: dict[str, torch.Tensor], batch: dict[str, Any], task: str
) -> pd.DataFrame:
    targets = batch["targets"]
    base: dict[str, Any] = {
        "sample_id": batch["sample_ids"],
        "match_id": batch["match_ids"].detach().cpu().numpy(),
        "current_event_index": batch["current_event_indices"].detach().cpu().numpy(),
        "event_true": targets["raw_event_10"].detach().cpu().numpy(),
    }
    if task == "team":
        probabilities = predictions["team_logits"].softmax(dim=-1)
        base.update(
            team_true=targets["team_actor"].detach().cpu().numpy(),
            team_pred=probabilities.argmax(dim=-1).detach().cpu().numpy(),
            team_same_probability=probabilities[:, 1].detach().cpu().numpy(),
        )
        return pd.DataFrame(base)

    ptr = batch["graph"]["player"].ptr
    scores = predictions["player_scores"]
    raw_ids = batch["graph"]["player"].raw_id
    predicted_raw: list[int] = []
    ranks: list[int] = []
    candidate_counts: list[int] = []
    for row in range(len(batch["sample_ids"])):
        start, stop = int(ptr[row]), int(ptr[row + 1])
        local_scores = scores[start:stop]
        target = int(targets["player_local"][row])
        order = torch.argsort(local_scores, descending=True)
        rank = int(torch.nonzero(order == target, as_tuple=False)[0]) + 1
        predicted_raw.append(int(raw_ids[start + int(order[0])]))
        ranks.append(rank)
        candidate_counts.append(stop - start)
    base.update(
        player_true_raw=targets["player_raw"].detach().cpu().numpy(),
        player_pred_raw=np.asarray(predicted_raw, dtype=np.int64),
        player_rank=np.asarray(ranks, dtype=np.int64),
        player_mask=targets["player_mask"].detach().cpu().numpy(),
        candidate_count=np.asarray(candidate_counts, dtype=np.int64),
    )
    return pd.DataFrame(base)


def compute_actor_metrics(frame: pd.DataFrame, task: str, mean_loss: float) -> dict[str, Any]:
    result: dict[str, Any] = {"loss": float(mean_loss), "samples": int(len(frame))}
    if task == "team":
        labels = [0, 1]
        result["team"] = {
            "accuracy": float(accuracy_score(frame.team_true, frame.team_pred)),
            "macro_f1": float(f1_score(frame.team_true, frame.team_pred, labels=labels, average="macro", zero_division=0)),
            "same_team_rate": float(frame.team_true.mean()),
            "confusion_matrix": confusion_matrix(frame.team_true, frame.team_pred, labels=labels).tolist(),
        }
        return result
    active = frame[frame.player_mask.astype(bool)]
    ranks = active.player_rank.to_numpy(dtype=np.float64)
    result["player"] = {
        "valid_samples": int(len(active)),
        "coverage": float(len(active) / max(len(frame), 1)),
        "top1_accuracy": float(np.mean(ranks <= 1)),
        "top3_accuracy": float(np.mean(ranks <= 3)),
        "top5_accuracy": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_candidate_count": float(active.candidate_count.mean()),
    }
    return result


@torch.no_grad()
def evaluate(
    model: ActorScaleHGT,
    loader: DataLoader,
    config: ActorTrainingConfig,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    frames: list[pd.DataFrame] = []
    loss_sum = 0.0
    count = 0
    for raw_batch in loader:
        batch = _move_batch_to_device(raw_batch, device)
        predictions = model(batch)
        loss = actor_loss(predictions, batch, config.task)
        size = len(batch["sample_ids"])
        frames.append(_prediction_frame(predictions, batch, config.task))
        loss_sum += float(loss) * size
        count += size
    frame = pd.concat(frames, ignore_index=True)
    return compute_actor_metrics(frame, config.task, loss_sum / max(count, 1)), frame


def _train_epoch(
    model: ActorScaleHGT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ActorTrainingConfig,
    device: torch.device,
) -> float:
    model.train()
    total, count = 0.0, 0
    for raw_batch in loader:
        batch = _move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = actor_loss(model(batch), batch, config.task)
        loss.backward()
        clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        size = len(batch["sample_ids"])
        total += float(loss.detach()) * size
        count += size
    return total / max(count, 1)


def _selection_value(metrics: dict[str, Any], task: str) -> float:
    return float(metrics[task]["macro_f1" if task == "team" else "top1_accuracy"])


def _build_model(artifacts: ProtocolArtifacts, config: ActorTrainingConfig) -> ActorScaleHGT:
    return ActorScaleHGT(artifacts, config.task)


def run_actor_training(config: ActorTrainingConfig) -> dict[str, Any]:
    config.validate()
    set_seed(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = ProtocolArtifacts.load(config.artifact_path)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = _build_model(artifacts, config).to(device)
    initial_hash = actor_backbone_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = _loader("train", config, artifacts, True)
    validation_loader = _loader("validation", config, artifacts, False)
    checkpoint = output / "best.pt"
    history: list[dict[str, Any]] = []
    best = -float("inf")
    stale = 0
    started = time.monotonic()
    for epoch in range(1, config.max_epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, config, device)
        validation, _ = evaluate(model, validation_loader, config, device)
        selection = _selection_value(validation, config.task)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation": validation, "selection_value": selection})
        if selection > best + 1e-12:
            best, stale = selection, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "selection_value": selection,
                    "config": asdict(config),
                    "initial_backbone_sha256": initial_hash,
                    "source_sha256": _source_hash(),
                },
                checkpoint,
            )
        else:
            stale += 1
        if stale >= config.patience:
            break
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    validation, predictions = evaluate(model, validation_loader, config, device)
    predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    result = {
        "config": asdict(config),
        "best_epoch": int(state["epoch"]),
        "selection_value": float(state["selection_value"]),
        "history": history,
        "validation": validation,
        "test": None,
        "elapsed_seconds": time.monotonic() - started,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "initial_backbone_sha256": initial_hash,
        "source_sha256": _source_hash(),
        "test_accessed": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def evaluate_actor_checkpoint(config: ActorTrainingConfig, checkpoint: Path) -> dict[str, Any]:
    config.validate()
    set_seed(config.seed)
    artifacts = ProtocolArtifacts.load(config.artifact_path)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    saved = state["config"]
    if saved["task"] != config.task or saved["view"] != config.view:
        raise ValueError("Actor checkpoint task/view mismatch")
    model = _build_model(artifacts, config).to(device)
    model.load_state_dict(state["model"])
    split = "test" if config.evaluate_test else "validation"
    metrics, predictions = evaluate(model, _loader(split, config, artifacts, False), config, device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / f"{split}_predictions.parquet", index=False)
    result = {
        "config": asdict(config),
        "evaluation_only": True,
        "source_checkpoint": str(checkpoint),
        "best_epoch": int(state["epoch"]),
        "validation": metrics if split == "validation" else None,
        "test": metrics if split == "test" else None,
        "elapsed_seconds": 0.0,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "initial_backbone_sha256": state["initial_backbone_sha256"],
        "source_sha256": _source_hash(),
        "test_accessed": split == "test",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
