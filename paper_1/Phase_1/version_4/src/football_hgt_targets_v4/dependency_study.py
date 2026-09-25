"""Frozen-context probes for next-event target dependencies."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from football_benchmark.data import move_batch_to_device
from football_benchmark.protocol import ProtocolArtifacts

from .constants import CONFIRMATION_SEEDS, EXPERIMENT_ROOT
from .losses import position_loss, time_loss
from .metrics import compute_metrics
from .model import TargetStudyHGT, time_bucket_ids
from .training import TargetTrainingConfig, _loader, set_seed


DEPENDENCY_ROOT = EXPERIMENT_ROOT / "target_dependency_v1"
CACHE_ROOT = DEPENDENCY_ROOT / "cache"
CHECKPOINT_ROOT = EXPERIMENT_ROOT / "loss_balance_validation/j2_020"
POSITION_CONFIGS = (
    "p0_null",
    "p1_event_oracle",
    "p2_event_predicted",
    "p3_time_oracle",
    "p4_time_predicted",
    "p5_event_time_oracle",
    "p6_event_time_predicted",
    "p7_event_shuffled",
    "p8_time_shuffled",
    "p9_event_time_shuffled",
)
TIME_CONFIGS = (
    "t0_null",
    "t1_event_oracle",
    "t2_event_predicted",
    "t3_event_shuffled",
)
CONFIGS_BY_FAMILY = {"position": POSITION_CONFIGS, "time": TIME_CONFIGS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(seed: int) -> Path:
    return CHECKPOINT_ROOT / f"seed{seed}/best.pt"


def _checkpoint_config(state: dict[str, Any], device: str, split: str) -> TargetTrainingConfig:
    raw = dict(state["config"])
    for key in ("artifact_path", "output_dir", "sample_plan_path"):
        if raw.get(key) is not None:
            raw[key] = Path(raw[key])
    config = TargetTrainingConfig(**raw)
    return replace(
        config,
        device=device,
        num_workers=2,
        micro_batch_size=256,
        effective_batch_size=256,
        full_test=split == "test",
        max_train_samples=None,
        max_validation_samples=None,
        max_test_samples=None,
    )


@torch.no_grad()
def build_context_cache(seed: int, split: str, device_name: str) -> Path:
    """Cache frozen J2 contexts and targets for one split."""

    if seed not in CONFIRMATION_SEEDS or split not in {"train", "validation", "test"}:
        raise ValueError("Unsupported seed or split")
    output = CACHE_ROOT / f"seed{seed}/{split}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint(seed)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _checkpoint_config(state, device_name, split)
    artifacts = ProtocolArtifacts.load(config.artifact_path)
    set_seed(seed)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = TargetStudyHGT(
        artifacts, config.task, config.method, config.joint_methods
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    loader = _loader(split, config, artifacts, False)
    chunks: dict[str, list[Any]] = {
        "context": [],
        "event_probabilities": [],
        "base_time_seconds": [],
        "base_position_xy": [],
        "event_true": [],
        "time_true": [],
        "time_mask": [],
        "position_true": [],
        "position_mask": [],
        "zone_true": [],
        "match_ids": [],
        "current_event_indices": [],
    }
    sample_ids: list[str] = []
    max_difference = 0.0
    captured_contexts: list[torch.Tensor] = []
    hook = model.context_projection.register_forward_hook(
        lambda _module, _inputs, output: captured_contexts.append(output)
    )
    for batch_index, raw_batch in enumerate(loader):
        batch = move_batch_to_device(raw_batch, device)
        direct = model(batch)
        if len(captured_contexts) != 1:
            raise RuntimeError("Context hook did not capture exactly one model context")
        context = captured_contexts.pop()
        event_logits = model.event_head(context)
        log_delta = F.softplus(model.time_head(context).squeeze(-1))
        time_seconds = torch.expm1(log_delta).clamp(max=60.0)
        position = torch.sigmoid(model.position_head(context))
        if batch_index == 0:
            max_difference = max(
                float((direct["event_logits"] - event_logits).abs().max()),
                float((direct["time_seconds"] - time_seconds).abs().max()),
                float((direct["position_xy"] - position).abs().max()),
            )
            if max_difference >= 1e-6:
                raise RuntimeError(f"Frozen cache mismatch: {max_difference}")
        targets = batch["targets"]
        values = {
            "context": context,
            "event_probabilities": event_logits.softmax(dim=-1),
            "base_time_seconds": time_seconds,
            "base_position_xy": position,
            "event_true": targets["raw_event_10"],
            "time_true": targets["delta_seconds_60"],
            "time_mask": targets["time_mask"].bool(),
            "position_true": targets["position_xy"],
            "position_mask": targets["position_mask"].bool(),
            "zone_true": targets["zone_20"],
            "match_ids": batch["match_ids"],
            "current_event_indices": batch["current_event_indices"],
        }
        for name, value in values.items():
            chunks[name].append(value.detach().cpu())
        sample_ids.extend(batch["sample_ids"])
    hook.remove()
    payload = {name: torch.cat(values) for name, values in chunks.items()}
    payload.update(
        sample_ids=sample_ids,
        seed=seed,
        split=split,
        checkpoint=str(checkpoint),
        checkpoint_sha256=_sha256(checkpoint),
        cache_output_max_difference=max_difference,
    )
    torch.save(payload, output)
    return output


def load_cache(seed: int, split: str) -> dict[str, Any]:
    return torch.load(CACHE_ROOT / f"seed{seed}/{split}.pt", map_location="cpu", weights_only=False)


def _permutation(length: int, seed: int, split: str, config_name: str) -> torch.Tensor:
    token = f"{seed}:{split}:{config_name}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return torch.randperm(length, generator=torch.Generator().manual_seed(derived))


def _time_features(seconds: torch.Tensor, mask: torch.Tensor, oracle: bool) -> torch.Tensor:
    normalized = torch.log1p(seconds.clamp(0, 60)) / math.log(61.0)
    buckets = F.one_hot(time_bucket_ids(seconds), num_classes=4).float()
    period_break = (~mask).float() if oracle else torch.zeros_like(normalized)
    return torch.cat((normalized[:, None], buckets, period_break[:, None]), dim=1)


def build_conditions(cache: dict[str, Any], family: str, config_name: str) -> torch.Tensor:
    """Build fixed-width, parameter-matched conditions for one probe."""

    length = len(cache["sample_ids"])
    event_oracle = F.one_hot(cache["event_true"], num_classes=10).float()
    event_predicted = cache["event_probabilities"].float()
    time_oracle = _time_features(cache["time_true"], cache["time_mask"], True)
    time_predicted = _time_features(
        cache["base_time_seconds"], torch.ones(length, dtype=torch.bool), False
    )
    permutation = _permutation(length, cache["seed"], cache["split"], config_name)
    if family == "time":
        result = torch.zeros((length, 10), dtype=torch.float32)
        if config_name == "t1_event_oracle":
            result = event_oracle
        elif config_name == "t2_event_predicted":
            result = event_predicted
        elif config_name == "t3_event_shuffled":
            result = event_oracle[permutation]
        elif config_name != "t0_null":
            raise ValueError(config_name)
        return result
    if family != "position":
        raise ValueError(family)
    result = torch.zeros((length, 16), dtype=torch.float32)
    if config_name == "p1_event_oracle":
        result[:, :10] = event_oracle
    elif config_name == "p2_event_predicted":
        result[:, :10] = event_predicted
    elif config_name == "p3_time_oracle":
        result[:, 10:] = time_oracle
    elif config_name == "p4_time_predicted":
        result[:, 10:] = time_predicted
    elif config_name == "p5_event_time_oracle":
        result[:, :10], result[:, 10:] = event_oracle, time_oracle
    elif config_name == "p6_event_time_predicted":
        result[:, :10], result[:, 10:] = event_predicted, time_predicted
    elif config_name == "p7_event_shuffled":
        result[:, :10] = event_oracle[permutation]
    elif config_name == "p8_time_shuffled":
        result[:, 10:] = time_oracle[permutation]
    elif config_name == "p9_event_time_shuffled":
        result[:, :10], result[:, 10:] = event_oracle[permutation], time_oracle[permutation]
    elif config_name != "p0_null":
        raise ValueError(config_name)
    return result


class CacheDataset(Dataset):
    def __init__(self, cache: dict[str, Any], conditions: torch.Tensor) -> None:
        self.cache = cache
        self.conditions = conditions

    def __len__(self) -> int:
        return len(self.cache["sample_ids"])

    def __getitem__(self, index: int) -> int:
        return index


class ResidualProbe(nn.Module):
    def __init__(self, family: str) -> None:
        super().__init__()
        self.family = family
        condition_dim = 16 if family == "position" else 10
        output_dim = 2 if family == "position" else 1
        self.hidden = nn.Linear(64 + condition_dim, 64)
        self.output = nn.Linear(64, output_dim)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        context: torch.Tensor,
        condition: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.output(F.dropout(F.gelu(self.hidden(torch.cat((context, condition), dim=-1))), 0.1, self.training))
        if self.family == "position":
            logits = torch.logit(base.clamp(1e-6, 1 - 1e-6))
            return torch.sigmoid(logits + residual)
        base_log = torch.log1p(base)
        adjusted_log = (base_log + residual.squeeze(-1)).clamp(0.0, math.log(61.0))
        correction = torch.expm1(adjusted_log) - torch.expm1(base_log)
        return (base + correction).clamp(0.0, 60.0)


def _collate_indices(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def _loader_from_cache(
    cache: dict[str, Any], conditions: torch.Tensor, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    return DataLoader(
        CacheDataset(cache, conditions),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_indices,
        generator=torch.Generator().manual_seed(seed),
    )


def _batch(cache: dict[str, Any], conditions: torch.Tensor, indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    names = (
        "context", "base_time_seconds", "base_position_xy", "event_true",
        "time_true", "time_mask", "position_true", "position_mask", "zone_true",
    )
    result = {name: cache[name][indices].to(device) for name in names}
    result["condition"] = conditions[indices].to(device)
    return result


def _probe_loss(family: str, prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if family == "position":
        return position_loss(
            {"position_xy": prediction},
            {
                "position_xy": batch["position_true"],
                "position_mask": batch["position_mask"],
                "zone_20": batch["zone_true"],
            },
            "xy",
        )
    return time_loss(
        {"time_seconds": prediction},
        {"delta_seconds_60": batch["time_true"], "time_mask": batch["time_mask"]},
        "current_huber",
    )


@torch.no_grad()
def evaluate_probe(
    model: ResidualProbe,
    family: str,
    cache: dict[str, Any],
    conditions: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    outputs: list[np.ndarray] = []
    losses = 0.0
    count = 0
    loader = _loader_from_cache(cache, conditions, batch_size, False, cache["seed"])
    for indices in loader:
        values = _batch(cache, conditions, indices, device)
        base = values["base_position_xy"] if family == "position" else values["base_time_seconds"]
        prediction = model(values["context"], values["condition"], base)
        loss = _probe_loss(family, prediction, values)
        outputs.append(prediction.cpu().numpy())
        losses += float(loss) * len(indices)
        count += len(indices)
    prediction = np.concatenate(outputs)
    frame = pd.DataFrame(
        {
            "sample_id": cache["sample_ids"],
            "match_id": cache["match_ids"].numpy(),
            "current_event_index": cache["current_event_indices"].numpy(),
            "event_true": cache["event_true"].numpy(),
        }
    )
    if family == "position":
        frame["position_true_x"] = cache["position_true"][:, 0].numpy()
        frame["position_true_y"] = cache["position_true"][:, 1].numpy()
        frame["position_pred_x"] = prediction[:, 0]
        frame["position_pred_y"] = prediction[:, 1]
        frame["position_mask"] = cache["position_mask"].numpy()
        frame["zone_true"] = cache["zone_true"].numpy()
        from football_benchmark.mappings import position_to_zone
        frame["zone_pred"] = position_to_zone(torch.from_numpy(prediction)).numpy()
    else:
        frame["time_true"] = cache["time_true"].numpy()
        frame["time_pred"] = prediction
        frame["time_mask"] = cache["time_mask"].numpy()
        frame["time_bucket_true"] = time_bucket_ids(cache["time_true"]).numpy()
    return compute_metrics(frame, losses / max(count, 1)), frame


@dataclass(frozen=True)
class ProbeConfig:
    family: str
    name: str
    seed: int
    output_dir: Path
    device: str
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_train_samples: int | None = None
    max_validation_samples: int | None = None


def _slice_cache(cache: dict[str, Any], limit: int | None) -> dict[str, Any]:
    if limit is None or len(cache["sample_ids"]) <= limit:
        return cache
    result: dict[str, Any] = {}
    length = len(cache["sample_ids"])
    for name, value in cache.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (length,):
            result[name] = value[:limit]
        elif name == "sample_ids":
            result[name] = value[:limit]
        else:
            result[name] = value
    return result


def train_probe(config: ProbeConfig) -> dict[str, Any]:
    if config.name not in CONFIGS_BY_FAMILY[config.family]:
        raise ValueError(config.name)
    set_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    train_cache = _slice_cache(load_cache(config.seed, "train"), config.max_train_samples)
    validation_cache = _slice_cache(
        load_cache(config.seed, "validation"), config.max_validation_samples
    )
    train_conditions = build_conditions(train_cache, config.family, config.name)
    validation_conditions = build_conditions(validation_cache, config.family, config.name)
    model = ResidualProbe(config.family).to(device)
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    zero_initialization_verified = bool(
        torch.count_nonzero(initial_state["output.weight"]) == 0
        and torch.count_nonzero(initial_state["output.bias"]) == 0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = _loader_from_cache(train_cache, train_conditions, config.batch_size, True, config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.output_dir / "best.pt"
    history, best, stale = [], None, 0
    started = time.monotonic()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for indices in train_loader:
            values = _batch(train_cache, train_conditions, indices, device)
            base = values["base_position_xy"] if config.family == "position" else values["base_time_seconds"]
            loss = _probe_loss(config.family, model(values["context"], values["condition"], base), values)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss) * len(indices); count += len(indices)
        metrics, _ = evaluate_probe(model, config.family, validation_cache, validation_conditions, device, config.batch_size)
        value = metrics["position"]["distance_mae_m"] if config.family == "position" else metrics["time"]["mae_seconds"]
        history.append({"epoch": epoch, "train_loss": total / count, "validation": metrics, "selection_value": value})
        if best is None or value < best - 1e-12:
            best, stale = value, 0
            torch.save({"model": model.state_dict(), "config": config.__dict__, "epoch": epoch, "selection_value": value}, checkpoint)
        else:
            stale += 1
        if stale >= config.patience: break
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    metrics, frame = evaluate_probe(model, config.family, validation_cache, validation_conditions, device, config.batch_size)
    frame.to_parquet(config.output_dir / "validation_predictions.parquet", index=False)
    result = {
        "config": {**config.__dict__, "output_dir": str(config.output_dir)},
        "best_epoch": state["epoch"], "validation": metrics, "test": None,
        "history": history, "elapsed_seconds": time.monotonic() - started,
        "test_accessed": False,
        "zero_initialization_verified": zero_initialization_verified,
    }
    (config.output_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def evaluate_test_probe(config: ProbeConfig, checkpoint: Path) -> dict[str, Any]:
    device = torch.device(config.device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ResidualProbe(config.family).to(device)
    model.load_state_dict(state["model"])
    cache = load_cache(config.seed, "test")
    conditions = build_conditions(cache, config.family, config.name)
    metrics, frame = evaluate_probe(model, config.family, cache, conditions, device, config.batch_size)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(config.output_dir / "test_predictions.parquet", index=False)
    result = {
        "config": {**config.__dict__, "output_dir": str(config.output_dir)},
        "best_epoch": state["epoch"], "validation": None, "test": metrics,
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": _sha256(checkpoint),
        "test_accessed": True,
    }
    (config.output_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
