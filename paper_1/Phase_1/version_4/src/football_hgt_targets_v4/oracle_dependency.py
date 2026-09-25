"""Frozen Partial-L2 caches and lightweight Oracle dependency heads."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from football_benchmark.constants import PITCH_LENGTH_METERS, PITCH_WIDTH_METERS
from football_benchmark.protocol import ProtocolArtifacts

from .actor_training import player_cross_entropy
from .constants import (
    CONFIRMATION_SEEDS,
    FEASIBILITY_ARTIFACT,
    PHASE_ROOT,
    SAMPLE_PLAN,
)
from .five_task_training import FiveTaskTrainingConfig, _loader
from .model import build_partial_l2_model
from .oracle_dependency_study import (
    CONDITIONS,
    FAMILIES,
    cache_path,
    source_checkpoint,
    validation_lock_path,
)
from .training import _move_batch_to_device, set_seed


ROSTER_PATH = PHASE_ROOT / "data/whyscout/raw/matches/matches_England.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_roster_team_map(path: Path = ROSTER_PATH) -> dict[int, dict[int, int]]:
    """Return pre-match match/player/team mappings from lineup and bench only."""

    rows = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, dict[int, int]] = {}
    for match in rows:
        mapping: dict[int, int] = {}
        for raw_team, team in match["teamsData"].items():
            team_id = int(raw_team)
            formation = team.get("formation") or {}
            for group in ("lineup", "bench"):
                for player in formation.get(group, ()):  # pre-match roster only
                    player_id = int(player.get("playerId", 0))
                    if player_id <= 0:
                        continue
                    previous = mapping.get(player_id)
                    if previous is not None and previous != team_id:
                        raise RuntimeError(
                            f"Player {player_id} has two teams in match {match['wyId']}"
                        )
                    mapping[player_id] = team_id
        if not mapping:
            raise RuntimeError(f"No roster mapping for match {match['wyId']}")
        result[int(match["wyId"])] = mapping
    return result


def _cache_loader(seed: int, split: str, artifacts: ProtocolArtifacts) -> DataLoader:
    config = FiveTaskTrainingConfig(
        mode="five_f80",
        output_dir=Path("."),
        seed=seed,
        device="cpu",
        batch_size=256,
        num_workers=2,
        artifact_path=FEASIBILITY_ARTIFACT,
        sample_plan_path=SAMPLE_PLAN,
        full_test=split == "test",
        evaluate_test=split == "test",
    )
    return _loader(split, config, artifacts, False)


@torch.no_grad()
def build_oracle_cache(seed: int, split: str, device_name: str) -> Path:
    """Cache frozen Partial-L2 contexts, candidate states, and predictions."""

    if seed not in CONFIRMATION_SEEDS or split not in {"train", "validation", "test"}:
        raise ValueError("Unsupported seed or split")
    if split == "test" and not validation_lock_path().exists():
        raise RuntimeError("Validation dependency decision must be locked before test cache")
    output = cache_path(seed, split)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = source_checkpoint(seed)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    set_seed(seed)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = build_partial_l2_model(artifacts).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    roster = load_roster_team_map()
    loader = _cache_loader(seed, split, artifacts)

    sample_values: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "main_context", "player_context", "base_event_logits",
            "base_position_xy", "event_true", "position_true", "position_mask",
            "zone_true", "team_true", "player_local", "player_mask",
            "player_raw", "match_ids", "current_event_indices",
            "target_team_raw", "team_mapping_valid",
        )
    }
    candidate_values: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "candidate_states", "candidate_scores", "candidate_raw",
            "candidate_team_raw", "candidate_team_valid",
        )
    }
    counts: list[torch.Tensor] = []
    sample_ids: list[str] = []
    maximum_difference = 0.0
    captured_player_inputs: list[torch.Tensor] = []
    hook = model.player_actor_scorer.register_forward_pre_hook(
        lambda _module, inputs: captured_player_inputs.append(inputs[0])
    )

    for batch_index, raw_batch in enumerate(loader):
        batch = _move_batch_to_device(raw_batch, device)
        predictions, contexts = model.forward_with_contexts(batch)
        if len(captured_player_inputs) != 1:
            raise RuntimeError("Player scorer hook did not capture exactly one input")
        player_input = captured_player_inputs.pop()
        main_context = contexts["f80"]
        player_context = contexts["player"]
        graph = batch["graphs"]["f80"]
        event_logits = predictions["event_logits"]
        position = predictions["position_xy"]
        player_scores = predictions["player_scores"]
        candidate_states = player_input[:, 64:]
        if batch_index == 0:
            maximum_difference = max(
                float((predictions["event_logits"] - event_logits).abs().max()),
                float((predictions["position_xy"] - position).abs().max()),
                float((predictions["player_scores"] - player_scores).abs().max()),
            )
            if maximum_difference >= 1e-6:
                raise RuntimeError(f"Frozen Partial-L2 cache mismatch: {maximum_difference}")

        targets = batch["targets"]
        raw_ids = graph["player"].raw_id.detach().cpu().long()
        ptr = graph["player"].ptr.detach().cpu().long()
        batch_match_ids = batch["match_ids"].detach().cpu().long()
        candidate_teams = torch.full_like(raw_ids, -1)
        candidate_valid = torch.zeros_like(raw_ids, dtype=torch.bool)
        target_teams = torch.full((len(batch["sample_ids"]),), -1, dtype=torch.long)
        target_valid = torch.zeros(len(batch["sample_ids"]), dtype=torch.bool)
        player_raw = targets["player_raw"].detach().cpu().long()
        for row, match_id in enumerate(batch_match_ids.tolist()):
            mapping = roster.get(int(match_id))
            if mapping is None:
                raise RuntimeError(f"Missing pre-match roster for {match_id}")
            start, stop = int(ptr[row]), int(ptr[row + 1])
            for index in range(start, stop):
                team_id = mapping.get(int(raw_ids[index]))
                if team_id is not None:
                    candidate_teams[index] = team_id
                    candidate_valid[index] = True
            team_id = mapping.get(int(player_raw[row]))
            if team_id is not None:
                target_teams[row] = team_id
                target_valid[row] = True

        values = {
            "main_context": main_context,
            "player_context": player_context,
            "base_event_logits": event_logits,
            "base_position_xy": position,
            "event_true": targets["raw_event_10"],
            "position_true": targets["position_xy"],
            "position_mask": targets["position_mask"].bool(),
            "zone_true": targets["zone_20"],
            "team_true": targets["team_actor"],
            "player_local": targets["player_local"],
            "player_mask": targets["player_mask"].bool(),
            "player_raw": targets["player_raw"],
            "match_ids": batch["match_ids"],
            "current_event_indices": batch["current_event_indices"],
            "target_team_raw": target_teams,
            "team_mapping_valid": target_valid,
        }
        for name, value in values.items():
            sample_values[name].append(value.detach().cpu())
        candidates = {
            "candidate_states": candidate_states,
            "candidate_scores": player_scores,
            "candidate_raw": raw_ids,
            "candidate_team_raw": candidate_teams,
            "candidate_team_valid": candidate_valid,
        }
        for name, value in candidates.items():
            candidate_values[name].append(value.detach().cpu())
        counts.append(ptr[1:] - ptr[:-1])
        sample_ids.extend(batch["sample_ids"])

    hook.remove()

    payload = {name: torch.cat(values) for name, values in sample_values.items()}
    payload.update({name: torch.cat(values) for name, values in candidate_values.items()})
    all_counts = torch.cat(counts)
    payload["candidate_ptr"] = torch.cat(
        (torch.zeros(1, dtype=torch.long), all_counts.cumsum(0))
    )
    payload.update(
        sample_ids=sample_ids,
        seed=seed,
        split=split,
        source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=_sha256(checkpoint),
        cache_output_max_difference=maximum_difference,
        roster_path=str(ROSTER_PATH),
        roster_sha256=_sha256(ROSTER_PATH),
    )
    torch.save(payload, output)
    return output


def load_oracle_cache(seed: int, split: str) -> dict[str, Any]:
    return torch.load(cache_path(seed, split), map_location="cpu", weights_only=False)


def within_match_donors(cache: dict[str, Any], token: str) -> torch.Tensor:
    """Deterministically rotate valid rows within each match."""

    cache_key = f"_donors_{token}"
    if cache_key in cache:
        return cache[cache_key]
    length = len(cache["sample_ids"])
    result = torch.arange(length)
    match_ids = cache["match_ids"].long()
    valid = cache["player_mask"].bool()
    for match_id in torch.unique(match_ids).tolist():
        rows = torch.nonzero((match_ids == match_id) & valid, as_tuple=False).flatten()
        if rows.numel() <= 1:
            continue
        digest = hashlib.sha256(
            f"{cache['seed']}:{cache['split']}:{token}:{match_id}".encode()
        ).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
        ordered = rows[torch.randperm(rows.numel(), generator=generator)]
        result[ordered] = ordered.roll(1)
    cache[cache_key] = result
    return result


def player_state_condition(
    cache: dict[str, Any], condition: str, indices: torch.Tensor
) -> torch.Tensor:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    condition_rows = indices
    if condition == "shuffled":
        condition_rows = within_match_donors(cache, "player_state")[indices]
    targets = cache["candidate_ptr"][condition_rows] + cache["player_local"][condition_rows]
    values = cache["candidate_states"][targets].clone()
    available = cache["player_mask"][condition_rows].bool()
    values[~available] = 0
    if condition == "null":
        values.zero_()
    return values


def player_team_condition(cache: dict[str, Any], condition: str) -> torch.Tensor:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    cache_key = f"_team_condition_{condition}"
    if cache_key in cache:
        return cache[cache_key]
    length = len(cache["sample_ids"])
    donors = torch.arange(length)
    if condition == "shuffled":
        donors = within_match_donors(cache, "true_team")
    counts = cache["candidate_ptr"][1:] - cache["candidate_ptr"][:-1]
    owners = torch.repeat_interleave(torch.arange(length), counts)
    donor_rows = donors[owners]
    result = torch.zeros((int(cache["candidate_ptr"][-1]), 2), dtype=torch.float32)
    if condition == "null":
        cache[cache_key] = result
        return result
    result[:, 0] = cache["team_true"][donor_rows].float()
    donor_team = cache["target_team_raw"][donor_rows]
    result[:, 1] = (
        cache["candidate_team_valid"].bool()
        & cache["team_mapping_valid"][donor_rows].bool()
        & (cache["candidate_team_raw"] == donor_team)
    ).float()
    cache[cache_key] = result
    return result


class OracleHead(nn.Module):
    def __init__(self, family: str, dropout: float = 0.1) -> None:
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(family)
        self.family = family
        input_dim = 130 if family == "player_team" else 128
        output_dim = 1 if family == "player_team" else 2 if family == "position_player_state" else 10
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, output_dim)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.network(values)
        return torch.sigmoid(output) if self.family == "position_player_state" else output.squeeze(-1) if self.family == "player_team" else output


class IndexDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return index


def index_loader(length: int, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        IndexDataset(length), batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda rows: torch.tensor(rows, dtype=torch.long),
        generator=torch.Generator().manual_seed(seed),
    )


def candidate_batch(cache: dict[str, Any], rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = cache["candidate_ptr"][rows + 1] - cache["candidate_ptr"][rows]
    candidates = torch.cat([
        torch.arange(int(cache["candidate_ptr"][row]), int(cache["candidate_ptr"][row + 1]))
        for row in rows.tolist()
    ])
    owners = torch.repeat_interleave(torch.arange(rows.numel()), counts)
    ptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    return candidates, owners, ptr


def head_inputs(
    cache: dict[str, Any], family: str, condition: str, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if family != "player_team":
        values = torch.cat(
            (cache["main_context"][rows], player_state_condition(cache, condition, rows)), dim=-1
        )
        return values, None
    candidates, owners, ptr = candidate_batch(cache, rows)
    team = player_team_condition(cache, condition)[candidates]
    values = torch.cat(
        (
            cache["player_context"][rows][owners],
            cache["candidate_states"][candidates],
            team,
        ),
        dim=-1,
    )
    return values, ptr


def probe_loss(
    family: str, prediction: torch.Tensor, cache: dict[str, Any], rows: torch.Tensor,
    ptr: torch.Tensor | None,
) -> torch.Tensor:
    if family == "event_player_state":
        return F.cross_entropy(
            prediction, cache["event_true"][rows.cpu()].to(prediction.device)
        )
    if family == "position_player_state":
        cpu_rows = rows.cpu()
        mask = cache["position_mask"][cpu_rows].bool().to(prediction.device)
        target = cache["position_true"][cpu_rows].to(prediction.device)
        values = F.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
        return values[mask].mean() if mask.any() else values.sum() * 0
    assert ptr is not None
    cpu_rows = rows.cpu()
    return player_cross_entropy(
        prediction, ptr,
        cache["player_local"][cpu_rows].to(prediction.device),
        cache["player_mask"][cpu_rows].bool().to(prediction.device),
    )


def _position_distances(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(
        ((prediction[:, 0] - target[:, 0]) * PITCH_LENGTH_METERS) ** 2
        + ((prediction[:, 1] - target[:, 1]) * PITCH_WIDTH_METERS) ** 2
    )


@torch.no_grad()
def evaluate_head(
    model: OracleHead, family: str, condition: str, cache: dict[str, Any],
    device: torch.device, batch_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    frames: list[pd.DataFrame] = []
    losses, total = 0.0, 0
    for rows in index_loader(len(cache["sample_ids"]), batch_size, False, int(cache["seed"])):
        inputs, ptr = head_inputs(cache, family, condition, rows)
        prediction = model(inputs.to(device))
        local_ptr = ptr.to(device) if ptr is not None else None
        loss = probe_loss(family, prediction, cache, rows, local_ptr)
        prediction = prediction.cpu()
        frame = pd.DataFrame({
            "sample_id": [cache["sample_ids"][index] for index in rows.tolist()],
            "match_id": cache["match_ids"][rows].numpy(),
            "current_event_index": cache["current_event_indices"][rows].numpy(),
            "player_mask": cache["player_mask"][rows].numpy(),
        })
        if family == "event_player_state":
            frame["event_true"] = cache["event_true"][rows].numpy()
            frame["event_pred"] = prediction.argmax(-1).numpy()
        elif family == "position_player_state":
            frame["position_mask"] = cache["position_mask"][rows].numpy()
            frame["position_true_x"] = cache["position_true"][rows, 0].numpy()
            frame["position_true_y"] = cache["position_true"][rows, 1].numpy()
            frame["position_pred_x"] = prediction[:, 0].numpy()
            frame["position_pred_y"] = prediction[:, 1].numpy()
            frame["event_true"] = cache["event_true"][rows].numpy()
            frame["zone_true"] = cache["zone_true"][rows].numpy()
        else:
            assert ptr is not None
            ranks, predicted_raw = [], []
            candidates, _, _ = candidate_batch(cache, rows)
            raw = cache["candidate_raw"][candidates]
            for local_row in range(rows.numel()):
                start, stop = int(ptr[local_row]), int(ptr[local_row + 1])
                order = prediction[start:stop].argsort(descending=True)
                target = int(cache["player_local"][rows[local_row]])
                ranks.append(int(torch.nonzero(order == target, as_tuple=False)[0]) + 1)
                predicted_raw.append(int(raw[start + int(order[0])]))
            frame["player_rank"] = ranks
            frame["player_pred_raw"] = predicted_raw
            frame["player_true_raw"] = cache["player_raw"][rows].numpy()
            frame["candidate_count"] = (ptr[1:] - ptr[:-1]).numpy()
            frame["team_mapping_valid"] = cache["team_mapping_valid"][rows].numpy()
        frames.append(frame)
        losses += float(loss) * len(rows)
        total += len(rows)
    frame = pd.concat(frames, ignore_index=True)
    metrics = family_metrics(family, frame)
    metrics["loss"] = losses / max(total, 1)
    return metrics, frame


def family_metrics(family: str, frame: pd.DataFrame) -> dict[str, Any]:
    if family == "event_player_state":
        labels = list(range(10))
        known = frame[frame.player_mask.astype(bool)]
        return {
            "samples": len(frame),
            "accuracy": float(accuracy_score(frame.event_true, frame.event_pred)),
            "macro_f1": float(f1_score(frame.event_true, frame.event_pred, labels=labels, average="macro", zero_division=0)),
            "known_player_samples": len(known),
            "known_player_accuracy": float(accuracy_score(known.event_true, known.event_pred)),
            "known_player_macro_f1": float(f1_score(known.event_true, known.event_pred, labels=labels, average="macro", zero_division=0)),
        }
    if family == "position_player_state":
        active = frame[frame.position_mask.astype(bool)].copy()
        active["distance_m"] = _position_distances(
            active[["position_pred_x", "position_pred_y"]].to_numpy(),
            active[["position_true_x", "position_true_y"]].to_numpy(),
        )
        known = active[active.player_mask.astype(bool)]
        return {
            "samples": len(active),
            "distance_mae_m": float(active.distance_m.mean()),
            "distance_median_m": float(active.distance_m.median()),
            "known_player_samples": len(known),
            "known_player_distance_mae_m": float(known.distance_m.mean()),
            "known_player_distance_median_m": float(known.distance_m.median()),
        }
    active = frame[frame.player_mask.astype(bool)]
    ranks = active.player_rank.to_numpy(float)
    return {
        "samples": len(active),
        "coverage": len(active) / max(len(frame), 1),
        "top1": float(np.mean(ranks <= 1)),
        "top3": float(np.mean(ranks <= 3)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


@dataclass(frozen=True)
class OracleProbeConfig:
    family: str
    condition: str
    seed: int
    output_dir: Path
    device: str
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    max_train_samples: int | None = None
    max_validation_samples: int | None = None

    def validate(self) -> None:
        if self.family not in FAMILIES or self.condition not in CONDITIONS:
            raise ValueError("Unsupported family or condition")


def _limited(cache: dict[str, Any], limit: int | None) -> dict[str, Any]:
    if limit is None or len(cache["sample_ids"]) <= limit:
        return cache
    length = len(cache["sample_ids"])
    candidate_stop = int(cache["candidate_ptr"][limit])
    result: dict[str, Any] = {}
    for name, value in cache.items():
        if name == "sample_ids": result[name] = value[:limit]
        elif name == "candidate_ptr": result[name] = value[: limit + 1]
        elif isinstance(value, torch.Tensor) and value.shape[:1] == (length,): result[name] = value[:limit]
        elif isinstance(value, torch.Tensor) and value.shape[:1] == (int(cache["candidate_ptr"][-1]),): result[name] = value[:candidate_stop]
        else: result[name] = value
    return result


def _selection_value(family: str, metrics: dict[str, Any]) -> tuple[float, ...]:
    if family == "event_player_state":
        return (-metrics["known_player_macro_f1"], -metrics["known_player_accuracy"], metrics["loss"])
    if family == "position_player_state":
        return (metrics["known_player_distance_mae_m"], metrics["loss"])
    return (-metrics["top1"], -metrics["mrr"], metrics["loss"])


def train_oracle_probe(config: OracleProbeConfig) -> dict[str, Any]:
    config.validate()
    set_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    train_cache = _limited(load_oracle_cache(config.seed, "train"), config.max_train_samples)
    validation_cache = _limited(load_oracle_cache(config.seed, "validation"), config.max_validation_samples)
    model = OracleHead(config.family).to(device)
    initial_hash = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = index_loader(len(train_cache["sample_ids"]), config.batch_size, True, config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    best: tuple[float, ...] | None = None
    stale = 0
    started = time.monotonic()
    for epoch in range(1, config.max_epochs + 1):
        model.train(); loss_sum = 0.0; count = 0
        for rows in loader:
            inputs, ptr = head_inputs(train_cache, config.family, config.condition, rows)
            prediction = model(inputs.to(device))
            loss = probe_loss(
                config.family, prediction, train_cache, rows,
                ptr.to(device) if ptr is not None else None,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(rows); count += len(rows)
        metrics, _ = evaluate_head(model, config.family, config.condition, validation_cache, device, config.batch_size)
        value = _selection_value(config.family, metrics)
        history.append({"epoch": epoch, "train_loss": loss_sum / max(count, 1), "validation": metrics, "selection_value": value})
        if best is None or value < best:
            best, stale = value, 0
            torch.save({"model": model.state_dict(), "config": asdict(config), "epoch": epoch, "selection_value": value, "initial_hash": initial_hash}, checkpoint)
        else:
            stale += 1
        if stale >= config.patience:
            break
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    metrics, frame = evaluate_head(model, config.family, config.condition, validation_cache, device, config.batch_size)
    frame.to_parquet(config.output_dir / "validation_predictions.parquet", index=False)
    result = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "best_epoch": state["epoch"], "validation": metrics, "test": None,
        "history": history, "elapsed_seconds": time.monotonic() - started,
        "initial_hash": initial_hash, "test_accessed": False,
    }
    (config.output_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def evaluate_test_oracle_probe(config: OracleProbeConfig, checkpoint: Path) -> dict[str, Any]:
    device = torch.device(config.device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = OracleHead(config.family).to(device); model.load_state_dict(state["model"])
    cache = load_oracle_cache(config.seed, "test")
    metrics, frame = evaluate_head(model, config.family, config.condition, cache, device, config.batch_size)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(config.output_dir / "test_predictions.parquet", index=False)
    result = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "best_epoch": state["epoch"], "validation": None, "test": metrics,
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": _sha256(checkpoint),
        "test_accessed": True,
    }
    (config.output_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


@torch.no_grad()
def player_scores_from_head(
    model: OracleHead, cache: dict[str, Any], condition: str,
    device: torch.device, batch_size: int = 1024,
) -> torch.Tensor:
    model.eval(); chunks: list[torch.Tensor] = []
    for rows in index_loader(len(cache["sample_ids"]), batch_size, False, int(cache["seed"])):
        values, _ = head_inputs(cache, "player_team", condition, rows)
        chunks.append(model(values.to(device)).cpu())
    return torch.cat(chunks)


def hard_true_team_frame(
    cache: dict[str, Any], scores: torch.Tensor, source: str
) -> pd.DataFrame:
    """Evaluate masked and unmasked scores on one identical valid sample set."""

    rows: list[dict[str, Any]] = []
    ptr = cache["candidate_ptr"]
    for row, sample_id in enumerate(cache["sample_ids"]):
        start, stop = int(ptr[row]), int(ptr[row + 1])
        target = int(cache["player_local"][row])
        target_index = start + target
        target_team = int(cache["target_team_raw"][row])
        valid_candidates = cache["candidate_team_valid"][start:stop].bool()
        same_team = valid_candidates & (
            cache["candidate_team_raw"][start:stop] == target_team
        )
        preserved = bool(same_team[target]) if 0 <= target < same_team.numel() else False
        eligible = bool(cache["player_mask"][row]) and bool(
            cache["team_mapping_valid"][row]
        ) and preserved
        if not eligible:
            continue
        local_scores = scores[start:stop]
        unmasked_order = local_scores.argsort(descending=True)
        masked_scores = local_scores.clone(); masked_scores[~same_team] = -torch.inf
        masked_order = masked_scores.argsort(descending=True)
        rows.append({
            "sample_id": sample_id,
            "match_id": int(cache["match_ids"][row]),
            "current_event_index": int(cache["current_event_indices"][row]),
            "source": source,
            "unmasked_rank": int(torch.nonzero(unmasked_order == target, as_tuple=False)[0]) + 1,
            "hard_rank": int(torch.nonzero(masked_order == target, as_tuple=False)[0]) + 1,
            "candidate_count": stop - start,
            "hard_candidate_count": int(same_team.sum()),
        })
    return pd.DataFrame(rows)
