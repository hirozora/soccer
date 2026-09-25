"""Frozen Player-posterior conditions and lightweight downstream probes."""

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
from football_benchmark.constants import PITCH_LENGTH_METERS, PITCH_WIDTH_METERS
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F

from .constants import CONFIRMATION_SEEDS, POSSESSION_GRAPH_ROOT
from .oracle_dependency import index_loader, load_oracle_cache
from .oracle_dependency_study import cache_path as oracle_cache_path
from .partial_sharing_study import training_dir as partial_training_dir
from .player_posterior_study import (
    CONDITIONS,
    EPSILON,
    FAMILIES,
    TEAM_PRIOR_LAMBDA,
    condition_cache_path,
    decision_path,
)
from .team_candidate_prior import load_team_candidate_cache
from .training import set_seed


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _graph_paths() -> dict[int, Path]:
    index = pd.read_csv(POSSESSION_GRAPH_ROOT / "metadata/match_index.csv")
    return {
        int(row.match_id): POSSESSION_GRAPH_ROOT / str(row.graph_path)
        for row in index.itertuples()
    }


def _anchor_metadata(cache: dict[str, Any]) -> dict[str, torch.Tensor]:
    length = len(cache["sample_ids"])
    result = {
        "anchor_team_raw": torch.full((length,), -1, dtype=torch.long),
        "event_role": torch.zeros(length, dtype=torch.long),
        "control_state": torch.zeros(length, dtype=torch.long),
        "switch_confirmed": torch.zeros(length, dtype=torch.bool),
        "target_seen": torch.zeros(length, dtype=torch.bool),
    }
    paths = _graph_paths()
    match_ids = cache["match_ids"].long()
    event_indices = cache["current_event_indices"].long()
    for match_id in torch.unique(match_ids).tolist():
        rows = torch.nonzero(match_ids == match_id, as_tuple=False).flatten()
        graph = torch.load(paths[int(match_id)], map_location="cpu", weights_only=False)
        event = graph["node_stores"]["event"]
        team = graph["node_stores"]["team"]
        for row in rows.tolist():
            anchor = int(event_indices[row])
            if not 0 <= anchor < int(event["num_nodes"]):
                raise RuntimeError(f"Invalid anchor {anchor} for match {match_id}")
            team_local = int(event["team_local_index"][anchor])
            result["anchor_team_raw"][row] = int(team["raw_id"][team_local])
            result["event_role"][row] = int(event["event_role_index"][anchor])
            result["control_state"][row] = int(event["control_state_after_index"][anchor])
            result["switch_confirmed"][row] = bool(event["switch_confirmed"][anchor])
            player_local = int(cache["player_local"][row])
            result["target_seen"][row] = bool(
                (event["player_local_index"][: anchor + 1] == player_local).any()
            )
    if bool((result["anchor_team_raw"] <= 0).any()):
        raise RuntimeError("Failed to recover a causal anchor Team")
    return result


def _team_logits(cache: dict[str, Any], seed: int) -> torch.Tensor:
    checkpoint = partial_training_dir(seed) / "best_guarded_core.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    return F.linear(
        cache["main_context"].float(),
        state["team_actor_head.weight"].float(),
        state["team_actor_head.bias"].float(),
    )


def _expected_states(
    cache: dict[str, Any], team_logits: torch.Tensor, anchor_team_raw: torch.Tensor
) -> dict[str, torch.Tensor]:
    length = len(cache["sample_ids"])
    base_expected = torch.zeros((length, 64), dtype=torch.float32)
    team_expected = torch.zeros_like(base_expected)
    true_state = torch.zeros_like(base_expected)
    diagnostic_names = (
        "base_true_probability", "team_true_probability", "base_entropy",
        "team_entropy", "base_ess", "team_ess", "base_norm", "team_norm",
        "base_true_cosine", "team_true_cosine", "base_true_l2", "team_true_l2",
        "base_true_rank", "team_true_rank",
    )
    diagnostics = {name: torch.zeros(length, dtype=torch.float32) for name in diagnostic_names}
    probabilities = team_logits.softmax(dim=-1)
    ptr = cache["candidate_ptr"]
    for row in range(length):
        start, stop = int(ptr[row]), int(ptr[row + 1])
        scores = cache["candidate_scores"][start:stop].float()
        states = cache["candidate_states"][start:stop].float()
        candidate_team = cache["candidate_team_raw"][start:stop].long()
        valid = cache["candidate_team_valid"][start:stop].bool()
        same = candidate_team == int(anchor_team_raw[row])
        prior = torch.where(same, probabilities[row, 1], probabilities[row, 0])
        contribution = torch.zeros_like(scores)
        contribution[valid] = (
            torch.log(prior[valid] + EPSILON) - math.log(0.5)
        ) * TEAM_PRIOR_LAMBDA
        base_post = scores.softmax(dim=0)
        team_post = (scores + contribution).softmax(dim=0)
        base_expected[row] = (base_post.unsqueeze(-1) * states).sum(dim=0)
        team_expected[row] = (team_post.unsqueeze(-1) * states).sum(dim=0)
        target = int(cache["player_local"][row])
        true_state[row] = states[target]
        diagnostics["base_true_probability"][row] = base_post[target]
        diagnostics["team_true_probability"][row] = team_post[target]
        diagnostics["base_true_rank"][row] = int(
            torch.nonzero(base_post.argsort(descending=True) == target, as_tuple=False)[0]
        ) + 1
        diagnostics["team_true_rank"][row] = int(
            torch.nonzero(team_post.argsort(descending=True) == target, as_tuple=False)[0]
        ) + 1
        diagnostics["base_entropy"][row] = -(base_post * base_post.clamp_min(1e-12).log()).sum()
        diagnostics["team_entropy"][row] = -(team_post * team_post.clamp_min(1e-12).log()).sum()
        diagnostics["base_ess"][row] = 1.0 / base_post.square().sum()
        diagnostics["team_ess"][row] = 1.0 / team_post.square().sum()
        diagnostics["base_norm"][row] = torch.linalg.vector_norm(base_expected[row])
        diagnostics["team_norm"][row] = torch.linalg.vector_norm(team_expected[row])
        diagnostics["base_true_cosine"][row] = F.cosine_similarity(
            base_expected[row].unsqueeze(0), true_state[row].unsqueeze(0)
        )[0]
        diagnostics["team_true_cosine"][row] = F.cosine_similarity(
            team_expected[row].unsqueeze(0), true_state[row].unsqueeze(0)
        )[0]
        diagnostics["base_true_l2"][row] = torch.linalg.vector_norm(base_expected[row] - true_state[row])
        diagnostics["team_true_l2"][row] = torch.linalg.vector_norm(team_expected[row] - true_state[row])
    return {
        "base_post_condition": base_expected,
        "team_post_condition": team_expected,
        "true_player_state": true_state,
        **diagnostics,
    }


def _verify_prior_cache(
    seed: int, split: str, oracle: dict[str, Any], team_logits: torch.Tensor,
    anchor_team_raw: torch.Tensor,
) -> None:
    if split not in {"validation", "test"}:
        return
    reference = load_team_candidate_cache(seed, split)
    if oracle["sample_ids"] != reference["sample_ids"]:
        raise RuntimeError("Oracle and Team-prior cache samples differ")
    tensor_pairs = (
        (oracle["candidate_ptr"], reference["candidate_ptr"], "candidate ptr", 0.0),
        (oracle["candidate_scores"], reference["candidate_scores"], "Player scores", 1e-5),
        (team_logits, reference["team_logits"], "Team logits", 1e-5),
        (anchor_team_raw, reference["anchor_team_raw"], "anchor Team", 0.0),
    )
    for actual, expected, name, tolerance in tensor_pairs:
        if actual.dtype.is_floating_point:
            difference = float((actual - expected).abs().max())
            if difference >= tolerance:
                raise RuntimeError(f"{name} differs by {difference}")
        elif not torch.equal(actual, expected):
            raise RuntimeError(f"{name} differs")

    probabilities = team_logits.softmax(-1)
    reference_probabilities = reference["team_logits"].softmax(-1)
    ptr = oracle["candidate_ptr"]
    for row in torch.nonzero(oracle["player_mask"].bool(), as_tuple=False).flatten().tolist():
        start, stop = int(ptr[row]), int(ptr[row + 1])
        candidate_team = oracle["candidate_team_raw"][start:stop]
        valid = oracle["candidate_team_valid"][start:stop].bool()
        same = candidate_team == int(anchor_team_raw[row])
        prior = torch.where(same, probabilities[row, 1], probabilities[row, 0])
        contribution = torch.zeros(stop - start)
        contribution[valid] = TEAM_PRIOR_LAMBDA * (
            torch.log(prior[valid] + EPSILON) - math.log(0.5)
        )
        reference_prior = torch.where(
            reference["candidate_same_anchor"][start:stop].bool(),
            reference_probabilities[row, 1], reference_probabilities[row, 0],
        )
        reference_contribution = torch.zeros(stop - start)
        reference_valid = reference["candidate_team_valid"][start:stop].bool()
        reference_contribution[reference_valid] = TEAM_PRIOR_LAMBDA * (
            torch.log(reference_prior[reference_valid] + EPSILON) - math.log(0.5)
        )
        target = int(oracle["player_local"][row])
        actual_order = (oracle["candidate_scores"][start:stop] + contribution).argsort(descending=True)
        expected_order = (
            reference["candidate_scores"][start:stop] + reference_contribution
        ).argsort(descending=True)
        actual_rank = int(torch.nonzero(actual_order == target, as_tuple=False)[0])
        expected_rank = int(torch.nonzero(expected_order == target, as_tuple=False)[0])
        if actual_rank != expected_rank or int(actual_order[0]) != int(expected_order[0]):
            raise RuntimeError(f"TeamPost ranking differs for row {row}")


def build_condition_cache(seed: int, split: str) -> Path:
    if seed not in CONFIRMATION_SEEDS or split not in {"train", "validation", "test"}:
        raise ValueError("Unsupported seed or split")
    if split == "test" and not decision_path().exists():
        raise RuntimeError("Validation dependency decision required before test cache")
    oracle = load_oracle_cache(seed, split)
    metadata = _anchor_metadata(oracle)
    logits = _team_logits(oracle, seed)
    _verify_prior_cache(seed, split, oracle, logits, metadata["anchor_team_raw"])
    if split in {"validation", "test"}:
        # Reuse the already validated GPU-produced logits so downstream
        # conditions exactly match the locked Team-prior experiment.
        logits = load_team_candidate_cache(seed, split)["team_logits"].clone()
    expected = _expected_states(oracle, logits, metadata["anchor_team_raw"])
    keep = (
        "main_context", "event_true", "position_true", "position_mask", "zone_true",
        "player_mask", "team_true", "match_ids", "current_event_indices",
    )
    payload = {name: oracle[name].clone() for name in keep}
    payload.update(metadata)
    payload.update(expected)
    payload["candidate_count"] = (
        oracle["candidate_ptr"][1:] - oracle["candidate_ptr"][:-1]
    )
    payload.update(
        sample_ids=list(oracle["sample_ids"]), seed=seed, split=split,
        team_logits=logits, source_oracle_cache=str(oracle_cache_path(seed, split)),
        source_model_checkpoint=str(partial_training_dir(seed) / "best_guarded_core.pt"),
    )
    output = condition_cache_path(seed, split)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output


def load_condition_cache(seed: int, split: str) -> dict[str, Any]:
    return torch.load(condition_cache_path(seed, split), map_location="cpu", weights_only=False)


def condition_values(cache: dict[str, Any], condition: str, rows: torch.Tensor) -> torch.Tensor:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition == "null":
        return torch.zeros((rows.numel(), 64), dtype=torch.float32)
    return cache[f"{condition}_condition"][rows].float()


class PosteriorProbeHead(nn.Module):
    def __init__(self, family: str, dropout: float = 0.1) -> None:
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(family)
        output_dim = 10 if family == "event" else 2
        self.family = family
        self.network = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, output_dim)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.network(values)
        return output if self.family == "event" else torch.sigmoid(output)


def head_inputs(cache: dict[str, Any], condition: str, rows: torch.Tensor) -> torch.Tensor:
    return torch.cat((cache["main_context"][rows].float(), condition_values(cache, condition, rows)), dim=-1)


def probe_loss(
    family: str, prediction: torch.Tensor, cache: dict[str, Any], rows: torch.Tensor
) -> torch.Tensor:
    if family == "event":
        return F.cross_entropy(prediction, cache["event_true"][rows].to(prediction.device))
    mask = cache["position_mask"][rows].bool().to(prediction.device)
    target = cache["position_true"][rows].to(prediction.device)
    values = F.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
    return values[mask].mean() if mask.any() else values.sum() * 0.0


def _distances(frame: pd.DataFrame) -> np.ndarray:
    return np.sqrt(
        ((frame.position_pred_x - frame.position_true_x) * PITCH_LENGTH_METERS) ** 2
        + ((frame.position_pred_y - frame.position_true_y) * PITCH_WIDTH_METERS) ** 2
    )


def family_metrics(family: str, frame: pd.DataFrame) -> dict[str, Any]:
    if family == "event":
        known = frame[frame.player_mask.astype(bool)]
        return {
            "samples": len(frame),
            "accuracy": float(accuracy_score(frame.event_true, frame.event_pred)),
            "macro_f1": float(f1_score(frame.event_true, frame.event_pred, labels=range(10), average="macro", zero_division=0)),
            "known_player_samples": len(known),
            "known_player_accuracy": float(accuracy_score(known.event_true, known.event_pred)),
            "known_player_macro_f1": float(f1_score(known.event_true, known.event_pred, labels=range(10), average="macro", zero_division=0)),
        }
    active = frame[frame.position_mask.astype(bool)].copy()
    active["distance_m"] = _distances(active)
    known = active[active.player_mask.astype(bool)]
    return {
        "samples": len(active),
        "distance_mae_m": float(active.distance_m.mean()),
        "distance_median_m": float(active.distance_m.median()),
        "known_player_samples": len(known),
        "known_player_distance_mae_m": float(known.distance_m.mean()),
        "known_player_distance_median_m": float(known.distance_m.median()),
    }


@torch.no_grad()
def evaluate_head(
    model: PosteriorProbeHead, family: str, condition: str, cache: dict[str, Any],
    device: torch.device, batch_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    frames, loss_sum, count = [], 0.0, 0
    for rows in index_loader(len(cache["sample_ids"]), batch_size, False, int(cache["seed"])):
        prediction = model(head_inputs(cache, condition, rows).to(device))
        loss = probe_loss(family, prediction, cache, rows)
        prediction = prediction.cpu()
        frame = pd.DataFrame({
            "sample_id": [cache["sample_ids"][index] for index in rows.tolist()],
            "match_id": cache["match_ids"][rows].numpy(),
            "current_event_index": cache["current_event_indices"][rows].numpy(),
            "player_mask": cache["player_mask"][rows].numpy(),
            "event_true": cache["event_true"][rows].numpy(),
            "zone_true": cache["zone_true"][rows].numpy(),
            "event_role": cache["event_role"][rows].numpy(),
            "control_state": cache["control_state"][rows].numpy(),
            "switch_confirmed": cache["switch_confirmed"][rows].numpy(),
            "target_seen": cache["target_seen"][rows].numpy(),
        })
        if family == "event":
            frame["event_pred"] = prediction.argmax(-1).numpy()
        else:
            frame["position_mask"] = cache["position_mask"][rows].numpy()
            frame["position_true_x"] = cache["position_true"][rows, 0].numpy()
            frame["position_true_y"] = cache["position_true"][rows, 1].numpy()
            frame["position_pred_x"] = prediction[:, 0].numpy()
            frame["position_pred_y"] = prediction[:, 1].numpy()
        frames.append(frame)
        loss_sum += float(loss) * len(rows)
        count += len(rows)
    frame = pd.concat(frames, ignore_index=True)
    result = family_metrics(family, frame)
    result["loss"] = loss_sum / max(count, 1)
    return result, frame


@dataclass(frozen=True)
class PosteriorProbeConfig:
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
    result: dict[str, Any] = {}
    for name, value in cache.items():
        if name == "sample_ids":
            result[name] = value[:limit]
        elif isinstance(value, torch.Tensor) and value.shape[:1] == (length,):
            result[name] = value[:limit]
        else:
            result[name] = value
    return result


def _selection_value(family: str, values: dict[str, Any]) -> tuple[float, ...]:
    if family == "event":
        return (
            -values["known_player_macro_f1"],
            -values["known_player_accuracy"],
            values["loss"],
        )
    return (values["known_player_distance_mae_m"], values["loss"])


def train_probe(config: PosteriorProbeConfig) -> dict[str, Any]:
    config.validate()
    set_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    train = _limited(load_condition_cache(config.seed, "train"), config.max_train_samples)
    validation = _limited(load_condition_cache(config.seed, "validation"), config.max_validation_samples)
    model = PosteriorProbeHead(config.family).to(device)
    initial_hash = _state_hash(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = index_loader(len(train["sample_ids"]), config.batch_size, True, config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.output_dir / "best.pt"
    history, best, stale = [], None, 0
    started = time.monotonic()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for rows in loader:
            prediction = model(head_inputs(train, config.condition, rows).to(device))
            loss = probe_loss(config.family, prediction, train, rows)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(rows)
            count += len(rows)
        values, _ = evaluate_head(
            model, config.family, config.condition, validation, device, config.batch_size
        )
        selection = _selection_value(config.family, values)
        history.append({
            "epoch": epoch, "train_loss": loss_sum / max(count, 1),
            "validation": values, "selection_value": selection,
        })
        if best is None or selection < best:
            best, stale = selection, 0
            torch.save({
                "model": model.state_dict(), "config": asdict(config), "epoch": epoch,
                "selection_value": selection, "initial_hash": initial_hash,
            }, checkpoint)
        else:
            stale += 1
        if stale >= config.patience:
            break
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    values, frame = evaluate_head(
        model, config.family, config.condition, validation, device, config.batch_size
    )
    frame.to_parquet(config.output_dir / "validation_predictions.parquet", index=False)
    result = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "best_epoch": state["epoch"], "validation": values, "test": None,
        "history": history, "elapsed_seconds": time.monotonic() - started,
        "initial_hash": initial_hash, "test_accessed": False,
    }
    (config.output_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def evaluate_test_probe(
    config: PosteriorProbeConfig, checkpoint: Path
) -> dict[str, Any]:
    if not decision_path().exists():
        raise RuntimeError("Validation dependency decision required")
    device = torch.device(config.device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = PosteriorProbeHead(config.family).to(device)
    model.load_state_dict(state["model"])
    cache = load_condition_cache(config.seed, "test")
    values, frame = evaluate_head(
        model, config.family, config.condition, cache, device, config.batch_size
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(config.output_dir / "test_predictions.parquet", index=False)
    result = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "best_epoch": state["epoch"], "validation": None, "test": values,
        "source_checkpoint": str(checkpoint), "test_accessed": True,
    }
    (config.output_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result
