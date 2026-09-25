"""Frozen Partial-L2 caches and Team-aware candidate score transforms."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from football_benchmark.protocol import ProtocolArtifacts

from .constants import CONFIRMATION_SEEDS, FEASIBILITY_ARTIFACT, POSSESSION_GRAPH_ROOT
from .five_task_training import FiveTaskTrainingConfig, _loader
from .model import build_partial_l2_model
from .oracle_dependency import load_roster_team_map
from .team_candidate_prior_study import EPSILON, cache_path, lock_path, source_checkpoint
from .training import _move_batch_to_device, set_seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_loader(seed: int, split: str, artifacts: ProtocolArtifacts):
    config = FiveTaskTrainingConfig(
        mode="five_f80",
        output_dir=Path("."),
        seed=seed,
        device="cpu",
        batch_size=256,
        num_workers=2,
        artifact_path=FEASIBILITY_ARTIFACT,
        full_test=split == "test",
        evaluate_test=split == "test",
    )
    return _loader(split, config, artifacts, False)


def _anchor_team_raw_ids(graph: Any, artifacts: ProtocolArtifacts) -> torch.Tensor:
    """Read the visible anchor Event's Team through the Event->Team edge."""

    anchors = graph["event"].ptr[1:] - 1
    edges = graph[("event", "performed_by_team", "team")].edge_index
    event_to_team = torch.full(
        (int(graph["event"].num_nodes),), -1, dtype=torch.long, device=edges.device
    )
    event_to_team[edges[0]] = edges[1]
    team_nodes = event_to_team[anchors]
    if bool((team_nodes < 0).any()):
        raise RuntimeError("An anchor Event has no Event->Team relation")
    vocab = graph["team"].vocab_index[team_nodes].detach().cpu().long()
    inverse = {int(index): int(raw) for raw, index in artifacts.team_to_index.items()}
    raw = torch.tensor([inverse.get(int(value), -1) for value in vocab], dtype=torch.long)
    if bool((raw <= 0).any()):
        raise RuntimeError("An anchor Team cannot be mapped back to its raw ID")
    return raw


@torch.no_grad()
def build_team_candidate_cache(seed: int, split: str, device_name: str) -> Path:
    if seed not in CONFIRMATION_SEEDS or split not in {"validation", "test"}:
        raise ValueError("Unsupported seed or split")
    if split == "test" and not lock_path().exists():
        raise RuntimeError("Validation method lock is required before test cache")

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
    loader = _cache_loader(seed, split, artifacts)
    roster = load_roster_team_map()

    sample_names = (
        "match_ids", "current_event_indices", "event_true", "team_true",
        "player_local", "player_mask", "team_logits", "anchor_team_raw",
        "event_role", "control_state", "switch_confirmed",
    )
    candidate_names = (
        "candidate_scores", "candidate_raw", "candidate_team_raw",
        "candidate_team_valid", "candidate_same_anchor",
    )
    samples: dict[str, list[torch.Tensor]] = {name: [] for name in sample_names}
    candidates: dict[str, list[torch.Tensor]] = {name: [] for name in candidate_names}
    counts: list[torch.Tensor] = []
    sample_ids: list[str] = []

    for raw_batch in loader:
        batch = _move_batch_to_device(raw_batch, device)
        predictions = model(batch)
        graph = batch["graphs"]["f80"]
        targets = batch["targets"]
        anchor_team_raw = _anchor_team_raw_ids(graph, artifacts)
        raw_ids = graph["player"].raw_id.detach().cpu().long()
        ptr = graph["player"].ptr.detach().cpu().long()
        match_ids = batch["match_ids"].detach().cpu().long()
        candidate_team = torch.full_like(raw_ids, -1)
        candidate_valid = torch.zeros_like(raw_ids, dtype=torch.bool)
        candidate_same = torch.zeros_like(raw_ids, dtype=torch.bool)
        for row, match_id in enumerate(match_ids.tolist()):
            mapping = roster.get(int(match_id))
            if mapping is None:
                raise RuntimeError(f"Missing pre-match roster for match {match_id}")
            start, stop = int(ptr[row]), int(ptr[row + 1])
            for index in range(start, stop):
                team_id = mapping.get(int(raw_ids[index]))
                if team_id is not None:
                    candidate_team[index] = int(team_id)
                    candidate_valid[index] = True
                    candidate_same[index] = int(team_id) == int(anchor_team_raw[row])

        anchors = graph["event"].ptr[1:] - 1
        values = {
            "match_ids": batch["match_ids"],
            "current_event_indices": batch["current_event_indices"],
            "event_true": targets["raw_event_10"],
            "team_true": targets["team_actor"],
            "player_local": targets["player_local"],
            "player_mask": targets["player_mask"].bool(),
            "team_logits": predictions["team_logits"],
            "anchor_team_raw": anchor_team_raw,
            "event_role": graph["event"].event_role_index[anchors],
            "control_state": graph["event"].control_state_after_index[anchors],
            "switch_confirmed": graph["event"].switch_confirmed[anchors],
        }
        for name, value in values.items():
            samples[name].append(value.detach().cpu())
        candidate_values = {
            "candidate_scores": predictions["player_scores"],
            "candidate_raw": raw_ids,
            "candidate_team_raw": candidate_team,
            "candidate_team_valid": candidate_valid,
            "candidate_same_anchor": candidate_same,
        }
        for name, value in candidate_values.items():
            candidates[name].append(value.detach().cpu())
        counts.append(ptr[1:] - ptr[:-1])
        sample_ids.extend(batch["sample_ids"])

    payload: dict[str, Any] = {name: torch.cat(values) for name, values in samples.items()}
    payload.update({name: torch.cat(values) for name, values in candidates.items()})
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
        graph_root=str(POSSESSION_GRAPH_ROOT),
    )
    torch.save(payload, output)
    return output


def load_team_candidate_cache(seed: int, split: str) -> dict[str, Any]:
    return torch.load(cache_path(seed, split), map_location="cpu", weights_only=False)


def centered_log_prior(cache: dict[str, Any]) -> torch.Tensor:
    probabilities = cache["team_logits"].softmax(dim=-1)
    counts = cache["candidate_ptr"][1:] - cache["candidate_ptr"][:-1]
    sample_index = torch.repeat_interleave(torch.arange(len(counts)), counts)
    same = cache["candidate_same_anchor"].bool()
    valid = cache["candidate_team_valid"].bool()
    prior = torch.where(same, probabilities[sample_index, 1], probabilities[sample_index, 0])
    contribution = torch.zeros_like(prior)
    contribution[valid] = torch.log(prior[valid] + EPSILON) - math.log(0.5)
    return contribution


def _rank(scores: torch.Tensor, target: int) -> tuple[int, float, float]:
    order = torch.argsort(scores, descending=True)
    rank = int(torch.nonzero(order == target, as_tuple=False)[0]) + 1
    top = float(scores[int(order[0])])
    second = float(scores[int(order[1])]) if order.numel() > 1 else top
    return rank, 1.0 / rank, top - second


def prediction_frame(
    cache: dict[str, Any], method: str, *, lambda_value: float = 0.0
) -> pd.DataFrame:
    if method not in {"base", "soft", "hard"}:
        raise ValueError(f"Unknown candidate-prior method {method!r}")
    scores = cache["candidate_scores"].float()
    if method == "soft":
        scores = scores + float(lambda_value) * centered_log_prior(cache)
    probabilities = cache["team_logits"].softmax(dim=-1)
    team_pred = probabilities.argmax(dim=-1)
    ptr = cache["candidate_ptr"]
    rows: list[dict[str, Any]] = []
    for row, sample_id in enumerate(cache["sample_ids"]):
        start, stop = int(ptr[row]), int(ptr[row + 1])
        target = int(cache["player_local"][row])
        local = scores[start:stop].clone()
        hard_applied = False
        target_excluded = False
        local_raw = cache["candidate_raw"][start:stop]
        local_valid = cache["candidate_team_valid"][start:stop].bool()
        # Every roster contains one structural raw_id=0 UNK candidate. It is
        # never a known-player target and is excluded by hard masking rather
        # than making every sample appear mapping-incomplete.
        mapping_complete = bool(local_valid[local_raw > 0].all())
        if method == "hard" and mapping_complete:
            hard_applied = True
            desired_same = bool(int(team_pred[row]) == 1)
            keep = local_valid & (
                cache["candidate_same_anchor"][start:stop].bool() == desired_same
            )
            if bool(keep.any()):
                target_excluded = not bool(keep[target])
                local[~keep] = -torch.inf
            else:
                hard_applied = False
        if target_excluded:
            rank, reciprocal, margin = stop - start + 1, 0.0, float("nan")
        else:
            rank, reciprocal, margin = _rank(local, target)
        rows.append({
            "seed": int(cache["seed"]),
            "sample_id": sample_id,
            "match_id": int(cache["match_ids"][row]),
            "current_event_index": int(cache["current_event_indices"][row]),
            "event_true": int(cache["event_true"][row]),
            "player_mask": bool(cache["player_mask"][row]),
            "candidate_count": stop - start,
            "player_rank": rank,
            "reciprocal_rank": reciprocal,
            "score_margin": margin,
            "team_true": int(cache["team_true"][row]),
            "team_pred": int(team_pred[row]),
            "team_correct": int(team_pred[row]) == int(cache["team_true"][row]),
            "team_same_probability": float(probabilities[row, 1]),
            "team_confidence": float(probabilities[row].max()),
            "mapping_complete": mapping_complete,
            "hard_applied": hard_applied,
            "target_excluded": target_excluded,
            "event_role": int(cache["event_role"][row]),
            "control_state": int(cache["control_state"][row]),
            "switch_confirmed": bool(cache["switch_confirmed"][row]),
            "lambda": float(lambda_value),
            "method": method,
        })
    return pd.DataFrame(rows)


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    active = frame[frame.player_mask.astype(bool)]
    ranks = active.player_rank.to_numpy()
    return {
        "valid_samples": int(len(active)),
        "top1": float(np.mean(ranks <= 1)),
        "top3": float(np.mean(ranks <= 3)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(active.reciprocal_rank.mean()),
        "mapping_complete": float(active.mapping_complete.mean()),
        "hard_applicability": float(active.hard_applied.mean()),
        "team_accuracy": float(active.team_correct.mean()),
    }
