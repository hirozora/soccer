"""Diagnostics for low-rank task/age propagation residuals."""

from __future__ import annotations

from typing import Any

import torch

from .age_propagation import PROPAGATION_TASKS
from .model import AgePropagationPartialL2HGT


AGE_BUCKETS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80))


def _profile(
    model: AgePropagationPartialL2HGT, layer: int, task: str
) -> dict[str, Any]:
    device = next(model.parameters()).device
    ages = torch.arange(80, device=device)
    gates = model.gate_profile(layer, task, ages)
    normalized = gates / gates.sum().clamp_min(1e-12)
    expected = (normalized * ages).sum()
    entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum()
    entropy = entropy / torch.log(torch.tensor(80.0, device=device))
    ess = 1.0 / normalized.square().sum().clamp_min(1e-12)
    return {
        "gate": gates.detach().cpu().tolist(),
        "relative_gate": (gates - gates[0]).detach().cpu().tolist(),
        "normalized_gate": normalized.detach().cpu().tolist(),
        "expected_age": float(expected.detach().cpu()),
        "normalized_entropy": float(entropy.detach().cpu()),
        "effective_sample_size": float(ess.detach().cpu()),
        "bucket_gate_mean": {
            f"{start}_{stop - 1}": float(gates[start:stop].mean().detach().cpu())
            for start, stop in AGE_BUCKETS
        },
    }


def _embedding_diagnostics(model: AgePropagationPartialL2HGT) -> dict[str, Any]:
    if model.propagation_gate is None:
        return {}
    embedding = model.propagation_gate.pooling_embeddings.detach()
    if embedding.shape[0] == 1:
        return {"rows": embedding.cpu().tolist(), "pairs": {}}
    output: dict[str, Any] = {"rows": embedding.cpu().tolist(), "pairs": {}}
    for left_index, left in enumerate(PROPAGATION_TASKS):
        for right_index in range(left_index + 1, len(PROPAGATION_TASKS)):
            right = PROPAGATION_TASKS[right_index]
            left_value = embedding[left_index]
            right_value = embedding[right_index]
            denominator = left_value.norm() * right_value.norm()
            output["pairs"][f"{left}_{right}"] = {
                "distance": float((left_value - right_value).norm().cpu()),
                "cosine": float(
                    ((left_value * right_value).sum() / denominator.clamp_min(1e-12)).cpu()
                ),
            }
    return output


@torch.no_grad()
def propagation_snapshot(
    model: AgePropagationPartialL2HGT,
    batch: dict[str, Any],
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    for layer in model.propagation_residuals:
        layer.capture_diagnostics = True
    model(batch)
    tasks = PROPAGATION_TASKS if model.propagation_mode == "task" else ("event",)
    canonical = {
        f"layer{layer + 1}": {
            task: _profile(model, layer, task) for task in tasks
        }
        for layer in range(2)
    }
    output = {
        "mode": model.propagation_mode,
        "source_age_mismatch_count": model.source_age_mismatch_count(
            batch["graphs"]["f80"]
        ),
        "canonical": canonical,
        "embedding": _embedding_diagnostics(model),
        "residual": model.last_propagation_stats,
        "forward_calls": [
            layer.forward_calls for layer in model.propagation_residuals
        ],
    }
    model.train(was_training)
    for layer in model.propagation_residuals:
        layer.capture_diagnostics = False
    return output


def collect_propagation_diagnostics(
    model: AgePropagationPartialL2HGT,
    batch: dict[str, Any],
) -> dict[str, Any]:
    return propagation_snapshot(model, batch)
