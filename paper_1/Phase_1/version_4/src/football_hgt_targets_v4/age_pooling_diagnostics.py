"""Diagnostics for static shared/task-specific age-aware Event readout."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F

from .model import AgePoolingPartialL2HGT
from .training import _move_batch_to_device


TASKS = AgePoolingPartialL2HGT.AGE_TASKS
BUCKETS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80))


def _canonical_profiles(model: AgePoolingPartialL2HGT) -> dict[str, Any]:
    ages = torch.arange(80, device=model.pooling_embeddings.device)
    output: dict[str, Any] = {}
    distributions = {}
    for task in TASKS:
        scores = model.raw_age_scores(task, ages)
        relative = scores - scores[0]
        weights = torch.softmax(scores, dim=0)
        distributions[task] = weights
        output[task] = {
            "relative_score": relative.detach().cpu().tolist(),
            "canonical_weight": weights.detach().cpu().tolist(),
        }
    js = {}
    for left_index, left in enumerate(TASKS):
        for right in TASKS[left_index + 1:]:
            p, q = distributions[left], distributions[right]
            midpoint = 0.5 * (p + q)
            value = 0.5 * (
                torch.sum(p * (torch.log(p.clamp_min(1e-12)) - torch.log(midpoint)))
                + torch.sum(q * (torch.log(q.clamp_min(1e-12)) - torch.log(midpoint)))
            )
            js[f"{left}_{right}"] = float(value.detach())
    output["js_divergence"] = js
    return output


def _embedding_diagnostics(model: AgePoolingPartialL2HGT) -> dict[str, Any]:
    rows = model.pooling_embeddings
    output = {"rows": rows.detach().cpu().tolist(), "pairs": {}}
    if model.age_pooling_mode == "task":
        for left_index, left in enumerate(TASKS):
            for right_index in range(left_index + 1, len(TASKS)):
                right = TASKS[right_index]
                output["pairs"][f"{left}_{right}"] = {
                    "l2": float(torch.linalg.vector_norm(rows[left_index] - rows[right_index]).detach()),
                    "cosine": float(F.cosine_similarity(
                        rows[left_index].unsqueeze(0), rows[right_index].unsqueeze(0)
                    ).detach()),
                }
    return output


@torch.no_grad()
def age_pooling_snapshot(
    model: AgePoolingPartialL2HGT,
    batch: dict[str, Any],
) -> dict[str, Any]:
    """Small-batch profile stored every epoch."""
    graph = batch["graphs"]["f80"]
    return {
        "mode": model.age_pooling_mode,
        "source_age_mismatch_count": model.source_age_mismatch_count(graph),
        "canonical": _canonical_profiles(model),
        "embedding": _embedding_diagnostics(model),
    }


@torch.no_grad()
def collect_age_pooling_diagnostics(
    model: AgePoolingPartialL2HGT,
    loader: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Collect sample-normalized age-use summaries on a locked split."""
    was_training = model.training
    model.eval()
    values = {
        task: {
            "expected_age": [], "age_std": [], "entropy": [], "ess": [],
            "n50": [], "n80": [], "n90": [], "bucket_mass": [],
        }
        for task in TASKS
    }
    mismatch_count = 0
    sample_count = 0
    for raw_batch in loader:
        batch = _move_batch_to_device(raw_batch, device)
        graph = batch["graphs"]["f80"]
        mismatch_count += model.source_age_mismatch_count(graph)
        ptr = graph["event"].ptr
        for task in TASKS:
            ages, weights = model.age_weights(graph, task)
            ages = ages.detach().cpu()
            weights = weights.detach().cpu()
            ptr_cpu = ptr.detach().cpu()
            for sample_index in range(int(ptr_cpu.numel() - 1)):
                start, stop = int(ptr_cpu[sample_index]), int(ptr_cpu[sample_index + 1])
                sample_age = ages[start:stop].float()
                sample_weight = weights[start:stop]
                expected = torch.sum(sample_age * sample_weight)
                variance = torch.sum((sample_age - expected).square() * sample_weight)
                entropy = -torch.sum(sample_weight * torch.log(sample_weight.clamp_min(1e-12)))
                normalized_entropy = entropy / math.log(max(stop - start, 2))
                recent_order = torch.argsort(sample_age)
                cumulative = torch.cumsum(sample_weight[recent_order], dim=0)
                metrics = values[task]
                metrics["expected_age"].append(float(expected))
                metrics["age_std"].append(float(torch.sqrt(variance.clamp_min(0))))
                metrics["entropy"].append(float(normalized_entropy))
                metrics["ess"].append(float(1.0 / sample_weight.square().sum().clamp_min(1e-12)))
                for threshold, name in ((0.5, "n50"), (0.8, "n80"), (0.9, "n90")):
                    metrics[name].append(int(torch.searchsorted(
                        cumulative, torch.tensor(threshold)
                    ).item()) + 1)
                metrics["bucket_mass"].append([
                    float(sample_weight[(sample_age >= low) & (sample_age < high)].sum())
                    for low, high in BUCKETS
                ])
        sample_count += int(ptr.numel() - 1)

    summary: dict[str, Any] = {}
    for task, metrics in values.items():
        summary[task] = {}
        for name, entries in metrics.items():
            tensor = torch.tensor(entries, dtype=torch.float32)
            summary[task][name] = (
                {"mean": tensor.mean(dim=0).tolist(), "std": tensor.std(dim=0, unbiased=False).tolist()}
                if name == "bucket_mass"
                else {"mean": float(tensor.mean()), "std": float(tensor.std(unbiased=False))}
            )
    model.train(was_training)
    return {
        "mode": model.age_pooling_mode,
        "samples": sample_count,
        "source_age_mismatch_count": mismatch_count,
        "bucket_boundaries": BUCKETS,
        "canonical": _canonical_profiles(model),
        "embedding": _embedding_diagnostics(model),
        "sample_summary": summary,
    }
