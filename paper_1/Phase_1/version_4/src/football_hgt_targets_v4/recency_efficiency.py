"""Measured single-view and fused-view inference cost for recency controls."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from football_benchmark.protocol import ProtocolArtifacts

from .constants import CONFIRMATION_SEEDS
from .model import build_multiview_model, build_target_model
from .recency_control_study import ALL_CONFIGS, RECENCY_CONTROL_ROOT, validation_dir
from .training import TargetTrainingConfig, _loader, _move_batch_to_device


def _config_and_model(config_name: str, device: torch.device) -> tuple[TargetTrainingConfig, torch.nn.Module, ProtocolArtifacts]:
    seed = CONFIRMATION_SEEDS[0]
    checkpoint = validation_dir(config_name, seed) / "best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    values = dict(state["config"])
    for field in ("artifact_path", "sample_plan_path", "output_dir"):
        if values.get(field) is not None:
            values[field] = Path(values[field])
    config = TargetTrainingConfig(**values)
    config = replace(config, device=str(device), num_workers=2, evaluate_test=False, full_test=False)
    artifacts = ProtocolArtifacts.load(config.artifact_path)
    model = (
        build_multiview_model(artifacts, config.fusion_mode)
        if config.fusion_mode is not None
        else build_target_model(
            artifacts, config.task, config.method, config.joint_methods,
            graph_variant=config.graph_variant,
            possession_topology=config.possession_topology,
            possession_feature_level=config.possession_feature_level,
        )
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return config, model, artifacts


def _graphs(batch: dict[str, Any]) -> list[Any]:
    return list(batch["graphs"].values()) if "graphs" in batch else [batch["graph"]]


def _batch_counts(batch: dict[str, Any]) -> tuple[int, int, int]:
    events = nodes = edges = 0
    for graph in _graphs(batch):
        events += int(graph["Event"].num_nodes)
        nodes += sum(int(graph[node_type].num_nodes) for node_type in graph.node_types)
        edges += sum(int(graph[edge_type].edge_index.shape[1]) for edge_type in graph.edge_types)
    return events, nodes, edges


@torch.inference_mode()
def benchmark_efficiency(device_name: str = "cuda:0", batches: int = 20) -> Path:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("Efficiency benchmark requires CUDA")
    rows = []
    for config_name in ALL_CONFIGS:
        config, model, artifacts = _config_and_model(config_name, device)
        loader = _loader("validation", config, artifacts, False)
        iterator = iter(loader)
        raw_batches = []
        count_events = count_nodes = count_edges = count_samples = 0
        collate_started = time.perf_counter()
        for _ in range(min(batches, len(loader))):
            raw = next(iterator)
            raw_batches.append(raw)
            event_count, node_count, edge_count = _batch_counts(raw)
            count_events += event_count
            count_nodes += node_count
            count_edges += edge_count
            count_samples += len(raw["sample_ids"])
        collate_seconds = time.perf_counter() - collate_started
        first = _move_batch_to_device(raw_batches[0], device)
        for _ in range(5):
            model(first)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(20):
            model(first)
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - started
        e2e_started = time.perf_counter()
        for raw in raw_batches:
            model(_move_batch_to_device(raw, device))
        torch.cuda.synchronize(device)
        e2e_seconds = time.perf_counter() - e2e_started + collate_seconds
        rows.append({
            "config": config_name,
            "encoded_views": len(_graphs(raw_batches[0])),
            "samples_measured": count_samples,
            "events_per_sample": count_events / count_samples,
            "nodes_per_sample": count_nodes / count_samples,
            "edges_per_sample": count_edges / count_samples,
            "forward_ms_per_batch": 1000.0 * forward_seconds / 20.0,
            "forward_samples_per_second": 20.0 * len(first["sample_ids"]) / forward_seconds,
            "e2e_samples_per_second": count_samples / e2e_seconds,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        })
        del model, loader, raw_batches, first
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame.config == "f80"].iloc[0]
    for column in ("events_per_sample", "nodes_per_sample", "edges_per_sample", "forward_ms_per_batch", "peak_cuda_memory_bytes"):
        frame[f"{column}_relative_to_f80"] = frame[column] / baseline[column]
    output = RECENCY_CONTROL_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "efficiency.csv", index=False)
    metadata = {
        "device": str(device),
        "seed_checkpoint": int(CONFIRMATION_SEEDS[0]),
        "batch_size": int(config.micro_batch_size),
        "timed_batches": batches,
        "fusion_cost_contract": {
            "semantic_sf_b": "Cost(P1) + Cost(P2)",
            "recency_sf_b": "Cost(LP1) + Cost(LP2)",
        },
    }
    (output / "efficiency_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output / "efficiency.csv"
